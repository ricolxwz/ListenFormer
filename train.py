from utils_parallel import prepare_sub_folder, write_log, get_config, DictAverageMeter, get_scheduler
from data import get_data_loader
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import os
import sys
import tensorboardX
import shutil
from box import Box
from logger import create_logger
import time
import datetime
import git
import torch.distributed as dist
import torch.optim as optim
from networks.listener_generator import ListenerGenerator_trans_ca
from networks.speaker_generator import SpeakerGenerator

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--change_epoch', type=int, default=None)
    parser.add_argument('--config', type=str, default='configs/configs.yaml', help='Path to the config file.')
    parser.add_argument('--task', type=str, choices=['listener', 'speaker'], required=True)
    parser.add_argument('--time_size', type=int, required=True)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--anno_fn', type=str, default=None, help='specify anno file')
    parser.add_argument('--batch_size', type=int, default=None, help='specify batch size')
    parser.add_argument('--loss_weights', type=float, nargs=7, default=None, help='loss weights')
    parser.add_argument('--temporal_size', type=int, default=None, help='temporal size')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    return parser

def show_git_info():
    # get & show branch information and latest commit id
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch
    branch_name = branch.name
    latest_commit_sha = repo.head.object.hexsha
    print("Branch: {}".format(branch_name))
    print("Latest commit SHA: {}".format(latest_commit_sha))

if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()

    print("Use configs:", vars(opts))

    # Load experiment setting
    config = get_config(opts.config)  # 读取YAML配置文件
    config = Box(config)  # 使用Box将配置转换为支持属性访问的字典
    # 如果命令行中指定了batch_sze, anno_fn, loss_weights, temporal_size, max_epochs或者lr, 则使用命令行参数覆盖配置文件中的对应设置
    if opts.batch_size is not None:
        config.batch_size = opts.batch_size
    if opts.anno_fn is not None:
        config.anno_fn = opts.anno_fn
    if opts.loss_weights is not None:
        keys = sorted(list(config.loss_weights.keys()))
        new_dict = {k: opts.loss_weights[i] for i, k in enumerate(keys)}
        print('[WARNING] new loss weights:', new_dict)
        config.loss_weights = new_dict
    if opts.temporal_size is not None:
        config.temporal_size = opts.temporal_size
    if opts.max_epochs is not None:
        config.max_epochs = opts.max_epochs
    if opts.lr is not None:
        config.lr = opts.lr
    config.task = opts.task
    print('Using config:', config)

    # -- Init distributed
    # 在分布式训练中, rank表示当前进程的编号, world_size表示所有参与训练的进程总数, 例如, 如果你在使用4块GPU进行训练, 通常会为每块GPU分配一个进程, 此时, world_size为4, 而每个进程的rank分别为0, 1, 2, 3. rank表示每个进程在整个分布式系统(有好多台机器, 每台机器又有很多的GPU)的全局唯一标识, 而local_rank表示当前进程在所在的本台机器上分配到的GPU设备编号, 用于指定该进程使用哪块GPU. 当训练在多台机器上进行的时候, rank是跨机器的全局编号, 而local_rank只在单个机器内部起作用
    # main函数并不是只运行一次, 而是在每个进程都执行一次, 通常使用像torch.distributed.launch或者torchrun这样的工具启动程序的时候, 会自动创建多个独立的进程, 每个进程都运行同样的代码, 但是它们会收到不同的环境变量(例如RANK和local_rank). 这样, 每个进程都能调用torch.cuda_set_device(opts.LOCAL_RANK)来绑定到不同的GPU, 从而实现多GPU并行计算.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    opts.LOCAL_RANK = opts.local_rank
    torch.cuda.set_device(opts.LOCAL_RANK)  # 每个进程根据自己的local_rank绑定到对应的GPU
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)  # 使用NCCL作为通信后端, 通过环境变量获取初始化信息
    config.LOCAL_RANK = opts.LOCAL_RANK
    torch.distributed.barrier()  # 使用barrier进行进程同步, 确保所有进程都完成上述初始化后再继续执行后续代码

    seed = opts.SEED + dist.get_rank()  # 为每个进程生成一个独一无二的随机种子, 防止不同进程产生相同的随机数序列
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True  # 启动cudnn的benchmark模式, 该模式会在固定输入尺寸下自动寻找最优的卷积算法, 从而提升GPU计算性能, 加速训练过程

    output_directory = os.path.join(opts.output_path)
    checkpoint_directory = prepare_sub_folder(output_directory)
    logger = create_logger(output_directory, dist.get_rank(), os.path.basename(output_directory.strip('/')))  # 初始化日志记录器, 传入的参数包括输出目录, 当前进程的全局rank(通过dist.get_rank())获取以及输出目录的基本名称
    if dist.get_rank() == 0:  # 判断当前进程的rank是否为0, 这个进程比较特殊, 会负责一些全局性的工作, 避免重复操作
        shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # 如果当前进程是rank为0的进程, 则将配置文件opts.config复制到输出目录, 并命名为config.yaml

    # -- Init data loader
    loader = get_data_loader(config, opts.task, opts.time_size)  # 这里调用get_data_loader函数, 根据config, opts.task和opts.time_size等参数初始化数据加载器, 数据加载器负责从数据集加载数据, 并按照预定的批次和数据预处理方式提供数据, 供模型训练或者验证使用. 加载器返回的对象通常是一个可迭代对象, 每次迭代生成一个数据批次
    logger.info("Len loader: {}".format(len(loader)))

    # -- Init model
    if opts.task == 'listener':  # 如果是听者头部生成, 则使用ListenerGenerator_trans_ca类构建模型, 传入参数config
        model = ListenerGenerator_trans_ca(config)
    else:  # 如果是说者头部生成, 则使用SpeakerGenerator类构建模型, 传入参数config
        model = SpeakerGenerator(config)
    if opts.resume is not None:  # 如果opts.resume不为None, 说明需要从指定的检查点恢复模型参数
        print(f'resume model from {opts.resume}.')
        model.load_state_dict({k.replace('module.', '', 1): v for k, v in torch.load(opts.resume).items()})  # 这里的中是移除状态字典中每个键的前缀module., 这通常是因为模型保存的时候可能用了DataParallel或DistributedDataParallel封装, 导致参数名带有module.前缀, 而恢复的时候需要和模型实际参数名匹配
    model = model.cuda()  # 将构建好的模型移动到GPU上运行, 调用model.cuda()之后, 模型会被分配到之前通过torch.cuda_set_device设置的当前GPU上
    logger.info(str(model))

    # -- Init optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            betas=config.betas, weight_decay=config.weight_decay)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = model.module  # 将模型包装为DistributedDataParallel(DDP)模型, 使其能够在多个GPU上进行并行训练, 使用DDP之后, 每个进程会有一个模型的副本, 进程间通过通信机制同步梯度, 这样, 在每个进程中计算得到的梯度会在反向传播的时候自动聚合, 实现多GPU的并行训练. 默认情况下, DDP期望模型中所有参数在前向传播过程中都会被使用, 如果某些参数由于模型结构的条件性计算(例如存在if-else分支, 动态计算图等)而没有参与当前前向传播, DDP可能会报错. broadcase_buffers=False表示不自动同步模型中的缓冲区(例如batch normalization的统计信息)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计模型中所有需要梯度更新的参数总数. p.numel()返回每个参数tensor中的元素个数, 只有requires_grad为True的参数才会被计算在内.
    logger.info(f'number of params: {n_parameters}')

    lr_scheduler = get_scheduler(optimizer, config)

    meter = DictAverageMeter(*['iter_time', 'TOTAL_LOSS', 'loss_angle', 'loss_angle_spiky', 'loss_exp', 'loss_exp_spiky', 'loss_trans', 'loss_trans_spiky', 'loss_crop', 'loss_crop_spiky'])  # *[]的作用是将一个列表中的元素解包为独立的参数传递给函数或者构造函数
    iterations = 0
    max_iter = config.max_epochs * len(loader)  # 最大迭代次数

    for epoch in range(config.max_epochs):  # 遍历所有的epoch
        meter.reset()  # 重置用于记录和统计指标的对象, 使得每个epoch的统计数据都是独立的
        lr_scheduler.step()  # 更新学习率, 根据预设的策略调整学习率
        loader.sampler.set_epoch(epoch)  # 设置数据加载器中采样器的当前轮数, 通常用于分布式训练时保证数据分布的随机性
        model.train()  # 将模型设置为训练模式, 启用诸如dropout, BN等训练时特有的行为
        postfix = "%03d" % (epoch + 1)
        for id, data in enumerate(loader):  # 遍历数据加载器中的所有数据批次, 每个批次数据会依次被处理
            audio, driven_signal, init_signal, target_signal, lengths, _, attitude = data  # 将加载的批次数据拆分为多个变量. 这里使用的下划线_来忽略数据中的一个字段

            # 对每个tensor调用.cuda()方法, 将数据从CPU转移到GPU
            audio = audio.cuda().float()
            driven_signal = driven_signal.cuda().float()
            init_signal = init_signal.cuda().float()
            target_signal = target_signal.cuda().float()
            lengths = lengths.cuda().long()
            attitude = attitude.cuda().float()

            # 这里计算当前时间和开始时间的差值, 得到数据加载的时间, 这里貌似忘记定义start_data了
            elapse_data = time.time() - start_data

            # Main training code
            
            loss, loss_dict, pred_3dmm_dynam = model(
                audio,
                driven_signal,
                init_signal,
                lengths,
                epoch + 1,
                opts.change_epoch,
                target_signal,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update({
                'iter_time': {'val': elapse_iter},
                **loss_dict,
            })

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                etas = meter.iter_time.avg * (max_iter - iterations - 1)
                memory = torch.cuda.max_memory_allocated() / 1024. / 1024.
                loss_log = ''
                for loss_key, loss_item in loss_dict.items():
                    loss_log += f'{loss_key} {meter[loss_key].val:.4f}*{loss_item["weight"]} ({meter[loss_key].avg:.4f})\t'
                log = f'Train: [{iterations + 1}/{max_iter}]\teta {datetime.timedelta(seconds=int(etas))}\ttime {meter.iter_time.val:.2f} ({meter.iter_time.avg:.2f})\t'
                log += f'time_data {elapse_data:.2f}\t'
                log += loss_log
                log += f'mem {memory:.0f}MB'
                logger.info(log)
                write_log(log, output_directory)
            iterations += 1

        # Save network weights
        if (epoch+1) % 50 == 0:
            model_state_dict_name = os.path.join(checkpoint_directory, f'Epoch_{epoch + 1:03d}.bin')
            if dist.get_rank() == 0:
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_state_dict_name)
            logger.info(f'save epoch {epoch + 1} model to {model_state_dict_name}')

