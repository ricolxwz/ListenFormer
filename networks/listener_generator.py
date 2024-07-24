import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange, reduce, repeat
import torch.nn.init as init
import sys
from .stepwise_fusion import LSTMStepwiseFusion
from .transformer.decoder import TransformerDecoder
from .transformer.encoder import TransformerEncoder, ConformerEncoder
from .att import AttentionBlock
from .utils.mask import make_non_pad_mask, subsequent_mask, make_pad_mask
import math

class ListenerGenerator_trans_ca(nn.Module):
    def __init__(
        self,
        param,
    ):
        super().__init__()
        generator_cfg = param.model.generator
        self.generator_cfg = generator_cfg
        self.loss_weights = param.loss_weights
        self.loss_weights['TOTAL_LOSS'] = 1
        self.loss_names = ['TOTAL_LOSS'] + sorted(list(self.loss_weights.keys()))
        self.dynam_3dmm_split = [3, 64, 3, 3]


        self.driven_proj = nn.Linear(73, 256)

        self.ca = AttentionBlock(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1)

        self.encoder = TransformerEncoder(input_size=45, output_size=256, linear_units=1024, input_layer='linear', num_blocks=3, dropout_rate=0.1, attention_heads=4)
        self.decoder = TransformerDecoder(vocab_size=73, encoder_output_size=256, num_blocks=3, linear_units=1024, input_layer='embed', use_output_layer=True, dropout_rate=0.1, attention_heads=4)

    def predict(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target,
    ):

        driven = self.driven_proj(driven)
        encoder_out, encoder_mask, pos = self.encoder(audio, lengths)
        encoder_mask_bool = ~encoder_mask.squeeze(1).bool()
        
        encoder_out = self.ca(tgt=encoder_out,
                             memory=driven,
                             tgt_key_padding_mask=encoder_mask_bool,
                             memory_key_padding_mask=encoder_mask_bool,
                             pos=pos,
                             query_pos=pos) # [t*h*w, b, c]
        
        init = init.unsqueeze(1)

        ## teacher_forcing
        if False:
            frame_num = encoder_out.shape[1]
            bs = encoder_out.shape[0]
            decoder_in = init
            
            #print(frame_num)
            for i in range(frame_num):
                d_length = i + 1
                #print(d_length)
                d_length = torch.tensor([d_length]).cuda()
                d_length = d_length.repeat(bs)
                d_length = torch.minimum(d_length, lengths)

                #print(d_length.shape)
                #d_length = d_length.unsqueeze(0)
                new_out, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length)
                #print(new_out.shape)
                new_out = new_out[:,-1,:].unsqueeze(1) 
                decoder_in =  torch.cat((decoder_in, new_out), 1)
            decoder_out = decoder_in[:,1:,:].contiguous()

        else:
            if epoch <= change_epoch:
                target = torch.cat((init, target), 1)
            else:
                target = init.repeat(1, target.shape[1]+1,  1)
            decoder_out, _, _ = self.decoder(encoder_out, encoder_mask, target, lengths, epoch, change_epoch)

        
        return decoder_out

    def decode_period_sl_initlast_cat(
        self,
        audio,
        driven,
        init,
        lengths
    ):
        epoch = 500
        period = 90
        slide = 80 
        frame_nums = driven.shape[1]
        #rounds = (frame_nums - 1) // slide + 1
        init = init.unsqueeze(1)
        current = 0
        while (current + period) < frame_nums :
            audio_cat = audio[:, int(current):int(current + period), :]
            driven_cat = driven[:, int(current):int(current + period), :]
            lengths = period
            lengths = torch.tensor([lengths]).cuda()
            #driven_cat = torch.cat((audio_cat, driven_cat), 2) 
            #driven_cat = torch.cat((driven, driven_diff), 2)
            #driven_diff_cat = torch.diff(driven_cat, dim=1)
            #driven_diff_cat = F.pad(driven_diff_cat, (0,0,1,0), "constant", 0)
            #driven_cat = torch.cat((driven_cat, driven_diff_cat), 2)
            driven_cat = self.driven_proj(driven_cat)
            encoder_out, encoder_mask, pos = self.encoder(audio_cat, lengths)
            encoder_mask_bool = ~encoder_mask.squeeze(1).bool()
            encoder_out = self.ca(tgt=encoder_out,
                                 memory=driven_cat,
                                 tgt_key_padding_mask=encoder_mask_bool,
                                 memory_key_padding_mask=encoder_mask_bool,
                                 pos=pos,
                                 query_pos=pos) # [t*h*w, b, c] 
            frame_num = encoder_out.shape[1]
            if current == 0:
                decoder_in = init.repeat(1, frame_num,  1)
            else:
                decoder_in = decoder_out[:, -1, :].unsqueeze(1).repeat(1, frame_num,  1)
            #decoder_in = init_all
            #print(frame_num)
            d_length = frame_num
            #print(d_length)
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, change_epoch)
            if current == 0: 
                decoder_out = decoder_in
            else: 
                decoder_in_right = decoder_in[:, int(period-slide):, :]
                #decoder_out = torch.cat((decoder_out_left, decoder_ovlp, decoder_in_right), 1)
                decoder_out = torch.cat((decoder_out, decoder_in_right), 1)
            current = current + slide
        if current != 0:

            audio_cat = audio[:, -period:, :]
            driven_cat = driven[:, -period:, :]
            lengths = period
            lengths = torch.tensor([lengths]).cuda()
            #driven_cat = torch.cat((audio_cat, driven_cat), 2) 
            #driven_cat = audio_cat
            #encoder_out, encoder_mask = self.encoder(driven_cat, lengths)
            #driven_diff_cat = torch.diff(driven_cat, dim=1)
            #driven_diff_cat = F.pad(driven_diff_cat, (0,0,1,0), "constant", 0)
            #driven_cat = torch.cat((driven_cat, driven_diff_cat), 2)
            driven_cat = self.driven_proj(driven_cat)
            encoder_out, encoder_mask, pos = self.encoder(audio_cat, lengths)
            encoder_mask_bool = ~encoder_mask.squeeze(1).bool()
            encoder_out = self.ca(tgt=encoder_out,
                                 memory=driven_cat,
                                 tgt_key_padding_mask=encoder_mask_bool,
                                 memory_key_padding_mask=encoder_mask_bool,
                                 pos=pos,
                                 query_pos=pos) # [t*h*w, b, c]
            #tgt_length = lengths - 1
            frame_num = encoder_out.shape[1]
            if current == 0:
                decoder_in = init.repeat(1, frame_num,  1)
            else:
                decoder_in = decoder_out[:, -1, :].unsqueeze(1).repeat(1, frame_num,  1)
            #decoder_in = init_all
            #print(frame_num)
            d_length = frame_num
            #print(d_length)
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, change_epoch)
            #print(frame_num)

            ####
            decoder_in_right = decoder_in[:, int(period +decoder_out.shape[1] - frame_nums):, :]
            decoder_out = torch.cat((decoder_out, decoder_in_right), 1)

        else:
            audio_cat = audio
            driven_cat = driven 
            #driven_cat = torch.cat((audio_cat, driven_cat), 2) 
            #driven_cat = audio_cat
            #encoder_out, encoder_mask = self.encoder(driven_cat, lengths)
            #driven_diff_cat = torch.diff(driven_cat, dim=1)
            #driven_diff_cat = F.pad(driven_diff_cat, (0,0,1,0), "constant", 0)
            #driven_cat = torch.cat((driven_cat, driven_diff_cat), 2)
            driven_cat = self.driven_proj(driven_cat)
            encoder_out, encoder_mask, pos = self.encoder(audio_cat, lengths)
            encoder_mask_bool = ~encoder_mask.squeeze(1).bool()
            encoder_out = self.ca(tgt=encoder_out,
                                 memory=driven_cat,
                                 tgt_key_padding_mask=encoder_mask_bool,
                                 memory_key_padding_mask=encoder_mask_bool,
                                 pos=pos,
                                 query_pos=pos) # [t*h*w, b, c]
            #tgt_length = lengths - 1
            frame_num = encoder_out.shape[1]
            decoder_in = init.repeat(1, frame_num,  1)
            #decoder_in = init_all
            #print(frame_num)
            d_length = frame_num
            #print(d_length)
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, change_epoch)
            #print(frame_num)
            decoder_out = decoder_in

        assert decoder_out.shape[1] == audio.shape[1]
        
        return decoder_out

    def get_3dmm_loss(self, pred, gt):
        b, t, c = pred.shape
        xpred = pred.view(b * t, c)
        xgt = gt.view(b * t, c)
        pairwise_distance = F.pairwise_distance(xpred, xgt)
        loss = torch.mean(pairwise_distance)
        spiky_loss = self.get_spiky_loss(pred, gt)
        return loss, spiky_loss
    
    def get_spiky_loss(self, pred, gt):
        b, t, c = pred.shape
        pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]
        gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]
        pred_spiky = pred_spiky.view(b * (t - 1), c)
        gt_spiky = gt_spiky.view(b * (t - 1), c)
        pairwise_distance = F.pairwise_distance(pred_spiky, gt_spiky)
        return torch.mean(pairwise_distance)

    def get_loss(
        self,
        pred_3dmm_dynam,
        oth_listener_3dmm_dynam,
    ):
        bs = pred_3dmm_dynam.size(0)
        # angle / exp / trans loss
        pd_angle, pd_exp, pd_trans, pd_crop = torch.split(pred_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        gt_angle, gt_exp, gt_trans, gt_crop = torch.split(oth_listener_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        angle_loss, angle_spiky_loss = self.get_3dmm_loss(pd_angle, gt_angle)
        exp_loss, exp_spiky_loss = self.get_3dmm_loss(pd_exp, gt_exp)
        trans_loss, trans_spiky_loss = self.get_3dmm_loss(pd_trans, gt_trans)
        crop_loss, crop_spiky_loss = self.get_3dmm_loss(pd_crop, gt_crop)

        loss = angle_loss       * self.loss_weights['loss_angle'] + \
               angle_spiky_loss * self.loss_weights['loss_angle_spiky'] + \
               exp_loss         * self.loss_weights['loss_exp'] + \
               exp_spiky_loss   * self.loss_weights['loss_exp_spiky'] + \
               trans_loss       * self.loss_weights['loss_trans'] + \
               trans_spiky_loss * self.loss_weights['loss_trans_spiky'] + \
               crop_loss        * self.loss_weights['loss_crop'] + \
               crop_spiky_loss  * self.loss_weights['loss_crop_spiky']

        with torch.no_grad():
            loss_dict = {
                'TOTAL_LOSS': {
                    'val': loss.item(),
                    'n': bs,
                },
                'loss_angle': {
                    'val': angle_loss.item(),
                    'n': bs,
                },
                'loss_angle_spiky': {
                    'val': angle_spiky_loss.item(),
                    'n': bs,
                },
                'loss_exp': {
                    'val': exp_loss.item(),
                    'n': bs,
                },
                'loss_exp_spiky': {
                    'val': exp_spiky_loss.item(),
                    'n': bs,
                },
                'loss_trans': {
                    'val': trans_loss.item(),
                    'n': bs,
                },
                'loss_trans_spiky': {
                    'val': trans_spiky_loss.item(),
                    'n': bs,
                },
                'loss_crop': {
                    'val': crop_loss.item(),
                    'n': bs,
                },
                'loss_crop_spiky': {
                    'val': crop_spiky_loss.item(),
                    'n': bs,
                },
            }
            for key in loss_dict:
                loss_dict[key]['weight'] = self.loss_weights[key]
        return loss, loss_dict

    
    def forward(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target=None,
    ):
        if target is not None:
            pred = self.predict(
                audio,
                driven,
                init,
                lengths,
                epoch,
                change_epoch,
                target
            )
        else:
            pred = self.decode_period_sl_initlast_cat(
                audio,
                driven,
                init,
                lengths,
            )
            

        if target is not None:
            for i in range(audio.size(0)):
                pred[i, lengths[i] - 1:, :] = 0.
            target_expand = torch.zeros_like(pred)
            target_expand[:, :-1, :] = target
            loss, loss_dict = self.get_loss(
                pred,
                target_expand,
            )
            return loss, loss_dict, pred
        return pred
