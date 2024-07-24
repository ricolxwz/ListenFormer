# ListenFormer
Code for the paper 'ListenFormer: Responsive Listening Head Generation with Non-autoregressive Transformers'

## Installation
```bash
pip install -r requirements.txt
```
## Data Preparation (For ViCo dataset)
Following the [vico_data](https://github.com/dc3ea9f/vico_challenge_baseline?tab=readme-ov-file#data-preparation)

## Train and inference:
```bash
./run.sh
./eval.sh
```

## Rendering:
Following the [vico_render](https://github.com/dc3ea9f/vico_challenge_baseline?tab=readme-ov-file#render-to-videos)

## Eval:
Following the [vico_eval](https://github.com/dc3ea9f/vico_challenge_baseline?tab=readme-ov-file#evaluation)


## Generated videos with negative attitude
<table>
  <tr>
    <td><img src="show/Qualitative Evaluation (negative attitude)/1_ViCo.gif" alt="图片1" width="200"/><br>ViCo</td>
    <td><img src="show/Qualitative Evaluation (negative attitude)/2_PCHG.gif" alt="图片2" width="200"/><br>PCHG</td>
    <td><img src="show/Qualitative Evaluation (negative attitude)/3_DSPN.gif" alt="图片3" width="200"/><br>DSPN</td>
    <td><img src="show/Qualitative Evaluation (negative attitude)/4_Ours.gif" alt="图片4" width="200"/><br>Ours</td>
  </tr>
</table>

## Acknowledgement 
We build our project base on [vico](https://github.com/dc3ea9f/vico_challenge_baseline). The codes for transformers are derived from [wenet](https://github.com/wenet-e2e/wenet/tree/main/wenet/transformer).
