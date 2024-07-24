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


## Acknowledgement 
We build our project base on [vico](https://github.com/dc3ea9f/vico_challenge_baseline). The codes for transformers are derived from [wenet](https://github.com/wenet-e2e/wenet/tree/main/wenet/transformer).
