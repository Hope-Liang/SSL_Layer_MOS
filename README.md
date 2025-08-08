# Selection of Layers from Self-supervised Learning Models for Predicting Mean-Opinion-Score of Speech

This is the official implementation for the results reported in the paper "Selection of Layers from Self-supervised Learning Models for Predicting Mean-Opinion-Score of Speech" accepted at IEEE ASRU 2025.

Authors: Xinyu Liang, Fredrik Cumlin
Emails: hopeliang@icloud.com, fcumlin@gmail.com

## Training
The framework is Gin configurable; specifying model and dataset is done with a Gin config. See examples in `configs/*.gin`. The SSL features should be pre-generated and the path should passed as an argument to the training script.

## SSL fine-tune on MOS prediction
The code can be found in [Speech_SSL_finetune_MOS](https://github.com/Hope-Liang/Speech_SSL_finetune_MOS).

## Model checkpoints
All projection head model checkpoints can be found in [Google Drive](https://drive.google.com/file/d/1_u8-l5pVZVoPXVVOi8HOcRxj31UScmMC/view?usp=share_link), with 5 runs for each layer of each SSL model on each of the three datasets.

## Inference
1. Download the model checkpoints from the Google Drive link.
2. Run `python inference.py --audio_path=<YOUR_AUDIO_PATH> --ssl_model=<SELECTED_SSL_MODEL> --ssl_layer=<SELECTED_SSL_LAYER> --ckpt=<SELECTED_MODEL_CKPT_FROM>`