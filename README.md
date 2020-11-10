# Training pipeline for DeepNano-Coral basecaller

This is a training pipeline for our Coral Edge TPU basecaller (https://arxiv.org/abs/2011.04312).
For the basecaller itself visit https://github.com/fmfi-compbio/coral-basecaller.


## Prerequisites

- You have to have a GPU with at least 11GB of memory in order to train our models (or go and fix CTC computation in the code, but be warned that TF CTC is a mess of different versions of CTC).
- You also should have Coral Edge TPU & install their SDK. While it is not technically necessary to own Coral in order to retrain the networks in the paper, it is highly recommended.
Without Coral, you can still validate TFLite files that the training produces, however, TFLite inference on x64-86 is not optimized and you are likely going to wait forever to get results (guesstimate is that the CPU inference is 100x slower than Coral)
- Install https://github.com/fmfi-compbio/coral-basecaller for evaluation of results.

## Installation

You need to install `pip install tf-nightly-gpu biopython` in order to run the training process. We used `tf-nightly-gpu==2.4.0.dev20200719` but any TF2.4 version should be fine so you might alternatively consider installing `tensorflow-gpu` with version pinned to `2.4`.

## Running

To run experiments in the paper
1) download training dataset: TODO
2) run
```
TF_FORCE_GPU_ALLOW_GROWTH=true python train.py --data_dir=/path/to/training/dataset --batch_size 100 --schedule linear_6000_001 --config paper_bonito_f128_k7_r5 --outname paper_bonito_f128_k7_r5
```
This will produce `models/paper_bonito_f128_k7_r5.tflite` in about a day of training on GTX 1080 Ti. 

To see a full list of available network configurations to train, see `configs.py`


## Evaluation
In order to evaluate on Coral you need to first compile the model:
`edgetpu_compiler -a -s models/paper_bonito_f128_k7_r5.tflite`, this should produce `paper_bonito_f128_k7_r5_edgetpu.tflite` file.
Then follow basecalling instructions at https://github.com/fmfi-compbio/coral-basecaller
