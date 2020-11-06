## Prerequisites

You have to have GPU in order to train our models (or go and fix CTC computation in the code, but be warned that TF CTC is a mess of different versions of CTC).
You also should have Coral Edge TPU & install their SDK. While it is not technically necessary to own Coral in order to retrain the networks in the paper, it is highly recommended.
Without Coral, you can still validate TFLite files that the training produces, however, TFLite inference on x64-86 is not optimized and you are likely going to wait forever to get results (guesstimate is that the CPU inference is 100x slower than Coral)

## Installation


## Running

To run experiments in the paper
1) download training dataset: TODO
2) run `TF_FORCE_GPU_ALLOW_GROWTH=true python train.py --data_dir=/path/to/training/dataset --batch_size 100 --schedule linear_6000_001 --config paper_bonito_f128_k7_r5 --outname tmp`

To see a full list of available network configurations, see `configs.py`
