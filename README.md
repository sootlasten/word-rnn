# word-rnn

An implementation of a lstm model inspired by [Zaremba et al., 2014](https://arxiv.org/abs/1409.2329) and Tensorflow's official 
[RNN tutorial](https://www.tensorflow.org/tutorials/recurrent/). The code is developed in Python 2.7.

The motivation for creating the model is to compare the performance of my [character-level language model](https://github.com/sootlasten/char-rnn) 
to a model that models language at a world-level.

### Training
To train the model, the script `train.py` must be run with the command `python train.py`. The script loads hyperparemeters from the file
`params.txt` (by default) and runs the model from `model.py` with these parameters set. The input file is located in `data/input.txt` by default. 
During training, after each epoch, the code prints some information about the performace of the model to standard output. It also saves 
the checkpoint files (that contain weights and additional info needed to reload the model) to `ckpts/` (by default) after each epoch. Each checkpoint 
file can be later loaded with the name `model_e[x].ckpt`, where `x` specifies the epoch number. Furthermore, it saves some info about the model to 
`ckpts/info.pickle` (by default), so the network can be rebuilt afterwards.

### Sampling
To sample from the model, the script `sample.py` must be run. This script also takes command line arguments (see `python sample.py -h` 
for additional info). The only mandatory argument, though, is `--info_file`, which should point to the info files saved in 
the training phase. By default, the sampled text is saved into `sample.txt`.

### TODO
1. when sampling, `<UNK>` and `<EOS>` should be dealt with