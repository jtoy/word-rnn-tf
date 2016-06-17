# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.

# Training Parameters Details
For a list of parameters that can be change, take a look at the 'train.py' file.

'--data_dir'
you can tell the model where the data is in by using the '--data_dir' param, the file that contains the data need to be named as "input.txt". If you are training on different dataset make sure you use a different '--data_dir' for each of them.

'--save_dir'
The '--save_dir' tells the model where to save the model checkpoints data and the validation loss json file. If you are training on different dataset make sure you use a different '--save_dir' for each of them, otherwise the checkpoint data and validation loss data might not save correctly, or override the previous one.  The reason is that we only save the best 4 checkpoints with the best validation loss score for a given dataset. that way we can try to train the model with different parameters and find the best model from them.

'--save_every'
The '--save_every' tells the model how often to run the validation loss and save the checkpoint file if the score is lower than the previous saved validation loss score.

'--training_data_ratio'
The '--training_data_ratio' tells the model to split the dataset into training data and validation data base on the ratio set in this param. the default is 0.8, means the training data will contain 80% of the data, and validation will have 20%.

'--write_summary_every'
The '--write_summary_every' tells the model how often to write to the tensorboard event file.

# Sampling
To sample from a checkpointed model, `python sample.py`.
For a list of parameters that can be change, take a look at the 'sample.py' file.

The sampling will always use the checkpoint that has the best validation loss score.



# Roadmap
- Add explanatory comments
- Expose more command-line arguments
- Compare accuracy and performance with char-rnn
- implement feedback from users


