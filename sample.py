import numpy as np
import tensorflow as tf

import argparse
import time
import os
import json
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default='The ',
                       help='prime text')
    parser.add_argument('--sample_rule', type=int, default=0,
                       help='1 will use argmax, 0 will use temperatured sampling')
    parser.add_argument('--temperature', type=float, default=1.,
                       help='temperature to scale the probabilities')
    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    val_loss_file = args.save_dir + '/val_loss.json'
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        if os.path.exists(val_loss_file):
            with open(val_loss_file, "r") as text_file:
                text = text_file.read()
                loss_json = json.loads(text)
                losses = loss_json.keys()
                losses.sort(key=lambda x: float(x))
                loss = losses[0]
                model_checkpoint_path =  loss_json[loss]['checkpoint_path']
                #print(model_checkpoint_path)
                saver.restore(sess, model_checkpoint_path)
                result = model.sample(sess, chars, vocab, args.n, args.prime, args.sample_rule, args.temperature)
                print(result) #add this back in later, not sure why its not working
                output = "/data/output/"+ str(int(time.time())) + ".txt"
                with open(output, "w") as text_file:
                    text_file.write(result)
                print(output)


if __name__ == '__main__':
    main()
