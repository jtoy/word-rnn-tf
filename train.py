from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/test',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--training_data_ratio', type=float, default=0.8,
                       help='training data split ratio')
    parser.add_argument('--write_summary_every', type=int, default=500,
                        help='write summary frequency')
    args = parser.parse_args()
    train(args)

def train(args):
    print(args)
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.training_data_ratio)
    args.vocab_size = data_loader.vocab_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    #sess = tf.InteractiveSession()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('/tmp', sess.graph)

        step = 0
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            #print("model learning rate is {}".format(model.lr.eval()))
            data_loader.reset_batch_pointer('train')

            state = model.initial_state.eval()
            for b in xrange(data_loader.ntrain):
                start = time.time()
                x, y = data_loader.next_batch('train')

                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                step = e * data_loader.ntrain + b
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(step,
                            args.num_epochs * data_loader.ntrain,
                            e, train_loss, end - start))

                if step % args.write_summary_every == 0:
                    # training loss
                    summary_str = sess.run(summary_op, feed_dict=feed)
                    summary_writer.add_summary(summary_str, step)

                if step % args.save_every == 0 or (step + 1) == (args.num_epochs * data_loader.ntrain):
                    # eval validation loss
                    data_loader.reset_batch_pointer('validation')
                    validation_state = model.initial_state.eval()
                    val_losses = 0
                    for n in xrange(data_loader.nvalidation):
                        x, y = data_loader.next_batch('validation')
                        val_feed = {model.input_data: x, model.targets: y, model.initial_state: validation_state}
                        validation_loss, validation_state = sess.run([model.cost, model.final_state], val_feed)
                        val_losses += validation_loss

                    validation_loss = val_losses / data_loader.nvalidation
                    print("validation loss is {}".format(validation_loss))

                    # write top 5 validation loss to a json file
                    args_dict = vars(args)
                    args_dict['step'] = step
                    val_loss_file = args.save_dir + '/val_loss.json'
                    loss_json = ''
                    save_new_checkpoint = False
                    time_int = int(time.time())
                    args_dict['checkpoint_path'] = os.path.join(args.save_dir, 'model.ckpt-'+str(time_int))
                    if os.path.exists(val_loss_file):
                        with open(val_loss_file, "r") as text_file:
                            text = text_file.read()
                            if text == '':
                                loss_json = {validation_loss: args_dict}
                                save_new_checkpoint = True
                            else:
                                loss_json = json.loads(text)
                                losses = loss_json.keys()
                                if len(losses) > 3:
                                    losses.sort(key=lambda x: float(x), reverse=True)
                                    loss = losses[0]
                                    if validation_loss < float(loss):
                                        to_be_remove_ckpt_file_path =  loss_json[loss]['checkpoint_path']
                                        to_be_remove_ckpt_meta_file_path = to_be_remove_ckpt_file_path + '.meta'
                                        print("removed checkpoint {}".format(to_be_remove_ckpt_file_path))
                                        if os.path.exists(to_be_remove_ckpt_file_path):
                                            os.remove(to_be_remove_ckpt_file_path)
                                        if os.path.exists(to_be_remove_ckpt_meta_file_path):
                                            os.remove(to_be_remove_ckpt_meta_file_path)
                                        del(loss_json[loss])
                                        loss_json[validation_loss] = args_dict
                                        save_new_checkpoint = True
                                else:
                                    loss_json[validation_loss] = args_dict
                                    save_new_checkpoint = True
                    else:
                       loss_json = {validation_loss: args_dict}
                       save_new_checkpoint = True

                    if save_new_checkpoint:
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step = time_int)
                        print("model saved to {}".format(checkpoint_path + '-' + str(time_int)))

                        with open(val_loss_file, "w") as text_file:
                            json.dump(loss_json, text_file)



if __name__ == '__main__':
    main()
