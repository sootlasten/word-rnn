from __future__ import print_function

import os
import time
import cPickle as pickle

import tensorflow as tf
import numpy as np

import process_data

INFO_FILENAME = "info.pickle"


class Network():

    def __init__(self, n_h_layers, n_h_units, seq_length,
                 cell_type, vocab_size):
        # network parameters
        self.n_h_layers = n_h_layers
        self.n_h_units = n_h_units
        self.seq_length = seq_length
        self.cell_type = cell_type
        self.vocab_size = vocab_size

        self.x, self.keep_prob, self.final_state, self.init_state, \
        self.preds, self.logits, self.saver = self._build_network()

    def sample(self, n, prime_text, vocab_dict, reverse_vocab_dict, ckpt_path):
        """Sample from the model."""
        def get_words_ids(text):
            """Get words from prime text and convert them to word ids."""
            words = prime_text.replace('\n', ' <EOS> ').split()

            word_ids = []
            for word in words:
                default = vocab_dict['<UNK>']  # if input word is unknown
                id = vocab_dict.get(word, default)
                word_ids.append(id)

            return word_ids

        with tf.Session() as sess:
            self.saver.restore(sess, ckpt_path)  # restore model params

            # SAMPLE ...
            if not prime_text:  # if no prime text, choose a random word
                prime_text = vocab_dict.keys()[np.random.randint(self.vocab_size)]

            prime_text_ids = get_words_ids(prime_text)

            prev_state = None
            # feed in prime text
            for word_id in prime_text_ids:
                word_id = np.array(word_id).reshape((1, 1))  # convert int to numpy array

                feed_dict = {self.x: word_id, self.keep_prob: 1.0}
                if prev_state: feed_dict[self.init_state] = prev_state

                pred, prev_state = sess.run([self.preds, self.final_state], feed_dict)

            # now start generating novel text
            gen_text_ids = prime_text_ids
            pred_id = np.argmax(pred)
            gen_text_ids.append(pred_id)

            for _ in xrange(n):
                pred_id_arr = np.array(pred_id).reshape((1, 1))

                feed_dict = {self.x: pred_id_arr, self.keep_prob: 1.0,
                             self.init_state: prev_state}
                pred, prev_state = sess.run([self.preds, self.final_state], feed_dict)

                pred_id = np.argmax(pred)
                gen_text_ids.append(pred_id)

            # ids to text
            words = [reverse_vocab_dict[id] for id in gen_text_ids]
            return " ".join(map(str, words))

    def train(self, batch_size, eta, grad_clip, keep_prob, n_epochs,
              train_frac, checkpoint_dir):
        # data preprocessing
        data, _, vocab_dict, reverse_vocab_dict = \
            process_data.build_dataset(self.vocab_size)

        self._print_model_info(batch_size, eta, grad_clip, keep_prob,
                               n_epochs, train_frac, len(data))

        # save model info (num hidden layers etc.) to a file
        info_path = os.path.join(checkpoint_dir, INFO_FILENAME)
        self._save_info(info_path, vocab_dict, reverse_vocab_dict)

        # build graph for training ...
        y = tf.placeholder(tf.int32, [None, self.seq_length], name='target')
        y_reshaped = tf.reshape(y, [-1])  # flatten into 1-D

        total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.logits, y_reshaped))

        # for gradient clipping
        optimizer = tf.train.AdamOptimizer(eta)
        gvs = optimizer.compute_gradients(total_loss)
        clipped_gvs = [(tf.clip_by_value(
            grad, -grad_clip, grad_clip), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(clipped_gvs)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # do the actual training
            tr_data, val_data = process_data.split_data(data, train_frac)
            best_val_err, best_epoch_n = float('inf'), 0

            print("train data size:         %d" % len(tr_data))
            print("val data size:           %d\n" % len(val_data))

            print("Training...")
            for epoch_n in xrange(1, n_epochs + 1):
                start_time = time.time()

                # full pass over the TRAINING SET
                gen_tr_batch = process_data.gen_batches(
                    tr_data, batch_size, self.seq_length)

                prev_state = None
                total_tr_err = 0
                for tr_batches_n, (x_tr, y_tr) in enumerate(gen_tr_batch, 1):
                    feed_dict = {self.x: x_tr, y: y_tr, self.keep_prob: keep_prob}
                    if prev_state:
                        feed_dict[self.init_state] = prev_state

                    err, prev_state, _ = sess.run([total_loss, self.final_state, train_step],
                                                   feed_dict=feed_dict)
                    total_tr_err += err

                # full pass over the VALIDATION SET
                gen_val_batch = process_data.gen_batches(
                    val_data, batch_size, self.seq_length)

                prev_state = None
                total_val_err = 0

                for val_batches_n, (x_val, y_val) in enumerate(gen_val_batch, 1):
                    feed_dict = {self.x: x_val, y: y_val, self.keep_prob: 1.0}
                    if prev_state:
                        feed_dict[self.init_state] = prev_state

                    err, prev_state = sess.run([total_loss, self.final_state],
                                               feed_dict=feed_dict)
                    total_val_err += err

                # output info
                total_tr_err /= tr_batches_n
                total_val_err /= val_batches_n

                if total_val_err < best_val_err:
                    best_val_err = total_val_err
                    best_epoch_n = epoch_n

                # save network parameters
                ckpt_path = os.path.join(checkpoint_dir, "model_e{}".format(epoch_n))
                self.saver.save(sess, ckpt_path)

                print("Epoch %d completed in %d seconds" % (epoch_n, time.time() - start_time))
                print("Training loss:       %f" % total_tr_err)
                print("Validation loss:     %f" % total_val_err)
                print("Best epoch:          %d" % best_epoch_n)
                print("Best val loss:       %f\n" % best_val_err)

    def _build_network(self):
        """Input shape should be (batch_size, seq_length, 1)."""
        x = tf.placeholder(tf.int32, [None, self.seq_length], name='input')

        embeddings = tf.get_variable('embedding_matrix',
                                     [self.vocab_size, self.n_h_units])
        rnn_input = tf.nn.embedding_lookup(embeddings, x)

        # choose cell type
        if self.cell_type == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(self.n_h_units)
        elif self.cell_type == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(self.n_h_units)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(self.n_h_units)

        # dropout between layers
        keep_prob = tf.placeholder(tf.float32)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # stack multiple cells on top of each other
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_h_layers)

        batch_size = tf.shape(x)[0]
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_out, final_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=init_state)

        with tf.variable_scope('softmax'):
            w_out = tf.get_variable('W_out', [self.n_h_units, self.vocab_size])
            b = tf.get_variable('b', [self.vocab_size],
                                initializer=tf.constant_initializer(0.0))

        # reshape
        rnn_out = tf.reshape(rnn_out, [-1, self.n_h_units])

        logits = tf.matmul(rnn_out, w_out) + b
        preds = tf.nn.softmax(logits)

        # add ops to save and restore all the variables
        saver = tf.train.Saver()

        return x, keep_prob, final_state, init_state, preds, logits, saver

    def _save_info(self, file_path, vocab_dict, reverse_vocab_dict):
        """Save info about model hyperparameters and training data stuff."""
        param_dict = {
            'n_h_layers': self.n_h_layers,
            'n_h_units': self.n_h_units,
            'cell_type': self.cell_type,
            'vocab_size': self.vocab_size,
            'vocab_dict': vocab_dict,
            'reverse_vocab_dict': reverse_vocab_dict}

        with open(file_path, 'wb') as f:
            pickle.dump(param_dict, f)

    @staticmethod
    def _model_params_size():
        """Returns the number of trainable parameters of the model."""
        return np.sum([np.prod(var._variable._shape)
                       for var in tf.trainable_variables()])

    def _print_model_info(self, batch_size, eta, grad_clip, keep_prob,
                          n_epochs, train_frac, data_size):
        print("Model hyperparams and info:")
        print("---------------------------")
        print("n_h_layers:              %d" % self.n_h_layers)
        print("n_h_units:               %d" % self.n_h_units)
        print("batch_size:              %d" % batch_size)
        print("seq_length:              %d" % self.seq_length)
        print("vocab_size:              %d\n" % self.vocab_size)

        # print("drop_p:                  {:.3f}".format(self.drop_p))
        print("grad_clip:               %d" % grad_clip)
        print("keep_prob:               {:.3f}".format(keep_prob))
        print("eta:                     {:.6f}".format(eta))
        print("n_epochs:                %d" % n_epochs)
        print("train_frac:              {:.2f}\n".format(train_frac))

        print("data_size:               %d" % data_size)
        print("# of trainabe params:    %d" % Network._model_params_size())
        print("# of examples in batch:  %d\n" % (self.seq_length * batch_size))


if __name__ == '__main__':
    net = Network(
        n_h_layers=3,
        n_h_units=700,
        seq_length=100,
        cell_type='GRU')

    net.train(
        batch_size=30,
        eta=2e-2,
        grad_clip=5,
        n_epochs=1000000000000,
        train_frac=0.8,
        checkpoint_dir='ckpts')
