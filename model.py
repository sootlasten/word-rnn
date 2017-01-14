from __future__ import print_function

import time

import tensorflow as tf

import process_data
from process_data import vocab_size


class Network():

    def __init__(self, n_h_layers, n_h_units, seq_length, cell_type):
        # network parameters
        self.n_h_layers = n_h_layers
        self.n_h_units = n_h_units

        self.seq_length = seq_length

        self.x, self.final_state, self.init_state, self.preds, self.logits = \
            self._build_network(cell_type)

    def train(self, batch_size, eta, grad_clip, n_epochs, train_frac):
        # BUILD GRAPH FOR TRAINING ...
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
            tr_data, val_data = process_data.split_data(train_frac)

            print("Training...")
            for epoch_n in xrange(1, n_epochs + 1):
                start_time = time.time()

                # full pass over the training set
                gen_tr_batch = process_data.gen_batches(
                    tr_data, batch_size, self.seq_length)

                prev_state = None

                for tr_batches_n, (x_tr, y_tr) in enumerate(gen_tr_batch, 1):
                    feed_dict = {self.x: x_tr, y: y_tr}
                    if prev_state:
                        feed_dict[self.init_state] = prev_state

                    loss, prev_state, _ = sess.run([total_loss, self.final_state, train_step],
                                                   feed_dict=feed_dict)
                    print(loss)

    def _build_network(self, cell_type):
        """Input shape should be (batch_size, seq_length, 1)."""
        x = tf.placeholder(tf.int32, [None, self.seq_length], name='input')

        embeddings = tf.get_variable('embedding_matrix', [vocab_size, self.n_h_units])
        rnn_input = tf.nn.embedding_lookup(embeddings, x)

        # choose cell type
        if cell_type == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(self.n_h_units)
        elif cell_type == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(self.n_h_units)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(self.n_h_units)

        # stack multiple cells on top of each other
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_h_layers)

        batch_size = tf.shape(x)[0]
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_out, final_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=init_state)

        with tf.variable_scope('softmax'):
            w_out = tf.get_variable('W_out', [self.n_h_units, vocab_size])
            b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))

        # reshape
        rnn_out = tf.reshape(rnn_out, [-1, self.n_h_units])

        logits = tf.matmul(rnn_out, w_out) + b
        preds = tf.nn.softmax(logits)

        return x, final_state, init_state, preds, logits

if __name__ == '__main__':
    net = Network(
        n_h_layers=3,
        n_h_units=100,
        seq_length=50,
        cell_type='LSTM')

    net.train(
        batch_size=30,
        eta=2e-3,
        grad_clip=5,
        n_epochs=10,
        train_frac=0.95)