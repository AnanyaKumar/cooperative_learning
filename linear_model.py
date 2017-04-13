
import os
import sys

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class LinearModel:

    def __init__(self, input_size, output_size, checkpoint=None):
        self.input_size = input_size
        self.output_size = output_size

        self.sess = tf.Session()
        self._build()
        self.saver = tf.train.Saver()

        if checkpoint is not None:
            self.restore_model(checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.iter_number = 0

    ## Public methods

    def predict(self, state):
        """
        state: B x I
        """
        output = self.sess.run(self.out, feed_dict={self.input_state: state})
        return output

    def train(self, state, target):
        """
        state: B x I
        target: B x A
        """
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary],
                feed_dict={self.input_state: state, self.target: target})
        self.summary_writer.add_summary(summary, self.iter_number)

        return loss

    def save_model(self, path):
        """
        Usage:
            model.save_model('saved_models/my_model.ckpt')
        """
        save_path = self.saver.save(self.sess, path)
        print('Model saved to "%s"' % os.path.abspath(save_path))

    def restore_model(self, path):
        """
        For some reason, Tensorflow ONLY supports absolute paths when using Saver.
        """
        if not os.path.isabs(path):
            print('Provided checkpoint path "%s" is not absolute, model not restored.' % path)
            sys.exit()

        self.saver.restore(self.sess, path)

    ## Private methods

    def _build(self):
        self.input_state = tf.placeholder(tf.float32, [None, self.input_size])

        # fc1
        self.fc1 = self._fc_layer(self.input_state, 64, 'fc1')

        # fc2
        self.fc2 = self._fc_layer(self.fc1, 64, 'fc2')

        # output (fc without relu)
        out_weight, out_bias = self._get_fc_weights(64, self.output_size, 'out')
        self.out = tf.matmul(self.fc2, out_weight)
        self.out = tf.nn.bias_add(self.out, out_bias)
        tf.summary.histogram('output_values', self.out)

        self.target = tf.placeholder(tf.float32, [None, self.output_size])
        self.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.out)
        # TODO: decay LR
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/')

    def _fc_layer(self, input_node, output_size, name):
        input_shape = input_node.get_shape().as_list()  # [None, input_size]

        fc_weight, fc_bias = self._get_fc_weights(input_shape[1], output_size, name)

        fc = tf.matmul(input_node, fc_weight)
        fc = tf.nn.bias_add(fc, fc_bias)
        fc = tf.nn.relu(fc)

        tf.summary.histogram(name+'_weight', fc_weight)
        tf.summary.histogram(name+'_bias', fc_bias)
        tf.summary.histogram(name+'_activations', fc)

        return fc

    def _get_fc_weights(self, input_size, output_size, name):
        fc_weight = tf.get_variable(
                name=name+'_weight',
                shape=[input_size, output_size],
                initializer=xavier_initializer())
        fc_bias = tf.Variable(tf.constant(0.0, shape=[output_size]), name=name+'_bias')

        return fc_weight, fc_bias
