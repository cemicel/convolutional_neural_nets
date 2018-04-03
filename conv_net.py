import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
import operator
import functools
import os
import sys
import batch_generator, generate_data


def get_conv_layer(input_data, num_chanels, num_filters, filter_shape, pool_shape, name):
    """ method for getting a convolutional layer
        with particular parameters as:
        num_of_chanels - here it is 1, for grayscale
        num_of_filters - self explained
        filter_shape - self explained
        pool_shape - maxpool shape
        strides for max pool """

    conv_filter_shape = [filter_shape[0], filter_shape[1], num_chanels, num_filters]

    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name + '_W')

    biases = tf.Variable(tf.truncated_normal([num_filters], name=name + '_b'))

    strides_conv = [1, 1, 1, 1]

    out_layer = tf.nn.conv2d(input_data, weights, strides=strides_conv, padding='SAME')

    out_layer += biases

    out_layer = tf.nn.relu(out_layer)

    ksize = [1, pool_shape[0], pool_shape[1], 1]

    strides = [1, 2, 2, 1]

    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def get_full_connected(shape=[None, None], prev_layer=None, is_last=False):
    '''method for getting a fully connected layer layers
       if last layer flag is false relu is apllied to the layer'''
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=.03, name='wd1'))
    biases = tf.Variable(tf.truncated_normal(shape=[shape[1]], stddev=.01, name='bd1'))
    dense_layer = tf.matmul(prev_layer, weights) + biases
    if is_last:
        return dense_layer

    dense_layer = tf.nn.relu(dense_layer)
    return dense_layer


def run(batch_size=5, epochs=1, mode='training'):
    image_hight = generate_data.img_height
    image_width = generate_data.img_width

    classes_numb = 10
    neurons_in_first_dense = 1024
    # -------------------------------------------------------------------------------------------------------------------

    X = tf.placeholder(tf.float32, [None, image_hight * image_width * 1], name='X_muliplied')
    X_shaped = tf.reshape(X, [-1, image_hight, image_width, 1], name='X_shaped_{}_{}'.format(image_hight, image_width))
    Y = tf.placeholder(tf.float32, [None, classes_numb], name="Y_labels")

    conv_layer_1 = get_conv_layer(X_shaped, num_chanels=1, num_filters=32, filter_shape=[4, 4], pool_shape=[2, 2],
                                  name='layer_1')
    conv_layer_2 = get_conv_layer(conv_layer_1, num_chanels=32, num_filters=64, filter_shape=[6, 6],
                                  pool_shape=[2, 2], name='layer_2')
    conv_layer_3 = get_conv_layer(conv_layer_2, num_chanels=64, num_filters=128, filter_shape=[8, 8],
                                  pool_shape=[4, 4], name='layer_3')
    flattened_shape = functools.reduce(operator.mul, [i.value for i in conv_layer_3.shape[1:]], 1)
    flattened = tf.reshape(conv_layer_3, [-1, flattened_shape])
    dense_layer_1 = get_full_connected([flattened_shape, neurons_in_first_dense], prev_layer=flattened, is_last=False)
    dense_layer_2 = get_full_connected([neurons_in_first_dense, classes_numb], prev_layer=dense_layer_1, is_last=True)
    Y_ = tf.nn.softmax(dense_layer_2)
    # -------------------------------------------------------------------------------------------------------------------
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer_2, labels=Y))
    optimiser = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # -------------------------------------------------------------------------------------------------------------------
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)

        if mode == 'training':

            batch_generator.init(batch_size=batch_size)
            test_input, test_label = batch_generator.get_test()
            total_batch = int(batch_generator.train_class_size / batch_size)
            test_acc_list = []
            for e in range(epochs):
                test_acc = 0
                avg_cost = 0

                for b in range(total_batch):
                    train_input, train_label = batch_generator.next_batch()
                    _, c = sess.run([optimiser, cross_entropy], feed_dict={X_shaped: train_input, Y: train_label})
                    avg_cost += c / total_batch

                test_acc = sess.run(accuracy, feed_dict={X_shaped: test_input, Y: test_label})
                print("Epoch:", (e + 1), ' cost: {}'.format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
                batch_generator.reset_batch_index()

                test_acc_list.append(test_acc)
            saver.save(sess, os.getcwd() + '/save/tmp/model.ckpt')

            plt.plot([i for i in range(epochs)], (test_acc_list))
            plt.show()
        else:

            test_input, test_label = get_data_test('/Users/volodymyrkepsha/Documents/github/cnn/test', for_test=True)
            saver.restore(sess,
                          "/Users/volodymyrkepsha/Documents/github/cnn/save/tmp/model.ckpt")
            print(np.round(sess.run(Y, feed_dict={X_shaped: test_input, Y: test_label})))

            print(np.round(sess.run(Y_, feed_dict={X_shaped: test_input, Y: test_label})))
            test_acc = sess.run(accuracy, feed_dict={X_shaped: test_input, Y: test_label})
            print(" test accuracy: {:.3f}".format(test_acc))


if __name__ == '__main__':

    try:
        try:
        mode = sys.argv[1]
        batch_size = sys.argv[2]
        epochs = sys.argv[3]
        run(batch_size=int(batch_size), epochs=int(epochs))
    except IndexError:
        print('provide type: training / test')
