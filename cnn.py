import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
import operator
import functools
import os


def get_data(path_r, numb_of_pictures=None, for_test=None):
    """method for getting numpy matrices of images and one-hot-encode labels for both training and tasting
     Data should be stored in separate files with names i.png (0<=i<=n)
    Parameters:
    - num_of_pics - provide particular number of pictures to be extracted,
         if None extracts all.
    - for_test - number of pics for testing. Cannot be bigger than 30% or less that 1
        by default 10%
    """

    paths = [x[0] for c, x in enumerate(os.walk(path_r)) if c > 0]

    f = paths[1]
    ath, dirs, files = next(os.walk(f))
    file_count = len(files) - 1
    test_input = []
    test_label = []
    train_input = []
    train_label = []

    if numb_of_pictures is None and for_test is None:
        num_of_pics = file_count
        for_test = file_count * len(paths) * .01
    elif numb_of_pictures is not None and for_test is None:
        for_test = numb_of_pictures * len(paths) * .01
        num_of_pics = numb_of_pictures
    elif numb_of_pictures is not None and for_test is not None:
        for_test = for_test
        num_of_pics = numb_of_pictures

    if numb_of_pictures < len(paths) or numb_of_pictures < (for_test * 3):
        raise ValueError('Not enough for training!')
    elif for_test < numb_of_pictures * len(paths) * .01:
        raise ValueError('Not enough for testing!')
    elif for_test > file_count * len(paths) * .3:
        raise ValueError('Too many for testing!')

    for p_count, path in enumerate(paths):
        one_hot_enc_arr = np.zeros(len(paths))
        one_hot_enc_arr[p_count] = 1
        print(path.split('/')[-1], one_hot_enc_arr)
        for pic in range(num_of_pics):
            if pic < for_test:
                test_input.append(scipy.misc.imread(path + '/{}.png'.format(pic), mode="L"))
                test_label.append(one_hot_enc_arr)
            else:
                train_input.append(scipy.misc.imread(path + '/{}.png'.format(pic), mode="L"))
                train_label.append(one_hot_enc_arr)

    train_input = np.expand_dims(train_input, -1)
    train_label = np.array(train_label)
    test_input = np.expand_dims(test_input, -1)
    test_label = np.array(test_label)

    # print('Train input shape: {}, train label shape: {}\nTest input shape: {}, test label shape: {}'.
    #      format(train_input.shape, (train_label).shape, test_input.shape, (test_label).shape))

    return train_input, train_label, test_input, test_label


def get_custom_input(path):
    image = scipy.misc.imread(path, mode="L")
    return np.expand_dims(image, -1)


def get_conv_layer(input_data, num_chanels, num_filters, filter_shape, pool_shape, name):
    """ method for getting a convolutional layer
        with particular parameters as:
        num_of_chanels - here it is 1, for grayscale
        num_of_filters - self explained
        filter_shape - self explained
        pool_shape - maxpool shape
        strides for max pool is 2x2"""

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


def run(data, train=True, iterations=1, test_custom=None):
    train_input, train_label, test_input, test_label = data

    image_hight = train_input.shape[1]
    image_width = train_input.shape[2]
    classes_numb = 10

    # -------------------------------------------------------------------------------------------------------------------

    X = tf.placeholder(tf.float32, [None, image_hight * image_width * 1], name='X_muliplied')

    X_shaped = tf.reshape(X, [-1, image_hight, image_width, 1], name='X_shaped_{}_{}'.format(image_hight, image_width))

    Y = tf.placeholder(tf.float32, [None, classes_numb], name="Y_labels")

    neurons_in_first_dense = 1024

    conv_layer_1 = get_conv_layer(X_shaped, num_chanels=1, num_filters=32, filter_shape=[5, 5], pool_shape=[2, 2],
                                  name='layer_1')
    conv_layer_2 = get_conv_layer(conv_layer_1, num_chanels=32, num_filters=64, filter_shape=[5, 5],
                                  pool_shape=[2, 2], name='layer_2')

    flattened_shape = functools.reduce(operator.mul, [i.value for i in conv_layer_2.shape[1:]], 1)
    flattened = tf.reshape(conv_layer_2, [-1, flattened_shape])

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
        if train:
            test_acc_list = []
            sess.run(init)
            for i in range(iterations):
                _, c = sess.run([optimiser, cross_entropy], feed_dict={X_shaped: train_input, Y: train_label})
                test_acc = sess.run(accuracy, feed_dict={X_shaped: test_input, Y: test_label})
                print("Epoch:", (i + 1), ' cost: {}'.format(c), " test accuracy: {:.3f}".format(test_acc))
                test_acc_list.append(test_acc)
            # for debuging output of the network
            # print(np.round(sess.run(Y_, feed_dict={X_shaped: train_input, Y: train_label}), 3))
            # print(np.round(sess.run(Y, feed_dict={X_shaped: train_input, Y: train_label}), 3))
            save_path = saver.save(sess,
                                   "/Users/volodymyrkepsha/Documents/Study/Python/Projects/neural_nets/neural_network_tensorflow/save_font/model.ckpt")
            plt.plot(range(iterations), test_acc_list)
            plt.show()

        elif test_custom is not None:
            saver.restore(sess,
                          "/Users/volodymyrkepsha/Documents/Study/Python/Projects/neural_nets/neural_network_tensorflow/save_font/model.ckpt")

            test_input = np.array(test_custom)
            test_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] )

            print(np.round(sess.run(Y_, feed_dict={X_shaped: test_input, Y: test_label})))

        else:
            saver.restore(sess,
                          "/Users/volodymyrkepsha/Documents/Study/Python/Projects/neural_nets/neural_network_tensorflow/save_font/model.ckpt")
            # print(np.round(sess.run(Y, feed_dict={X_shaped: test_input, Y: test_label})))
            # print(np.round(sess.run(Y_, feed_dict={X_shaped: test_input, Y: test_label})))


if __name__ == '__main__':
    root_path_ = 'data_text_form/'

    data_ = get_data(root_path_, numb_of_pictures=90, for_test=10)

    run(data=data_, train=True, iterations=50)

    path_custom_times = '/custom_path_of_image_28x168'
    image_time = get_custom_input(path=path_custom_times)
    run(data=data_, train=False, iterations=50, test_custom=[image_time])

