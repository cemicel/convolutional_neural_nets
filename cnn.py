import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from sklearn.utils import shuffle
import operator
import functools
import os

# -------------------------------------------------------------------------------------------------------------------

print(scipy.misc)

root_path = '/data_text_form'


def get_data(path_r, num_of_pics=None, for_test=None):
    paths = [x[0] for c, x in enumerate(os.walk(os.getcwd()+path_r)) if c > 0]

    f = paths[1]
    ath, dirs, files = next(os.walk(f))
    file_count = len(files) - 1
    test_input = []
    test_label = []
    train_input = []
    train_label = []
    if num_of_pics is None:
        num_of_pics = file_count
    if for_test is None and num_of_pics is None:
        for_test = file_count * len(paths) * .01
    elif for_test is None and num_of_pics is not None:
        for_test = num_of_pics * len(paths) * .01
    elif for_test < num_of_pics * len(paths) * .01:
        raise ValueError('Not enough for testing')
    elif for_test > file_count * len(paths) * .3:
        raise ValueError('Too many for testing')

    for p_count, path in enumerate(paths):
        one_hot_enc_arr = np.zeros(len(paths))
        for pic in range(num_of_pics):
            one_hot_enc_arr[p_count] = 1

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

    return train_input, train_label, test_input, test_label


train_input, train_label, test_input, test_label = get_data(root_path, num_of_pics=40)

image_hight = train_input.shape[1]
image_width = train_input.shape[2]
classes_numb = 10

# -------------------------------------------------------------------------------------------------------------------

X = tf.placeholder(tf.float32, [None, image_hight * image_width * 1], name='X_muliplied')

X_shaped = tf.reshape(X, [-1, image_hight, image_width, 1], name='X_shaped_{}_{}'.format(image_hight, image_width))

Y = tf.placeholder(tf.float32, [None, classes_numb], name="Y_labels")

neurons_in_first_dense = 1024


def get_conv_layer(input_data, num_chanels, num_filters, filter_shape, pool_shape, name):
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
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=.03, name='wd1'))
    biases = tf.Variable(tf.truncated_normal(shape=[shape[1]], stddev=.01, name='bd1'))
    dense_layer = tf.matmul(prev_layer, weights) + biases
    if is_last:
        return dense_layer

    dense_layer = tf.nn.relu(dense_layer)
    return dense_layer


conv_layer_1 = get_conv_layer(X_shaped, 1, 32, [5, 5], [2, 2], name='layer_1')
conv_layer_2 = get_conv_layer(conv_layer_1, 32, 64, [5, 5], [2, 2], name='layer_2')

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
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
test_acc_list = []

with tf.Session() as sess:
    test_acc_list = []
    sess.run(init)
    for i in range(3):
        _, c = sess.run([optimiser, cross_entropy], feed_dict={X_shaped: train_input, Y: train_label})
        test_acc = sess.run(accuracy, feed_dict={X_shaped: test_input, Y: test_label})
        print("Epoch:", (i + 1), ' cost: {}'.format(c), " test accuracy: {:.3f}".format(test_acc))
        test_acc_list.append(test_acc)

    #print(np.round(sess.run(Y_, feed_dict={X_shaped: train_input, Y: train_label}), 3))
    #print(np.round(sess.run(Y, feed_dict={X_shaped: train_input, Y: train_label}), 3))

    plt.plot(range(3), test_acc_list)
    plt.show()
