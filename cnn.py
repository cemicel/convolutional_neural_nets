import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import misc
from sklearn.utils import shuffle
import operator
import functools

# -------------------------------------------------------------------------------------------------------------------

length = 90
class_one = []
class_two = []

label_one = []
label_two = []

path_for_class_one = '/Users/volodymyrkepsha/Documents/Study/Python/Projects/FILE/my_data/2.1/cropped/{}.PNG'
path_for_class_two = '/Users/volodymyrkepsha/Documents/Study/Python/Projects/FILE/my_data/3.2/cropped/{}.PNG'

for i in range(length):
    class_one.append((misc.imread(path_for_class_one.format(i), mode="L")))
    label_one.append([1, 0])
for i in range(length):
    class_two.append((misc.imread(path_for_class_two.format(i), mode="L")))
    label_two.append([0, 1])

all_data = []
all_labels = []

all_data.extend(class_one)
all_data.extend(class_two)

all_labels.extend(label_one)
all_labels.extend(label_two)

all_data, all_labels = shuffle(all_data, all_labels, random_state=0)

train_input = np.expand_dims(np.array([all_data[i] for i in range(80)]), -1)
train_label = np.array([all_labels[i] for i in range(80)])

test_input = np.expand_dims((np.array([all_data[i] for i in range(80, 91)])), -1)
test_label = np.array([all_labels[i] for i in range(80, 91)])

image_hight = train_input.shape[1]
image_width = train_input.shape[2]

x = tf.placeholder(tf.float32, [None, image_hight * image_width * 1], name='X_muliplied')

x_shaped = tf.reshape(x, [-1, image_hight, image_width, 1], name='X_shaped_{}_{}'.format(image_hight, image_width))

y = tf.placeholder(tf.float32, [None, 2], name="Y_labels")

neurons_in_first_dense = 1024
classes_numb = 2

image_hight = train_input.shape[1]
image_width = train_input.shape[2]

# -------------------------------------------------------------------------------------------------------------------

X = tf.placeholder(tf.float32, [None, image_hight * image_width * 1], name='X_muliplied')

X_shaped = tf.reshape(X, [-1, image_hight, image_width, 1], name='X_shaped_{}_{}'.format(image_hight, image_width))

Y = tf.placeholder(tf.float32, [None, classes_numb], name="Y_labels")
neurons_in_first_dense = 1000


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

# -------------------------------------------------------------------------------------------------------------------

Y_ = tf.nn.softmax(dense_layer_2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer_2, labels=Y))
optimiser = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

range_val = 0

with tf.Session() as sess:
    sess.run(init)
    for i in range(30):
        range_val = i
        _, c = sess.run([optimiser, cross_entropy], feed_dict={X_shaped: train_input, Y: train_label})
        test_acc = sess.run(accuracy, feed_dict={X_shaped: test_input, Y: test_label})
        print("Epoch:", (i + 1), ' cost: {}'.format(c), " test accuracy: {:.3f}".format(test_acc))

    # print(np.round(sess.run(Y_, feed_dict={X_shaped: train_input, Y: train_label}), 3))
    # print(np.round(sess.run(Y, feed_dict={X_shaped: train_input, Y: train_label}), 3))
