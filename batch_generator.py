import numpy as np
import cv2
import os

batch_size_global = None
data_paths_global = None
train_class_size = None
test_class_size = None
index_generator_global = None
start = None
stop = None
current_work_dir = None
test_dir = None
train_data_paths = None
test_data_paths = None


def index_generator(start, stop):
    """ index generator for batch method """
    i = start
    while i < stop:
        yield (i, i + batch_size_global)
        i += batch_size_global


def init(batch_size):
    global batch_size_global, index_generator_global, train_class_size, stop, current_work_dir, train_data_paths, test_data_paths, test_class_size, train_class_size, start
    batch_size_global = batch_size
    current_work_dir = os.getcwd()
    train_data_paths = [current_work_dir + '/training/' + direct for direct in
                        os.listdir(current_work_dir + '/training')]
    test_data_paths = [current_work_dir + '/test/' + dir for dir in os.listdir(current_work_dir + '/test/')]
    test_class_size = len(os.listdir(test_data_paths[0]))

    train_class_size = len(os.listdir(train_data_paths[0]))
    start = 0
    index_generator_global = index_generator(start, train_class_size)
    stop = batch_size


def reset_batch_index():

    global start, stop, batch_size_global,index_generator_global
    start = 0
    stop = batch_size_global
    index_generator_global = index_generator(start, train_class_size)


def get_test():
    test_input = []
    test_label = []

    for p_count, path in enumerate(test_data_paths):
        one_hot_enc_arr = np.zeros(len(test_data_paths))

        one_hot_enc_arr[p_count] = 1

        for pic in range(test_class_size):
            # print(path + '/{}.png'.format(pic))

            test_input.append(cv2.imread(path + '/{}.png'.format(pic), cv2.IMREAD_GRAYSCALE))
            test_label.append(one_hot_enc_arr)

    test_input = np.expand_dims(test_input, -1)
    test_label = np.array(test_label)
    return test_input, test_label


def next_batch():
    train_input = []
    train_label = []

    global index_generator_global

    indexes = next(index_generator_global)

    for p_count, path in enumerate(train_data_paths):
        one_hot_enc_arr = np.zeros(len(train_data_paths))
        # print(p_count + 1, path.split('/')[-1], end=' ')

        one_hot_enc_arr[p_count] = 1

        for pic in range(indexes[0], indexes[1]):
            # print(path + '/{}.png'.format(pic))

            train_input.append(cv2.imread(path + '/{}.png'.format(pic), cv2.IMREAD_GRAYSCALE))
            train_label.append(one_hot_enc_arr)

    global start, stop, batch_size_global
    start += batch_size_global
    global stop
    stop += batch_size_global

    train_input = np.expand_dims(train_input, -1)
    train_label = np.array(train_label)

    return train_input, train_label
