import numpy as np
import tensorflow as tf


def generate_one_hot_chess_pieces(pieces):
    pool = 'KkRrBbQqNnPp'
    one_hot_dict = {}
    one_hot_dict[' '] = np.zeros(12)
    for index in range(len(pool)):
        p = pool[index]
        one_hot_dict[p] = np.zeros(12)
        one_hot_dict[p][index] = 1.

    one_hot = []

    for piece in pieces:
        one_hot.append(one_hot_dict[piece])
    return np.array(one_hot)


def generate_one_hot_num_array(array, num_uniq):
    one_hot = []
    one_hot = np.zeros([len(array), num_uniq])

    for index in range(len(array)):
        one_hot[index][int(array[index])] = 1.
    return one_hot


def read_data(data_file):
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    dimension = data.shape[1] - 1

    target = data[:, 0]
    data_without_target = data[:, 1:dimension + 1]
    data_without_target.astype(float)
    return data_without_target, target


def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)