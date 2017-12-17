import numpy as np


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