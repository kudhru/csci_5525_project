import csv

import chess
import chess.syzygy
import sys

import requests
import time
import numpy as np

from draw_chessboard import expand_fen
from utils import generate_one_hot_chess_pieces
from make_fen import *


def get_class_label(wdl):
    if wdl in [1,2]:
        return 1
    elif wdl in [-1,-2]:
        return 2
    else:
        return 0

if len(sys.argv) > 1:
    num_data_points = int(sys.argv[1])
else:
    num_data_points = 10000

train_fraction = 0.8
train_size = num_data_points * train_fraction

train_file = 'chess_fen_score_offline_{0}_train.csv'.format(num_data_points)
test_file = 'chess_fen_score_offline_{0}_test.csv'.format(num_data_points)

train_writer = csv.writer(open(train_file, 'wb'))
test_writer = csv.writer(open(test_file, 'wb'))

with chess.syzygy.open_tablebases("/Users/AkhilaRasamrtaMurthih/work/wdl") as tablebases:
    for index_data in range(num_data_points):
        if index_data % 2000 == 0:
            print index_data
        fen = make_fen(6)
        board = chess.Board(fen)

        try:
            wdl = tablebases.get_wdl(board)
            if wdl is not None:
                one_hot = generate_one_hot_chess_pieces(expand_fen(fen.split(' ')[0]))
                class_label = get_class_label(wdl)
                # append the label (class label) at the beginning of the flattened one_hot array
                one_hot = np.insert(one_hot, 0, class_label)

                # write the flattened one_hot data along with its label to the csv file.
                if index_data < train_size:
                    train_writer.writerow(one_hot)
                else:
                    test_writer.writerow(one_hot)
            # else:
                # fen_url = fen.replace(' ', '%20')
                # url ='https://syzygy-tables.info/api/v2?fen=6k1/q7/5p2/8/8/P7/4P3/7K%20w%20-%20-%200%201'
                # url = 'https://syzygy-tables.info/api/v2?fen={0}'.format(fen_url)
                # response = requests.get(url)
                # if response.status_code == 200:
                #     data = response.json()
                #     wdl = data['wdl']
                #     dtz = data['dtz']
                #     train_writer.writerow([wdl, dtz, fen, 6])
                #     print response.status_code, index_data, 6, fen
                # else:
                #     # print response.status_code, response.content, index_data, 6, fen
                #     if response.status_code == 429:
                #         time.sleep(5)
        except BaseException:
            print 'error {0}'.format(index_data)
