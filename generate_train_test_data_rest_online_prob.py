import csv

import numpy
import sys

import requests
import time
from make_fen import *




if len(sys.argv) > 1:
    num_data_points = int(sys.argv[1])
else:
    num_data_points = 10000

if len(sys.argv) > 2:
    num_pieces = sys.argv[2].split(',')
    num_pieces = [int(x) for x in num_pieces]
else:
    num_pieces = [3,4,5,6]

if len(sys.argv) > 3:
    save_file_name = sys.argv[3]
else:
    save_file_name = 'chess_fen_score_{0}.csv'.format(num_data_points)


writer = csv.writer(open(save_file_name, 'a+'))

sleep_counter = 3
for index_data in range(num_data_points):
    if index_data % 100 == 0:
        print index_data
    for piece in num_pieces:
        fen = make_fen(piece)
        fen_url = fen.replace(' ', '%20')
        # url ='https://syzygy-tables.info/api/v2?fen=6k1/q7/5p2/8/8/P7/4P3/7K%20w%20-%20-%200%201'
        url ='https://syzygy-tables.info/api/v2?fen={0}'.format(fen_url)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            wdl = data['wdl']
            dtz = data['dtz']
            writer.writerow([wdl, dtz, fen, piece])
        else:
            if response.status_code == 429:
                time.sleep(20)
            print response.status_code, response.content, index_data, piece, fen
        time.sleep(sleep_counter)


