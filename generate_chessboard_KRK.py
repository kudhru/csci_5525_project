


#####################################################################################################################################################################################################################################
    #                                           FEN GENERATOR FOR ROOK-KING-ROOK ENDINGS
# AUTHOR: Venugopal Mani (FEN), Dhruv Kumar (Board generator)
# DATE: Dec 5th, 2017
# VERSION: 1.1
# COPYRIGHT: All Rights Reserved

#####################################################################################################################################################################################################################################
import csv
import random
import numpy as np

import sys

from scipy import misc
from PIL import Image





# A method that generates any random square for a chess board
from draw_chessboard import DrawChessPosition, expand_fen
from generate_one_hot import generate_one_hot_chess_pieces

piece_map = {}

piece_map[np.inf] = 'K'
piece_map[-np.inf] = 'k'
piece_map[5] = 'R'
piece_map[-5] = 'r'

move= {}
move[0] = 'b'
move[1] = 'w'

def generate_chess_square():
    file = random.randint(0, 7)
    rank = random.randint(0, 7)
    return(file,rank)


# A method to print a given board
def print_chess_board(board):
    for i in range(8):
        for j in range(8):
            print(board[i][j]," ",)
        print("\n")


def generate_position():
    label = random.randint(0,1)
    if(label == 0):
        which_rook = -5
    else:
        which_rook = 5

    w_king = (0,0)
    b_king = (0,0)
    rook = (0,0)

    while w_king == b_king or w_king == rook or b_king == rook:

        w_king = generate_chess_square()
        b_king = generate_chess_square()
        rook = generate_chess_square()

    # print("White King: ",w_king,"Black King: ",b_king,"Rook: ",rook)

    board = np.zeros(shape = (8,8))
        
    board[w_king[0],w_king[1]]= np.inf
    board[b_king[0],b_king[1]] = -np.inf 
    board[rook[0],rook[1]] = which_rook 

    # print_chess_board(board)
    
    total = ""
    for i in range(8):
        curr = board[i]

        this_row = ""
        
        walker = 0

        for j in range(8):
            if board[i,j] == 0:
                walker = walker + 1

            else:
                if (walker != 0):
                    this_row = this_row + str(walker)
                    
                this_row += piece_map[board[i,j]]
                
                walker = 0
                
        if walker != 0:
            this_row = this_row+str(walker)

        total = total+"/"+this_row

    total = total[1:]

    
    return total, label





# image = a.draw('8/7K/8/4k3/8/8/3R4/8')

def generate_chess_image_flattened_data(num_data_points, save_file_name):
    writer = csv.writer(open(save_file_name, 'wb'))
    for iter in range(num_data_points):

        if iter % 5000 == 0:
            print "iteration {0}".format(iter)

        # generate the FEN randomly
        tup = generate_position()
        ans = tup[0]
        lab = tup[1]

        # draw the positions on the chessboard and save it in an image
        a = DrawChessPosition()
        image = a.draw(ans)
        image = image.resize((50,50), Image.ANTIALIAS)
        filename = "temp.png"
        image.save(filename)

        # convert the saved image data to a flat array
        image = misc.imread(filename, flatten=True)
        image = image.flatten()

        # append the label (class label) at the beginning of the flattened image array
        image = np.insert(image, 0, lab)

        # write the flattened image data along with its label to the csv file.
        writer.writerow(image)

    print("Congrats Miner! You have generated the dataset")






# this function generates a fen randomly and then then for every position (rook, king etc), generates a one-hot vector.
# finally all the one hot vectors are stored for a chessboard in a flat vector.
def generate_chess_one_hot_representation(num_data_points, save_file_name):
    writer = csv.writer(open(save_file_name, 'wb'))
    for iter in range(num_data_points):

        if iter % 5000 == 0:
            print "iteration {0}".format(iter)

        # generate the FEN randomly
        tup = generate_position()
        fen = tup[0]
        lab = tup[1]

        pieces = expand_fen(fen)

        one_hot = generate_one_hot_chess_pieces(pieces)

        # append the label (class label) at the beginning of the flattened one_hot array
        one_hot = np.insert(one_hot, 0, lab)

        # write the flattened one_hot data along with its label to the csv file.
        writer.writerow(one_hot)

    print("Congrats Miner! You have generated the dataset")

if sys.argv[1] is not None:
    num_train_data_points = int(sys.argv[1])
else:
    num_train_data_points = 10000

if sys.argv[2] is not None:
    num_test_data_points = int(sys.argv[2])
else:
    num_test_data_points = 1000

train_dir = 'train'
test_dir = 'test'

generate_chess_image_flattened_data(num_train_data_points, 'chess_train_{0}.csv'.format(num_train_data_points))
generate_chess_image_flattened_data(num_train_data_points, 'chess_train_{0}.csv'.format(num_train_data_points))

# generate_chess_one_hot_representation(num_train_data_points, 'chess_train_one_hot_{0}.csv'.format(num_train_data_points))
# generate_chess_one_hot_representation(num_test_data_points, 'chess_test_one_hot_{0}.csv'.format(num_test_data_points))