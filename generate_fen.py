#####################################################################################################################################################################################################################################
    #                                           FEN GENERATOR FOR ROOK-KING-ROOK ENDINGS
# AUTHOR: Venugopal Mani
# DATE: Dec 5th, 2017
# VERSION: 1.0
# COPYRIGHT: All Rights Reserved

#####################################################################################################################################################################################################################################
import random
import numpy as np

# A method that generates any random square for a chess board

piece_map = {}

piece_map[np.inf] = 'K'
piece_map[-np.inf] = 'k'
piece_map[5] = 'R'
piece_map[-5] = 'r'

move= {}
move[0] = 'b'
move[1] = 'w'

def generate_chess_square():

    file = random.randint(0,7)
    rank = random.randint(0,7)

    return(file,rank)


# A method to print a given board
def print_chess_board(board):
    for i in range(8):

        for j in range(8):
            print(board[i][j]," ",end="")

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

    print("White King: ",w_king,"Black King: ",b_king,"Rook: ",rook)

    board = np.zeros(shape = (8,8))
        
    board[w_king[0],w_king[1]]= np.inf
    board[b_king[0],b_king[1]] = -np.inf 
    board[rook[0],rook[1]] = which_rook 
    

    print_chess_board(board)
    
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

    total = total + " "+move[random.randint(0, 1)]+" -- 0 1"
    print(total)
                
generate_position()


