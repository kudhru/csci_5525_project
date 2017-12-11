


#####################################################################################################################################################################################################################################
    #                                           FEN GENERATOR FOR ROOK-KING-ROOK ENDINGS
# AUTHOR: Venugopal Mani (FEN), Dhruv Kumar (Board generator)
# DATE: Dec 5th, 2017
# VERSION: 1.1
# COPYRIGHT: All Rights Reserved

#####################################################################################################################################################################################################################################
import random
import numpy as np

import re
from PIL import Image
from PIL import ImageDraw




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

    #print("White King: ",w_king,"Black King: ",b_king,"Rook: ",rook)

    board = np.zeros(shape = (8,8))
        
    board[w_king[0],w_king[1]]= np.inf
    board[b_king[0],b_king[1]] = -np.inf 
    board[rook[0],rook[1]] = which_rook 
    

    #print_chess_board(board)
    
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

    
    return total,label
                










#! /usr/bin/env python
'''Code to draw chess board and pieces.

FEN notation to describe the arrangement of peices on a chess board.

White pieces are coded: K, Q, B, N, R, P, for king, queen, bishop,
rook knight, pawn. Black pieces use lowercase k, q, b, n, r, p. Blank
squares are noted with digits, and the "/" separates ranks.

As an example, the game starts at:

rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR

See: http://en.wikipedia.org/wiki/Forsyth-Edwards_Notation
'''


class BadChessboard(ValueError):
    pass


def expand_blanks(fen):
    '''Expand the digits in an FEN string into spaces

    >>> expand_blanks("rk4q3")
    'rk    q   '
    '''

    def expand(match):
        return ' ' * int(match.group(0))

    return re.compile(r'\d').sub(expand, fen)


def check_valid(expanded_fen):
    '''Asserts an expanded FEN string is valid'''
    match = re.compile(r'([KQBNRPkqbnrp ]{8}/){8}$').match
    if not match(expanded_fen + '/'):
        raise BadChessboard()


def expand_fen(fen):
    '''Preprocesses a fen string into an internal format.

    Each square on the chessboard is represented by a single 
    character in the output string. The rank separator characters
    are removed. Invalid inputs raise a BadChessboard error.
    '''
    expanded = expand_blanks(fen)
    check_valid(expanded)
    return expanded.replace('/', '')


def draw_board(n=8, sq_size=(20, 20)):
    '''Return an image of a chessboard.

    The board has n x n squares each of the supplied size.'''
    from itertools import cycle
    def square(i, j):
        return i * sq_size[0], j * sq_size[1]

    opaque_grey_background = 192, 255
    board = Image.new('LA', square(n, n), opaque_grey_background)
    draw_square = ImageDraw.Draw(board).rectangle
    whites = ((square(i, j), square(i + 1, j + 1))
              for i_start, j in zip(cycle((0, 1)), range(n))
              for i in range(i_start, n, 2))
    for white_square in whites:
        draw_square(white_square, fill='white')
    return board


class DrawChessPosition(object):
    '''Chess position renderer.

    Create an instance of this class, then call 
    '''

    def __init__(self):
        '''Initialise, preloading pieces and creating a blank board.'''
        self.n = 8
        self.create_pieces()
        self.create_blank_board()

    def create_pieces(self):
        '''Load the chess pieces from disk.

        Also extracts and caches the alpha masks for these pieces. 
        '''
        whites = 'KQBNRP'
        piece_images = dict(
            zip(whites, (Image.open('pieces/w%s.png' % p) for p in whites)))
        blacks = 'kqbnrp'
        piece_images.update(dict(
            zip(blacks, (Image.open('pieces/%s.png' % p) for p in blacks))))
        piece_sizes = set(piece.size for piece in piece_images.values())
        # Sanity check: the pieces should all be the same size
        assert len(piece_sizes) == 1
        self.piece_w, self.piece_h = piece_sizes.pop()
        self.piece_images = piece_images
        self.piece_masks = dict((pc, img.split()[3]) for pc, img in
                                self.piece_images.iteritems())

    def create_blank_board(self):
        '''Pre-render a blank board.'''
        self.board = draw_board(sq_size=(self.piece_w, self.piece_h))

    def point(self, i, j):
        '''Return the top left of the square at (i, j).'''
        w, h = self.piece_w, self.piece_h
        return i * h, j * w

    def square(self, i, j):
        '''Return the square at (i, j).'''
        t, l = self.point(i, j)
        b, r = self.point(i + 1, j + 1)
        return t, l, b, r

    def draw(self, fen):
        '''Return an image depicting the input position.

        fen - the first record of a FEN chess position.
        Clients are responsible for resizing this image and saving it,
        if required.
        '''
        board = self.board.copy()
        pieces = expand_fen(fen)
        images, masks, n = self.piece_images, self.piece_masks, self.n
        pts = (self.point(i, j) for j in range(n) for i in range(n))

        def not_blank(pt_pc):
            return pt_pc[1] != ' '

        for pt, piece in filter(not_blank, zip(pts, pieces)):
            board.paste(images[piece], pt, masks[piece])
        return board


# image = a.draw('8/7K/8/4k3/8/8/3R4/8')


counter = 1
label = {}

for _ in range(10000):
    
    
    tup = generate_position()

    ans = tup[0]
    lab = tup[1]

    print("Train",counter,":",ans)

    label[counter] = lab

    a = DrawChessPosition()
    image = a.draw(ans)
    filename = "train2/train"+str(counter)+".png"
    counter = counter+1
    image.save(filename)
    
print("Congrats Miner! You have generated the trainset")

counter = 1
with open("train/labels.csv","w") as f:

    for _ in range(10000):
        
        f.write(str(counter))
        f.write(",")
        f.write(str(label[counter]))
        f.write("\n")
        counter = counter + 1




                
                
    



counter = 1
label = {}

for _ in range(20):
    
    
    tup = generate_position()

    ans = tup[0]
    lab = tup[1]

    print("Test",counter,":",ans)

    label[counter] = lab

    a = DrawChessPosition()
    image = a.draw(ans)
    filename = "test2/test"+str(counter)+".png"
    counter = counter+1
    image.save(filename)
    
print("Congrats Miner! You have generated the testset")

counter = 1
with open("test/labels.csv","w") as f:

    for _ in range(2000):
        
        f.write(str(counter))
        f.write(",")
        f.write(str(label[counter]))
        f.write("\n")
        counter = counter + 1
