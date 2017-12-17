import numpy

wbf = ''
bbf = ''
chess_board = []


def chess_board_init():
    global wbf
    global bbf
    global chess_board

    wbf = ''
    bbf = ''

    chess_board = []
    for iter in range(8):
        board_row = []
        for j in range(8):
            board_row.append('')
        chess_board.append(board_row)

chess_board_init()

moves = {}
moves['K'] = []
moves['K'].append([0,-1,1])
moves['K'].append([1,-1,1])
moves['K'].append([1,0,1])
moves['K'].append([1,1,1])
moves['K'].append([0,1,1])
moves['K'].append([-1,1,1])
moves['K'].append([-1,0,1])
moves['K'].append([-1,-1,1])

moves['Q'] = []
moves['Q'].append([0,-1,7])
moves['Q'].append([1,-1,7])
moves['Q'].append([1,0,7])
moves['Q'].append([1,1,7])
moves['Q'].append([0,1,7])
moves['Q'].append([-1,1,7])
moves['Q'].append([-1,0,7])
moves['Q'].append([-1,-1,7])
moves['q']= moves['Q']

moves['R'] = []
moves['R'].append([0,-1,7])
moves['R'].append([1,0,7])
moves['R'].append([0,1,7])
moves['R'].append([-1,0,7])
moves['r']= moves['R']

moves['B'] = []
moves['B'].append([1,-1,7])
moves['B'].append([1,1,7])
moves['B'].append([-1,1,7])
moves['B'].append([-1,-1,7])
moves['b']= moves['B']

moves['N']= []
moves['N'].append([1,-2,1])
moves['N'].append([2,-1,1])
moves['N'].append([2,1,1])
moves['N'].append([1,2,1])
moves['N'].append([-1,2,1])
moves['N'].append([-2,1,1])
moves['N'].append([-2,-1,1])
moves['N'].append([-1,-2,1])
moves['n']= moves['N']

moves['P'] = []
moves['P'].append([-1,-1,1])
moves['P'].append([1,-1,1])

moves['p']=[]
moves['p'].append([-1,1,1])
moves['p'].append([1,1,1])

def get_empty_field():
    while True:
        x = numpy.random.randint(8)
        y = numpy.random.randint(8)
        if chess_board[x][y] == '':
            return x, y


def allowed_pos(x, y, piece):
    global wbf
    global bbf
    if piece in ['K', 'Q', 'R', 'B', 'N', 'P']:
        op_king = 'k'
    elif piece in ['k', 'q', 'r', 'b', 'n', 'p']:
        op_king = 'K'

    for d in range(len(moves[piece])):
        dx = moves[piece][d][0]
        dy = moves[piece][d][1]
        l = moves[piece][d][2]

        for s in range(l):
            px = x + s * dx
            py = y + s * dy
            if px >= 0 and py >= 0 and px < 8 and py < 8:
                if chess_board[px][py] != '':
                    if chess_board[px][py] == op_king:
                        return False
                    else:
                        s = l + 1
            else:
                s = l + 1

    # P / p not on baselines
    if piece == 'P' or piece == 'p':
        if y < 1 or y > 6:
            return False

    # B / b on black and white fields
    fc = 'w' if (x + y) % 2 == 0 else 'b'

    if piece == 'B':
        if wbf == fc:
            return False
        else:
            if wbf == '':
                wbf = fc

    if piece == 'b':
        if bbf == fc:
            return False
        else:
            if bbf == '':
                bbf = fc

    return True


def make_fen(num_pieces):
    chess_board_init()
    pool = 'QqRRrrBBbbNNnnPPPPPPPPpppppppp'
    pool_length = len(pool)


    # set black king
    k_x, k_y = get_empty_field()
    chess_board[k_x][k_y] = 'k'

    # set white king
    while True:
        x, y = get_empty_field()
        if allowed_pos(x, y, 'K'):
            chess_board[x][y] = 'K'
            break

    # set other options
    for n in range(2, num_pieces):

        # get random piece from pool
        while True:
            pos = numpy.random.randint(pool_length)
            if pool[pos] != '_':
                break

        cur_piece = pool[pos]
        pool = pool[0:pos] + '_' + pool[pos+1:]

        while True:
            x, y = get_empty_field()
            if allowed_pos(x, y, cur_piece):
                chess_board[x][y] = cur_piece
                break

    # construct FEN
    cFen = ''
    z = 0
    for y in range(8):
        for x in range(8):
            if x == 0:
                if z > 0:
                    cFen += str(z)
                if x > 0 or y > 0:
                    cFen += '/'
                z = 0
            if chess_board[x][y] == '':
                z += 1
            else:
                if z > 0:
                    cFen += str(z)
                cFen += chess_board[x][y]
                z = 0
    if z > 0:
        cFen += str(z)
    cFen += ' w - - 0 1'

    return cFen
