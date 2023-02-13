import chess.pgn
import os
from code.state import State
import numpy as np


def get_dataset(sample_num = None):
    X,Y = [],[]
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    pop = 0
    for games in os.listdir("../train_data"):
        pgn = open(os.path.join("../train_data", games))
        while 1:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            res = game.headers['Result']
            if res not in values:
                break
            value = values[res]
            board = game.board()
            for i,move in enumerate(game.main_line()):
                board.push(move)
                serial = State(board).serialize()
                X.append(serial) #input board
                Y.append(value)  #value
            print('parsing game:', pop, ',Got', len(X), 'examples')
            if sample_num is not None and len(X) >= sample_num:
                return X,Y
            pop += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    X,Y = get_dataset(1e5)
    np.savez("../processed/dataset.npz", X=X, Y=Y)







