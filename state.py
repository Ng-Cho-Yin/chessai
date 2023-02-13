import chess
import numpy as np


class State(object):
    def __init__(self,board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def serialize(self):
        assert self.board.is_valid()
        b_state = np.zeros(64,np.uint8) #b_state = board state
        for i in range(64):
            p_pos = self.board.piece_at(i) #p_pos = piece position
            if p_pos is not None:
                #print(i,p_pos.symbol())
                b_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                              "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[p_pos.symbol()]

        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert b_state[0] == 4
            b_state[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert b_state[7] == 4
            b_state[7] = 7
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert b_state[56] == 8+4
            b_state[56] = 8+7
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert b_state[63] == 8+4
            b_state[63] = 8+7
        if self.board.ep_square is not None:
            assert b_state[self.board.ep_square] == 0
            b_state[self.board.ep_square] = 8
        b_state = b_state.reshape(8,8)


        state = np.zeros((5,8,8),np.uint8)


        state[0] = (b_state >> 3) & 1
        state[1] = (b_state >> 2) & 1
        state[2] = (b_state >> 1) & 1
        state[3] = (b_state >> 0) & 1
        state[4] = (self.board.turn*1.0)

        return state

    def edges(self):
        return list(self.board.legal_moves)


if __name__ == '__main__':
    s = State()
    print(s.edges())
    print(s.serialize())















