import chess

class State(object):
    def __init__(self):
        self.board = chess.Board()

    def value(self):
        return 1

    def edges(self,count=False):
        if count:
            return self.board.legal_moves.count()
        return list(self.board.legal_moves)

    
if __name__ == "__main__":
    s = State()
    print(s.edges())