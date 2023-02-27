import torch
from state import State
from NN_train import Net
import chess
import chess.svg
from flask import Flask,Response,request
import time
import base64

class Valuator(object):
    def __init__(self):
        pth_values = torch.load("nets/value.pth")
        self.model = Net()
        self.model.load_state_dict(pth_values)

    def __call__(self,s):
        return (self.model(torch.FloatTensor(s.serialize())).item())

v = Valuator()
s = State()

#print(minimax(s,True,depth=3))
def minimax(state, maximizingPlayer, depth):
    move = []
    value = v(state)
    if depth <= 0 or state.board.is_game_over():
        return value, move
    if maximizingPlayer:
        value = -1000
        for legal_move in state.edges():
            state.board.push(legal_move)
            new_value = minimax(state, state.board.turn, depth=depth - 1)[0]
            if value < new_value:
                move = legal_move
                value = new_value
            state.board.pop()
        return value, move
    else:
        value = 1000
        for legal_move in state.edges():
            state.board.push(legal_move)
            new_value = minimax(state, state.board.turn, depth=depth - 1)[0]
            if value > new_value:
                move = legal_move
                value = new_value
            state.board.pop()
        return value, move

def explore_leaves(s,v):
    ret = []
    for e in s.edges():
        s.board.push(e)
        ret.append((v(s),e))
        s.board.pop()
    return ret

def computer_move():
    move = minimax(s,s.board.turn,depth=1)
    s.board.push(move[1])
    print('Computer move:',move)


app = Flask(__name__)

@app.route("/")
def hello_world():
    board_svg = base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')
    ret = '<html><head>'
    ret += '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>'
    ret += '<style>input { font-size: 30px; } button { font-size: 30px; }</style>'
    ret += '</head><body>'
    ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % board_svg
    ret += '<form action="/move"><input name="move" type="text"></input><input type="submit" value="Move"></form><br/>'
    return ret



@app.route("/move")
def move():
    if not s.board.is_game_over():
        move = request.args.get("move",default="")
        if move is not None and move != "":
            print("human move",move)
            s.board.push(chess.Move.from_uci(move))
            computer_move()
        else:
            print('invalid move')
    else:
        print('Game Over',s.board.result())
    return hello_world()



if __name__ == '__main__':
    app.run(debug=True)
    # move = minimax(s, s.board.turn, depth=2)
    # s.board.push(move[1])
    # print('Computer move:', move)
    # print(s.board)









# if __name__ =="__main__":
#     app = Flask(__name__)
#     while not s.board.is_game_over():
#         l = sorted(explore_leaves(s,v),key=lambda x:x[0],reverse=s.board.turn)
#         move = l[0]
#         print(move)
#         s.board.push(move[1])
#     print(s.board.result())


# if __name__ == '__main__':
#     app = Flask(__name__)
#     while not s.board.is_game_over():
#         output = minimax(s,s.board.turn,depth=1)
#         print("PLayer:{},Output:{}".format(s.board.turn,output))
#         s.board.push(output[1])
#
#     print(s.board.result())
#     print(s.board.outcome())













