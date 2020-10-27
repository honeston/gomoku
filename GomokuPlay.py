import numpy as np

class GomokuPlay:
    def play(self,create_input):
        board =  self.create_base_stracter(self.BOARD_LENGTH,self.BOARD_LENGTH,0)
        boardmask = self. create_base_stracter(self.BOARD_LENGTH,self.BOARD_LENGTH,1)
        return self.main_loop(board,boardmask,create_input)

    def create_base_stracter(self,x,y,num):
        board = [[num for i in range(x)] for j in range(y)]
        #board = [[1,0,1,1],[0,1,0,0],[0,1,1,0],[0,1,0,0]]
        return np.asarray(board)

    def main_loop(self,board,boardmask,create_input):
        turn = 1
        for i in range(self.BOARD_LENGTH*self.BOARD_LENGTH):
            inputdata = create_input(board,boardmask,turn)
            board[inputdata[1]][inputdata[0]] = turn
            boardmask[inputdata[1]][inputdata[0]] = 0
            if self.check_board(board,inputdata,turn):
                return (True,turn)
            turn *=-1
        return (False,0)

    def check_board(self,board,inputdata,turn):
        # check x
        x = self.check_board_vec(board,inputdata,np.asarray([1,0]),turn)
        # check y
        y = self.check_board_vec(board,inputdata,np.asarray([0,1]),turn)
        # check xy
        xy = self.check_board_vec(board,inputdata,np.asarray([1,1]),turn)

        if x >= self.WIN_LENGTH or y >= self.WIN_LENGTH or xy >= self.WIN_LENGTH:
            return True

    def check_board_vec(self,board,inputdata,vec,turn):
        sum = 1
        sum += self.check_node(board,inputdata + vec,vec,turn)
        vec *=-1
        sum += self.check_node(board,inputdata + vec,vec,turn)
        return sum

    def check_node(self,board,node,vec,turn):
        if node[0]<0 or node[0]>=self.BOARD_LENGTH:
            return 0
        if node[1]<0 or node[1]>=self.BOARD_LENGTH:
            return 0
        if board[node[1]][node[0]] != turn:
            return 0
        return self.check_node(board,node + vec,vec,turn) + 1

    # def create_input(self,board,boardmask,turn):
    #     x = int(input())
    #     y = int(input())
    #     return np.asarray([x,y])

    def __init__(self,board_length,win_length):
        self.WIN_LENGTH = win_length
        self.BOARD_LENGTH = board_length
