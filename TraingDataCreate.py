from GomokuPlay import GomokuPlay
import numpy as np
import random

def create_input(board,boardmask,turn):
    #print(board)
    randambord = [[0.0 for i in range(gomoku.BOARD_LENGTH)] for j in range(gomoku.BOARD_LENGTH)]
    randambord = np.asarray(randambord)
    for r in range(len(randambord)):
        for c in range(len(randambord[0])):
            randambord[r][c] = random.random()
    randambord *= boardmask
    max = np.argmax(randambord)
    x = max % gomoku.BOARD_LENGTH#int(input())
    y = max // gomoku.BOARD_LENGTH#int(input())
    input = np.asarray([x,y])
    #print(input)
    if turn == -1:
        # np.append(bord_histry[0],board, axis=0)
        # np.append(input_histry[0],input)
        print(board.shape)
    else:
        #print(bord_histry[1].shape)
        print(board.shape)
        print(board)
        print(bord_histry.shape)
        print(bord_histry)
        np.concatenate(bord_histry,board.copy())
        np.append(input_histry[1],input)

    return input

if __name__ == "__main__":
    bord_data=np.empty((5,5), int)
    input_data=np.empty((5,5), int)

    for i in range(1000):
        gomoku = GomokuPlay(5,3)
        #bord_histry=[2][1][gomoku.BOARD_LENGTH][gomoku.BOARD_LENGTH]
        bord_histry=np.empty((5,5), int)
        input_histry=np.asarray([[],[]])
        result = gomoku.play(create_input)
        print(bord_histry)

        if result[0]:
            #print("win:" +str(result[1]))
            if result[1] == -1:
                np.append(bord_data,bord_histry[0], axis=0)
                np.append(input_data,input_histry[0])
            else:
                np.append(bord_data,bord_histry[1], axis=0)
                np.append(input_data,input_histry[1])
        else:
            print("draw:" +str(result[1]))

    #print(bord_data)
    #np.savetxt('np_savetxt.txt', bord_data)
