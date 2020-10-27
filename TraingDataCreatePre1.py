from GomokuPlay import GomokuPlay
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop

def create_inputdata(board,boardmask,turn):
    #print(board)
    if turn == -1:
        randambord = [[0.0 for i in range(gomoku.BOARD_LENGTH)] for j in range(gomoku.BOARD_LENGTH)]
        randambord = np.asarray(randambord)
        for r in range(len(randambord)):
            for c in range(len(randambord[0])):
                randambord[r][c] = random.random()

    else:
        board2 = board.copy()
        board2 = board2.reshape(board_size*board_size)
        board2 += 1
        board2 = board2.astype('float32')
        board2 /= 2
        board2 = np.asarray([board2])
        global model
        randambord = model.predict(board2)[0]
        randambord = randambord.reshape(board_size,board_size)


    randambord *= boardmask
    max = np.argmax(randambord)
    x = max % gomoku.BOARD_LENGTH#int(inputdata())
    y = max // gomoku.BOARD_LENGTH#int(inputdata())
    inputdata = np.asarray([x,y])
    #print(inputdata)
    if turn == -1:
        global bord_histry2
        global inputdata2
        bord_histry2 = np.vstack([bord_histry2,board.reshape([board_size*board_size,])])
        inputdata2 = np.append(inputdata2,inputdata[0]+inputdata[1]*board_size)
    else:
        global bord_histry1
        global inputdata1
        bord_histry1 = np.vstack([bord_histry1,board.reshape([board_size*board_size,])])
        inputdata1 = np.append(inputdata1,inputdata[0]+inputdata[1]*board_size)

    # print(turn)
    # print(board)
    return inputdata

if __name__ == "__main__":

    model = keras.models.load_model('model.h5', compile=False)

    board_size = 10

    recraCount = 10

    counterMax = 100

    bord_histry = np.empty((0,board_size*board_size),dtype=np.int)
    inputdata_history = np.empty(0,dtype=np.int)

    for j in range(recraCount):
        count = 0
        for i in range(counterMax):
            bord_histry1 = np.zeros(board_size*board_size,dtype=np.int)
            bord_histry2 = np.zeros(board_size*board_size,dtype=np.int)
            inputdata1 = np.empty(0,dtype=np.int)
            inputdata2 = np.empty(0,dtype=np.int)
            gomoku = GomokuPlay(board_size,5)
            result = gomoku.play(create_inputdata)

            if result[0]:
                # print("win:" +str(result[1]))
                count+=result[1]
                if result[1] == -1:
                    bord_histry2 = np.delete(bord_histry2,0,0)
                    bord_histry = np.append(bord_histry,bord_histry2*-1, axis=0)
                    inputdata_history = np.append(inputdata_history,inputdata2)
                else:
                    bord_histry1 = np.delete(bord_histry1,0,0)
                    bord_histry = np.append(bord_histry,bord_histry1, axis=0)
                    inputdata_history = np.append(inputdata_history,inputdata1)
                    # print(bord_histry1)
                    # print(inputdata1)
            else:
                print("draw:" +str(result[1]))

        print("syouritu="+str(count/counterMax))
        # np.save('savedata\\bord_histry.npy',bord_histry)
        # np.save('savedata\\inputdata_history.npy',bord_histry)
        # np.save('savedata\\bord_histry.npy',bord_histry)
        # np.save('savedata\\inputdata_history.npy',inputdata_history)


    #print(bord_data)
    #np.savetxt('np_savetxt.txt', bord_data)
