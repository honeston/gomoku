from GomokuPlay import GomokuPlay
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop

def create_input(board,boardmask,turn):
    #print(board)
    # if turn == -1:

    # else:
    board2 = board.copy()
    board2 = board2.reshape(board_size*board_size)
    board2 += 1
    board2 = board2.astype('float32')
    board2 /= 2
    board2 = np.asarray([board2])
    if turn == -1:
        global model
        randambord = model.predict(board2)[0]
    else:
        global model2
        randambord = model2.predict(board2)[0]
    randambord = randambord.reshape(board_size,board_size)

    randambord2 = [[0.0 for i in range(gomoku.BOARD_LENGTH)] for j in range(gomoku.BOARD_LENGTH)]
    randambord2 = np.asarray(randambord2)
    if turn == -1:
        for r in range(len(randambord2)):
            for c in range(len(randambord2[0])):
                randambord2[r][c] = random.random()
        randambord *= (randambord2/4+0.75)

    randambord *= boardmask
    max = np.argmax(randambord)
    x = max % gomoku.BOARD_LENGTH#int(input())
    y = max // gomoku.BOARD_LENGTH#int(input())
    input = np.asarray([x,y])
    #print(input)
    if turn == -1:
        global bord_histry2
        global input2
        bord_histry2 = np.vstack([bord_histry2,board.reshape([board_size*board_size,])])
        input2 = np.append(input2,input[0]+input[1]*board_size)
    else:
        global bord_histry1
        global input1
        bord_histry1 = np.vstack([bord_histry1,board.reshape([board_size*board_size,])])
        input1 = np.append(input1,input[0]+input[1]*board_size)

    # print(turn)
    return input

def train(count):
    boadsize = 10
    batch_size = 128
    num_classes = boadsize*boadsize
    epochs = 20


    # the data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    global bord_histryp
    global input_historyp
    global bord_histryenp
    global input_historyenp


    x_train = bord_histryp
    y_train = input_historyp

    x_test = np.load('savedata\\bord_histryt.npy')
    y_test = np.load('savedata\\input_historyt.npy')

    x_train = x_train.reshape(int(x_train.size/(boadsize*boadsize)), (boadsize*boadsize))
    x_test = x_test.reshape(int(x_test.size/(boadsize*boadsize)), (boadsize*boadsize))

    x_train *= -1
    x_train += 1
    x_test += 1

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 2
    x_test /= 2




    x_trainen = bord_histryenp
    y_trainen = input_historyenp

    x_trainen = x_trainen.reshape(int(x_trainen.size/(boadsize*boadsize)), (boadsize*boadsize))

    x_trainen *= -1

    x_trainen += 1

    x_trainen = x_trainen.astype('float32')
    x_trainen /= 20



    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_trainen = keras.utils.to_categorical(y_trainen, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    global model
    # if(count==0):
    if(False):
        model = Sequential()
        #model = keras.models.load_model('model.h5', compile=False)


        model.add(Dense(32, activation='relu', input_shape=(100,)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation='softmax'))

        model.summary()
    if(count==0):
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

    if(x_train.size>0):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
        model.save('model.h5', include_optimizer=False)



    global model2
    # if(count==0):
    if(False):
        model2 = Sequential()
        #model2 = keras.model2s.load_model2('model2.h5', compile=False)


        model2.add(Dense(32, activation='relu', input_shape=(100,)))
        model2.add(Dropout(0.2))
        model2.add(Dense(64, activation='relu'))
        model2.add(Dropout(0.2))

        model2.add(Dense(num_classes, activation='softmax'))

        model2.summary()

    if(count==0):
        model2.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

    if(x_trainen.size>0):
        history = model2.fit(x_trainen, y_trainen,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
    # score = model2.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
        model2.save('model2.h5', include_optimizer=False)


if __name__ == "__main__":

    model = keras.models.load_model('model.h5', compile=False)
    model2 = keras.models.load_model('model2.h5', compile=False)

    board_size = 10

    recraCount = 10000

    counterMax = 1000



    for j in range(recraCount):
        count = 0
        bord_histry = np.empty((0,board_size*board_size),dtype=np.int)
        input_history = np.empty(0,dtype=np.int)
        bord_histryen = np.empty((0,board_size*board_size),dtype=np.int)
        input_historyen = np.empty(0,dtype=np.int)
        for i in range(counterMax):
            bord_histry1 = np.zeros(board_size*board_size,dtype=np.int)
            bord_histry2 = np.zeros(board_size*board_size,dtype=np.int)
            input1 = np.empty(0,dtype=np.int)
            input2 = np.empty(0,dtype=np.int)
            gomoku = GomokuPlay(board_size,5)
            result = gomoku.play(create_input)

            if result[0]:
                # print("win:" +str(result[1]))
                count1=0
                count+=result[1]
                if result[1] == -1:
                    bord_histry2 = np.delete(bord_histry2,0,0)
                    bord_histryen = np.append(bord_histryen,bord_histry2*-1, axis=0)
                    input_historyen = np.append(input_historyen,input2)
                else:
                    bord_histry1 = np.delete(bord_histry1,0,0)
                    bord_histry = np.append(bord_histry,bord_histry1, axis=0)
                    input_history = np.append(input_history,input1)
            else:
                print("draw:" +str(result[1]))
        print(bord_histry)
        print(input_history)
        print("syouritu="+str(count/counterMax))
        # np.save('savedata\\bord_histry.npy',bord_histry)
        # np.save('savedata\\input_history.npy',bord_histry)
        global bord_histryp
        global input_historyp
        global bord_histryenp
        global input_historyenp

        bord_histryp = bord_histry
        input_historyp = input_history
        bord_histryenp = bord_histryen
        input_historyenp = input_historyen

        print('maincount'+str(j)+'####################')

        train(j)

    #print(bord_data)
    #np.savetxt('np_savetxt.txt', bord_data)
