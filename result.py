from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import copy
import numpy
import json

raw_text = str(open("A Christmas Carol.txt", encoding='utf-8').read())[:1000]
slice_size = 100

character_map = json.load(open('model_config.json'))
character_map_length = len(character_map)

dataX = []
dataY = []
empty_frame = [0 for i in range(character_map_length)]
for i in range(len(raw_text) - slice_size - 1):
    dataFrameX = []
    for j in range(slice_size+1):
        temp = copy.deepcopy(empty_frame)
        temp[character_map[raw_text[i+j]]] = 1
        dataFrameX.append(temp)
    dataX.append(dataFrameX[:-1])
    dataY.append(dataFrameX[-1])

dataX = numpy.array(dataX)
dataY = numpy.array(dataY)

model = Sequential()
model.add(LSTM(1024, input_shape=(dataX.shape[1], dataX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(dataY.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(dataX, dataY, epochs=50, batch_size=64, callbacks=callbacks_list)

