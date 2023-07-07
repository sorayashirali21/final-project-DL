'!pip install jiwer'

import numpy as np
import os
import random
import sys
import io
import nltk

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.utils.data_utils import get_file

from __future__ import print_function

from jiwer import wer

from google.colab import drive

drive.mount('/content/drive')
%cd '/content/drive/My Drive/Persian Poem Generator'
#%%


drive.mount('/content/drive')
%cd '/content/drive/My Drive/Persian Poem Generator'
#%%
def getCharText(file_list):
    text = ''
    for dir in file_list:
        with open(file=path+dir, mode="r", encoding="utf8") as file:
            text = text + '\n' + file.read()
    return text
#%%



path = 'Database/Persian Poem/'
file_list = ['babaafzal_norm.txt']


InputText = getCharText(file_list)
InputText = InputText[6:]

print('corpus length:', len(InputText))

InputText



sentence_all = InputText.split("\n")

sentence_first = []
sentence_second = []

for i in range(len(sentence_all)):
  if(i%2==0):
    sentence_first.append(sentence_all[i])
  else:
    sentence_second.append(sentence_all[i])

nRow = min(len(sentence_first), len(sentence_second))

sentence_first = sentence_first[:nRow]
sentence_second = sentence_second[:nRow]

nMax = 0

for i in range(nRow):
    n = int(len(sentence_first[i]))
    if (n > nMax):
        nMax_index = i
        nMax = n

    n = int(len(sentence_second[i]))
    if (n > nMax):
        nMax_index = i
        nMax = n

print('maximum length of first and second hemistich : ', nMax)



for i in range(nRow):
  n = int(len(sentence_first[i]))
  sentence_first[i] = sentence_first[i] + (' '*(nMax-n))

  n = int(len(sentence_second[i]))
  sentence_second[i] = sentence_second[i] + (' '*(nMax-n))

  text = ''

  for i in range(nRow):
      text += (sentence_first[i] + ' ' + sentence_second[i] + '\n')

      print(text)

      # Sorting all the unique characters present in the text
      chars = sorted(list(set(text)))

      # Creating dictionaries to map each character to an index
      char_indices = dict((c, i) for i, c in enumerate(chars))
      indices_char = dict((i, c) for i, c in enumerate(chars))

      print('total chars:', len(chars))
      print('chars:', chars)

      maxlen = nMax + 1
      step = 1

      sentences = []
      next_chars = []

      for i in range(0, len(text) - maxlen, step):
          sentences.append(text[i: i + maxlen])
          next_chars.append(text[i + maxlen])

      print('nb sequences:', len(sentences))
      print('nb next_chars:', len(next_chars))
      print('sequences:', sentences)
      print('next_chars:', next_chars)

      # Hot encoding each character into a boolean vector
      print('Vectorization...')

      x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
      y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

      for i, sentence in enumerate(sentences):
          for t, char in enumerate(sentence):
              x[i, t, char_indices[char]] = 1
          y[i, char_indices[next_chars[i]]] = 1

          # Building the LSTM network for the task
          print('Build model...')

          model = Sequential()
          model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
          model.add(Dropout(0.2))
          model.add(LSTM(256))
          model.add(Dropout(0.2))
          model.add(Dense(len(chars), activation='softmax'))