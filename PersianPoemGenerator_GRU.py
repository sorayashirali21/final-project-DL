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
#%%

drive.mount('/content/drive')
%cd '/content/drive/My Drive/Persian Poem Generator'



def getCharText(file_list):
    text = ''
    for dir in file_list:
        with open(file=path+dir, mode="r", encoding="utf8") as file:
            text = text + '\n' + file.read()
    return text



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
    model.add(GRU(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars), activation='softmax'))


    def sample(preds, temperature=1.0):
        # function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def on_epoch_end(epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(maxlen):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

            print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


            callbacks = [print_callback]

            optimizer = RMSprop(lr=0.01)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)

            # params
            batch_size = 256
            epochs = 50

            # train
            model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

            # Save Model and Weights
            model.save('SaveModel/Model_Poem_GRU.h5')
            print("Model saved.")

            # Load saved model
            model = load_model('SaveModel/Model_Poem_GRU.h5')
            print("Model loaded.")

            # Defining function to generate new text based on the network's learnings

            def generate_text(length, diversity, sentence):
                generated = ''

                while (len(sentence) != maxlen):
                    sentence += ' '

                for i in range(length):
                    x_pred = np.zeros((1, maxlen, len(chars)))

                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = indices_char[next_index]

                    sentence = sentence[1:] + next_char
                    generated += next_char

                return generated

            # generate second of persian poem
            for diversity in np.arange(0.1, 1.0, 0.1):
                nVerses = 1
                hemistich_first = sentence_first[nVerses]
                hemistich_second = sentence_second[nVerses]

                while (len(hemistich_first) != maxlen):
                    hemistich_first += ' '

                while (len(hemistich_second) != maxlen):
                    hemistich_second += '\n'

                generated = generate_text(maxlen, diversity, hemistich_first)
                error = wer(hemistich_second, generated)

                print('diversity : %.1f' % diversity)
                print('First : ', hemistich_first)
                print('Second : ', generated)
                print('WER : %.2f' % error)
                print('----------------------------')

                def sample(preds, temperature=1.0):
                    # function to sample an index from a probability array
                    preds = np.asarray(preds).astype('float64')
                    preds = np.log(preds) / temperature
                    exp_preds = np.exp(preds)
                    preds = exp_preds / np.sum(exp_preds)
                    probas = np.random.multinomial(1, preds, 1)
                    return np.argmax(probas)

                def on_epoch_end(epoch, logs):
                    # Function invoked at end of each epoch. Prints generated text.
                    print()
                    print('----- Generating text after Epoch: %d' % epoch)

                    start_index = random.randint(0, len(text) - maxlen - 1)
                    for diversity in [0.2, 0.5, 1.0, 1.2]:
                        print('----- diversity:', diversity)

                        generated = ''
                        sentence = text[start_index: start_index + maxlen]
                        generated += sentence
                        print('----- Generating with seed: "' + sentence + '"')
                        sys.stdout.write(generated)

                        for i in range(maxlen):
                            x_pred = np.zeros((1, maxlen, len(chars)))
                            for t, char in enumerate(sentence):
                                x_pred[0, t, char_indices[char]] = 1.

                            preds = model.predict(x_pred, verbose=0)[0]
                            next_index = sample(preds, diversity)
                            next_char = indices_char[next_index]

                            generated += next_char
                            sentence = sentence[1:] + next_char

                            sys.stdout.write(next_char)
                            sys.stdout.flush()
                        print()

                        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

                        # a function to reduce the learning rate each time the learning
                        reduce_alpha = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                         patience=1, min_lr=0.001)

                        callbacks = [print_callback, reduce_alpha]

                        optimizer = RMSprop(lr=0.01)
                        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

                        # params
                        batch_size = 256
                        epochs = 50

                        # train
                        model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

                        # Save Model and Weights
                        model.save('SaveModel/Model_Poem_LSTM.h5')
                        print("Model saved.")

                        # Load saved model
                        model = load_model('SaveModel/Model_Poem_LSTM.h5')
                        print("Model loaded.")

                        ## **Test Model**

                        # Defining function to generate new text based on the network's learnings

                        def generate_text(length, diversity, sentence):
                            generated = ''

                            while (len(sentence) != maxlen):
                                sentence += ' '

                            for i in range(length):
                                x_pred = np.zeros((1, maxlen, len(chars)))

                                for t, char in enumerate(sentence):
                                    x_pred[0, t, char_indices[char]] = 1.

                                preds = model.predict(x_pred, verbose=0)[0]
                                next_index = sample(preds, diversity)
                                next_char = indices_char[next_index]

                                sentence = sentence[1:] + next_char
                                generated += next_char

                            return generated
                        # %%




# generate second of persian poem
for diversity in np.arange(0.1, 1.0, 0.1):
  nVerses = 1
  hemistich_first = sentence_first[nVerses]
  hemistich_second = sentence_second[nVerses]

  while(len(hemistich_first)!=maxlen):
    hemistich_first += ' '

  while(len(hemistich_second)!=maxlen):
    hemistich_second += '\n'

  generated = generate_text(maxlen, diversity, hemistich_first)
  error = wer(hemistich_second, generated)

  print('diversity : %.1f' % diversity)
  print('First : ', hemistich_first)
  print('Second : ', generated)
  print('WER : %.2f' % error)
  print('----------------------------')


