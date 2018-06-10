import json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
#specify the path of the file where you clone all the reviews.
path = 'reviewset.txt'
text = open(path).read().lower()
print('No of total characters in the dataset:', len(text))

ci = json.loads(open('ci.txt').read())
ic = json.loads(open('ic.txt').read())
chars = sorted(ci.keys())
print('The final set of characters we encounter in the dataset are:')
print(ic)
print('total no of characters:', len(chars))
maxlen = 512
step = 3
sentence_list = []
followed_characters = []
for i in range(0, len(text) - maxlen, step):
    sentence_list.append(text[i: i + maxlen])
    followed_characters.append(text[i + maxlen])
X = np.zeros((len(sentence_list), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentence_list), len(chars)), dtype=np.bool)
for i, sequence in enumerate(sentence_list):
    for t, char in enumerate(sequence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print('Initializing modules.....')
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(512, return_sequences=False))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

weight_updater= Adam(lr=0.002)
model.compile(loss='categorical_crossentropy', optimizer=weight_updater)
model.load_weights("update_weights")
def sample(preds, temperature=0.6):
    predicor= np.asarray(predictor).astype('float64')
    predictor = np.log(predictor) / temperature
    exp_preds = np.exp(predictor)
    predictor = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, preds, 1)
    return np.argmax(probabilities)

for iteration in range(1,60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    x = np.zeros((1, maxlen, len(chars)))
    predictor = model.predict(x, verbose=0)[0]
    
    model.fit(X, y, batch_size=128, epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for specification in [0.2, 0.4, 0.6]:
        print()
        print('----- specification:', specification)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- predcicting with the beginner: "' + sentence + '"')
        sys.stdout.write(generated)
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, ci[char]] = 1.

            predictor = model.predict(x, verbose=0)[0]
            next_index = sample(predictor, generated)
            next_char = ic[str(next_index)]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
model.save_weights("weight_updater")