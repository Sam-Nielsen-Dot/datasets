import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

if True:
    #load from .json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    model = loaded_model

filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

file = open("frankenstein-2.txt").read()
def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

# preprocess the input data, make tokens
processed_inputs = tokenize_words(file)

chars = sorted(list(set(processed_inputs)))

num_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_num = dict((c, i) for i, c in enumerate(chars))

random_seed = "hello world"
pattern = []

for k in random_seed:
    pattern.append(char_to_num[k])


file_object = open('sample.txt', 'a')
# Append 'hello' at the
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(38)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    print(result, end="")
    file_object.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
file_object.close()
print("Done")

