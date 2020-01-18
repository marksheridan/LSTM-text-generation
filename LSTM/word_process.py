import datetime

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, TimeDistributed, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K

import unicodedata
import re
import numpy as np
import os
import time
import shutil

import pandas as pd
import numpy as np
import string, os

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

f = open("data_text.txt","r")
corpus = [line for line in f]


def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, '')
    return text


def preprocess(data):
    output = []
    punct = '#$%&*+-/<=>@[\\]^_`{|}~\t\n'
    for line in data:
        pline = clean_special_chars(line.lower(), punct)
        output.append(pline)
    return output


def generate_dataset():
    processed_corpus = preprocess(corpus)
    output = []
    for line in processed_corpus:
        token_list = line
        for i in range(1, len(token_list)):
            data = []
            x_ngram = '<start> ' + token_list[:i + 1] + ' <end>'
            y_ngram = '<start> ' + token_list[i + 1:] + ' <end>'
            data.append(x_ngram)
            data.append(y_ngram)
            output.append(data)
    print("Dataset prepared with prefix and suffixes for teacher forcing technique")
    dummy_df = pd.DataFrame(output, columns=['input', 'output'])
    return output, dummy_df


class LanguageIndex():
    def __init__(self, lang):
            self.lang = lang
            self.word2idx = {}
            self.idx2word = {}
            self.vocab = set()
            self.create_index()
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i+1] = word

def max_length(t):
    return max(len(i) for i in t)

def load_dataset():
    pairs,df = generate_dataset()
    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_length_in, max_length_out = max_length(input_data), max_length(output_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")
    output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post")
    return input_data, output_data, in_lang, out_lang, max_length_in, max_length_out, df
#
#
input_data, teacher_data, input_lang, target_lang, len_input, len_target, df = load_dataset()
#
target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")
target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))



p = np.random.permutation(len(input_data))
input_data = input_data[p]
teacher_data = teacher_data[p]
target_data = target_data[p]


# pd.set_option('display.max_colwidth', -1)
# BUFFER_SIZE = len(input_data)
# BATCH_SIZE = 100
# embedding_dim = 300
# units = 10
# vocab_in_size = len(input_lang.word2idx)
# vocab_out_size = len(target_lang.word2idx)
# df.iloc[60:65]


# Create the Encoder layers first.
# encoder_inputs = Input(shape=(len_input,))
# encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)

# Use this if you dont need Bidirectional LSTM
# # encoder_lstm = LSTM(units=units, return_sequences=True, return_state=True)
# # #encoder_lstm = CuDNNLSTM(units=units, return_sequences=True, return_state=True)
# encoder_out, state_h, state_c = encoder_lstm(encoder_emb(encoder_inputs))

#encoder_lstm = Bidirectional( tf.compat.v1.keras.layers.LSTM(units=units, return_sequences=True, return_state=True))
#encoder_out, fstate_h, fstate_c, bstate_h, bstate_c = encoder_lstm(encoder_emb(encoder_inputs))
#state_h = Concatenate()([fstate_h,bstate_h])
#state_c = Concatenate()([bstate_h,bstate_c])


# encoder_states = [state_h, state_c]


# # Now create the Decoder layers.
# decoder_inputs = Input(shape=(None,))
# decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
# # decoder_lstm = LSTM(units=units, return_sequences=True, return_state=True)
# decoder_lstm = tf.compat.v1.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
# decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)
# # Two dense layers added to this model to improve inference capabilities.
# decoder_d1 = Dense(units, activation="relu")
# decoder_d2 = Dense(vocab_out_size, activation="softmax")
# decoder_out = decoder_d2(Dropout(rate=.2)(decoder_d1(Dropout(rate=.2)(decoder_lstm_out))))


# Finally, create a training model which combines the encoder and the decoder.
# Note that this model has three inputs:

#TRAIN
# model = Model(inputs = [encoder_inputs, decoder_inputs], outputs= decoder_out)

# We'll use sparse_categorical_crossentropy so we don't have to expand decoder_out into a massive one-hot array.
# Adam is used because it's, well, the best.

# model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])


#creating checkpoints

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#outputFolder = './checkpoints'
#if not os.path.exists(outputFolder):
#    os.makedirs(outputFolder)
#filepath=outputFolder+"/model-{epoch:02d}-{accuracy:.2f}.hdf5"

#checkpoint_callback = ModelCheckpoint(
#    filepath, monitor='accuracy', verbose=1,
#    save_best_only=False, save_weights_only=False,
#    save_frequency=1)

# Note, we use 20% of our data for validation.
# epochs = 20
# history = model.fit([input_data, teacher_data], target_data,
#                     callbacks=[tensorboard_callback],
#                  batch_size= BATCH_SIZE,
#                  epochs=epochs,
#                  validation_split=0.2)



# previous_epoch = epochs



# # Create the encoder model from the tensors we previously declared.
# encoder_model = Model(encoder_inputs, [encoder_out, state_h, state_c])

# # Generate a new set of tensors for our new inference decoder. Note that we are using new tensors,
# # this does not preclude using the same underlying layers that we trained on. (e.g. weights/biases).

# inf_decoder_inputs = Input(shape=(None,), name="inf_decoder_inputs")
# # We'll need to force feed the two state variables into the decoder each step.
# state_input_h = Input(shape=(units,), name="state_input_h")
# state_input_c = Input(shape=(units,), name="state_input_c")
# decoder_res, decoder_h, decoder_c = decoder_lstm(
#     decoder_emb(inf_decoder_inputs),
#     initial_state=[state_input_h, state_input_c])
# inf_decoder_out = decoder_d2(decoder_d1(decoder_res))
# inf_model = Model(inputs=[inf_decoder_inputs, state_input_h, state_input_c],
#                   outputs=[inf_decoder_out, decoder_h, decoder_c])

# inf_model.save("inferecnce_model_gpu_4.h5")
# encoder_model.save("encoder_model_gpu_4.h5")


# ###OUTPUT

inf_model = tf.keras.models.load_model("models/inferecnce_model_gpu2.h5", compile=False)
encoder_model = tf.keras.models.load_model("models/encoder_model_gpu2.h5", compile=False)


def sentence_to_vector(sentence, lang):
    pre = sentence
    vec = np.zeros(len_input)
    sentence_list = [lang.word2idx[s] for s in pre.split(' ')]
    for i, w in enumerate(sentence_list):
        vec[i] = w
    return vec


# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence, infenc_model, infmodel):
    sv = sentence_to_vector(input_sentence, input_lang)
    sv = sv.reshape(1, len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)

    i = 0
    start_vec = target_lang.word2idx["<start>"]
    stop_vec = target_lang.word2idx["<end>"]

    cur_vec = np.zeros((1, 1))
    cur_vec[0, 0] = start_vec
    cur_word = "<start>"
    output_sentence = ""

    while cur_word != "<end>" and i < (len_target - 1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        x_in = [cur_vec, sh, sc]
        [nvec, sh, sc] = infmodel.predict(x=x_in)
        cur_vec[0, 0] = np.argmax(nvec[0, 0])
        cur_word = target_lang.idx2word[np.argmax(nvec[0, 0])]
    return output_sentence


test = [
    'Thank ',
    'You are the ',
    'I wish ',
    'All of your ',
    'tha',
    't',
    'th',
    'than',
    'thank',
    'thanks ',
    'thanks fo',
    'thanks for ',
    'thanks for a',
    'You are app'

]

import pandas as pd

output = []
for t in test:
    output.append({"Input seq": t.lower(), "Pred. Seq": translate(t.lower(), encoder_model, inf_model)})
results_df = pd.DataFrame.from_dict(output)
results_df.head(len(test))
print(results_df)
