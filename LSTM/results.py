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

import enchant
d = enchant.Dict("en_US")

import re



config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

f = open("LSTM_model/data_text.txt","r")
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

print(30*'*')
print('models loaded')
print(30*'*')

def sentence_to_vector(sentence, lang, len_input):
    pre = sentence
    vec = np.zeros(len_input)
    sentence_list = [lang.word2idx[s] for s in pre.split(' ')]
    for i, w in enumerate(sentence_list):
        vec[i] = w
    return vec


# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence, infenc_model, infmodel, input_lang, target_lang, len_input, len_target):
    sv = sentence_to_vector(input_sentence, input_lang, len_input)
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





# test_string = 'thank you f'



def get_sentences(test_string, input_data, teacher_data, input_lang, target_lang, len_input, len_target, df, encoder_model, inf_model):

	prev_text = test_string
	# print(test_string.split(' ')[-1])
	output_test = []
	# print(test_string)
	flag = 1

	# if(d.check(test_string.split(' ')[-1]) == False):
	#   flag = 0
	#   print('not present')


	# try:
	# 	print(test_string.split(' ')[-1])
	# 	translate(test_string.split(' ')[-1], encoder_model, inf_model)

	# except KeyError:
	# 	print('key error boi')
	# 	flag = 0
	# print(test_string)
	# print(type(test_string))
	output_test.append({"Input": test_string.lower().rstrip(), "Pred": translate(test_string.lower(), encoder_model, inf_model, input_lang, target_lang, len_input, len_target)})
	# print(output_test[0].get("Pred", ""))
	# print('output_test : ', output_test)
	i = 0
	sentences = []
	all_sentences = []
	for word in output_test[0].get("Pred", "").split():
	    # print(11)
	    # print(word)
		

		# if(flag != 0):
	    prev_text = test_string + " " + word + " "
	    # else:
	    # prev_text = test_string + word + " "

	    test_string = prev_text + " "
	    # print('word', word)
	    # print('prev_text', prev_text)
	    i = i + 1 
	    # print(i)
	    if(i == 10):
	    	break
	    while(1):
	        # print(prev_text)
	        new_op = [] 
	        # print(prev_text)

	        prev_text = re.sub(' +', ' ', prev_text)

	        new_op.append({"Input": prev_text.lower().rstrip(), "Pred": translate(prev_text.lower(), encoder_model, inf_model, input_lang, target_lang, len_input, len_target)})
	        # print(new_op)
	        # result = pd.DataFrame.from_dict(new_op)
	        # result.head(len(prev_text))
	        # print(prev_text)

	        # prev_text = prev_text + next_word
	        # if(len(prev_text.split()) > 7):
	        # print(new_op)
	        
	        all_sentences.append(new_op)
	        prev_text = re.sub(' +', ' ', prev_text)
	        next_word = new_op[0].get("Pred", "").split()[0] + " "
	        next_word = re.sub(' +', ' ', next_word)
	        prev_text = prev_text + next_word
	        if('.' in prev_text):
	            # print(result)
	            sentences.append(prev_text)
	            break

	        # print(prev_text)
	        
	    # break
	    # print(result)

	# # sentences = set(sentences)

	for l_outer in all_sentences:
	    for l_inner in l_outer:
	        # print(l_inner.values())
	        p = l_inner.get("Input", "")
	        v = l_inner.get("Pred", "")
	        # print(p)
	        # print(v)
	        sentence = p + v
	        sentence = re.sub(' +', ' ', sentence)
	        sentences.append(sentence)

	        # sum(list(l_inner.values()), [])    
	# result = pd.DataFrame.from_dict(output_test)
	# result.head(len(test_string))


	# print(len(set(sentences)))
	sentences = list(set(sentences))
	print(sentences)
	return sentences


