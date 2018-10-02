from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import metrics as SKM
from sklearn import preprocessing
import gensim.models.keyedvectors as word2vec
import numpy as np
np.set_printoptions(threshold=np.nan)
from keras.optimizers import Adam,RMSprop

from keras.preprocessing import sequence,text
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import nltk
from nltk.stem import SnowballStemmer,WordNetLemmatizer
#nltk.download('wordnet')
lemma=WordNetLemmatizer()
from string import punctuation
from sklearn.decomposition import PCA as sklearnPCA
import re
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.initializers import glorot_uniform
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Bidirectional,Embedding
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score,roc_auc_score
from keras.callbacks import Callback


def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

num_features=100
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')


model = word2vec.KeyedVectors.load("word2vec.model")
print "Creating average feature vecs for training questions"
train['question']=clean_review(train.question.values)
test['question']=clean_review(test.question.values)


train_text=train.question.values
test_text=test.question.values

all_words=' '.join(train_text)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
print num_unique_word

r_len = []
for text in train_text:
    word = word_tokenize(text)
    l = len(word)
    r_len.append(l)

MAX_REVIEW_LEN = np.max(r_len)
print MAX_REVIEW_LEN

max_features = num_unique_word
max_words = 50#MAX_REVIEW_LEN

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_text))
X_train = tokenizer.texts_to_sequences(train_text)
X_test = tokenizer.texts_to_sequences(test_text)

trainDataVecs = sequence.pad_sequences(X_train, maxlen=max_words)
testDataVecs = sequence.pad_sequences(X_test, maxlen=max_words)
# norm_trainDataVecs=preprocessing.normalize([trainDataVecs])
# print norm_trainDataVecs
################################------------MODEL----------------#########################################


word_index = tokenizer.word_index
words=min(max_features, len(word_index))
embedding_matrix = np.zeros((words, 300))
for word, i in word_index.items():
    if i >= max_features:
        continue
    if word in model.vocab:
        embedding_vector = model.word_vec(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

sklearn_pca = sklearnPCA(n_components=100)
embedding_matrix = sklearn_pca.fit_transform(embedding_matrix)
# print embedding_matrix

max_sequence_size = max_features
classes_num = 1
my_model=Sequential()

my_model.add(Embedding(max_sequence_size,100,input_length=trainDataVecs.shape[1],weights=[embedding_matrix],trainable=True))
my_model.add(Bidirectional(LSTM(100,activation='tanh',init='glorot_uniform',recurrent_dropout = 0.2, dropout = 0.2)))

my_model.add(Dense(50,activation='sigmoid'))
my_model.add(Dropout(0.2))

#
my_model.add(Dense(classes_num, activation='softmax'))
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

print "Compiling..."
optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
my_model.compile(optimizer=optimizer,
               loss='binary_crossentropy',
                  metrics=['accuracy'])

print my_model.summary()
# # #################################---------TRAINING--------------------########################################

le=preprocessing.LabelEncoder()
le.fit(train['question_sentiment_gold'])
Y_train=le.transform(train['question_sentiment_gold'])
print Y_train

history=my_model.fit(trainDataVecs, Y_train, shuffle = True, batch_size = 50, epochs=20)#, validation_split=0.1, verbose=1)
#
my_model.save('word2vec_lstm.h5')




#########################-----------------TESTING-----------------_###############
# my_model=load_model('word2vec_lstm.h5')

weights = my_model.weights # weight tensors
#weights=[weight for weight in weights if my_model.get_layer(weight.name[:-2]).trainable]
gradients = my_model.optimizer.get_gradients(my_model.total_loss, weights) # gradient tensors

input_tensors = [my_model.inputs[0], # input data
                 my_model.sample_weights[0], # sample weights
                 my_model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)

inputs = [[trainDataVecs], # X input data
          [1], # sample weights
          [Y_train], # y labels
          0.005 # learning phase in TEST mode
]

print [a for a in zip(weights, get_gradients(inputs))]
###############################___________________PREDICTING__________###################################
# temp_predicted = my_model.predict(testDataVecs)
#
# le=preprocessing.LabelEncoder()
# le.fit(test['question_sentiment_gold'])
# Y_test=le.transform(test['question_sentiment_gold'])
#
# f0=SKM.f1_score(Y_test,temp_predicted,pos_label=0)
#
# print f0
