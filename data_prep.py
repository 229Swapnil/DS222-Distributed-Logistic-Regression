from __future__ import division
import numpy as np
import re
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz

train_path='./DBPedia.full/full_train.txt'
test_path='./DBPedia.full/full_test.txt'

file = open(train_path,'r')
file = file.readlines()
x_text_train = []
y_text_train = []
vocab = defaultdict(int)
for doc in file:
    doc = doc.lower().strip().split()
    labels = doc[0].split(",")
    doc = [re.sub(r'[^\w\s]','',word) for word in doc[3:] if re.sub(r'[^\w\s]','',word).isalpha()]
    for word in doc:
        vocab[word]+=1
    x_text_train.append(doc)
    y_text_train.append(labels)
        
        
file = open(test_path,'r')
file = file.readlines()
x_text_test = []
y_text_test = []
for doc in file:
    doc = doc.lower().strip().split()
    labels = doc[0].split(",")
    doc = [re.sub(r'[^\w\s]','',word) for word in doc[3:] if re.sub(r'[^\w\s]','',word).isalpha()]
    x_text_test.append(doc)
    y_text_test.append(labels)

for word,count in vocab.items():
    if count<75 or count>20000:
        del vocab[word]
        

index = 0
for word in vocab:
    vocab[word] = index
    index+=1


def get_data_matrix(text):
    x = np.zeros((len(text),len(list(vocab.keys()))))
    for i,sample in enumerate(text):
        for word in sample:
            if word in vocab: x[i,vocab[word]] = 1
    return x


x_train = get_data_matrix(x_text_train)
x_test = get_data_matrix(x_text_test)

del x_text_train
del x_text_test

label_list = list(set([y for labels in y_text_train for y in labels]))


y_train = np.zeros((len(y_text_train),len(label_list)))
for i,y in enumerate(y_text_train):
    den = len(y)
    for label in y:
        y_train[i,label_list.index(label)] = 1/den
    
y_test = [[label_list.index(label) for label in y] for y in y_text_test]

del y_text_train
del y_text_test

save_npz('x_train',coo_matrix(x_train))
save_npz('x_test',coo_matrix(x_test))
save_npz('y_train',coo_matrix(y_train))
np.save("y_test",y_test)

