from __future__ import division
import tensorflow as tf
import numpy as np
import h5py
import argparse
import time


y_train = load_npz("y_train.npz").toarray()
y_test = np.load("y_test.npy")
x_train = load_npz("x_train.npz").toarray()
x_test = load_npz("x_test.npz").toarray()

def get_acc(pred,true):
    acc = 0
    for i in range(pred[0].shape[0]):
        if pred[0][i] in true[i]: acc+=1
    return acc

def get_train_acc(pred,true):
    acc = 0
    for i in range(pred.shape[0]):
        if true[i,pred[i]]>0: acc+=1
    return acc


parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='lr', default='constant')
args = parser.parse_args()

n_class = y_train.shape[1]
batch_size = 2048
n_epochs = 30
Reg = 0.001

if args.lr == 'constant':
    starter_learning_rate = 0.005
    decay = 1
elif args.lr == 'decay':
    starter_learning_rate = 0.01
    decay = 0.95
else:
    starter_learning_rate = 0.001
    decay = 1.05



## Tensorflow Graph

tf.reset_default_graph()
#Placeholder
x = tf.placeholder(tf.float32,[None,x_train.shape[1]])
y = tf.placeholder(tf.float32,[None,n_class])

#Variables
W = tf.get_variable(name = 'Weights',initializer = tf.random_normal([x_train.shape[1],n_class]))
B = tf.get_variable(name = 'Bias',initializer = tf.random_normal([n_class]))
global_step = tf.Variable(0, trainable=False)

#Computation Ops
logit = tf.matmul(x,W) + B
sample_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y)
loss = tf.reduce_mean(sample_loss)
regularizer = tf.nn.l2_loss(W)
total_loss = loss + Reg*regularizer
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,int(x_train.shape[0]/batch_size), decay, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=global_step)
pred_class = tf.argmax(logit,axis = 1)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    epoch_loss = []
    test_acc = []
    begin = time.time()
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        total_acc = 0
        permute = np.random.permutation(x_train.shape[0])
        x_train = x_train[permute]
        y_train = y_train[permute]
        for batch in range(int(x_train.shape[0]/batch_size)):
            inp = x_train[batch*batch_size:(batch+1)*batch_size,:]
            targ = y_train[batch*batch_size:(batch+1)*batch_size,:]
            _,iter_loss, train_pred = sess.run([optimizer,loss,pred_class],feed_dict = {x:inp,y:targ})
            total_acc += (get_train_acc(train_pred,targ))/batch_size
            total_epoch_loss += iter_loss
        epoch_loss.append(total_epoch_loss/int(x_train.shape[0]/batch_size))
        print("Epoch: ",epoch+1,"/",n_epochs,", epoch_time: ",begin - time.time())
        print("training_loss: ",total_epoch_loss/int(x_train.shape[0]/batch_size), ", training acc: ",total_acc/int(x_train.shape[0]/batch_size))
        
        pred_ = sess.run([pred_class],feed_dict = {x:x_test})
        total_acc = get_acc(pred_,y_test)
        test_acc.append(total_acc/x_test.shape[0])
        print("test_acc: ",total_acc/x_test.shape[0])

np.save("epoch_loss "+args.lr,epoch_loss)
np.save("test_acc "+args.lr,test_acc)