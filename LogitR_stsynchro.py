from __future__ import division

import tensorflow as tf
import sys
import numpy as np
import time

from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz

# cluster specificationl
parameter_servers = ["10.24.1.218:2222","10.24.1.214:2223"]
workers = ["10.24.1.215:2224", "10.24.1.217:2225"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

y_train = load_npz("y_train.npz").toarray()
y_test = np.load("y_test.npy")
x_train = load_npz("x_train.npz").toarray()
x_test = load_npz("x_test.npz").toarray()

print("data_loaded")

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

# config
batch_size = 2048
learning_rate = 0.003
n_class = y_train.shape[1]
n_epochs = 30
Reg = 0.005


print("training_started")
if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
		# tf.reset_default_graph()
		#Placeholder
		x = tf.placeholder(tf.float32,[None,x_train.shape[1]])
		y = tf.placeholder(tf.float32,[None,n_class])

		#Variables
		W = tf.get_variable(name = 'Weights',initializer = tf.random_normal([x_train.shape[1],n_class]))
		B = tf.get_variable(name = 'Bias',initializer = tf.random_normal([n_class]))
		global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0),trainable = False, dtype=tf.int32)

		#Computation Ops
		logit = tf.matmul(x,W) + B
		sample_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y)
		loss = tf.reduce_mean(sample_loss)
		regularizer = tf.nn.l2_loss(W)
		total_loss = tf.reduce_mean(loss + Reg*regularizer)
		grad_op = tf.train.AdamOptimizer(learning_rate)
		rep_op = tf.contrib.opt.DropStaleGradientOptimizer(grad_op,staleness=20,use_locking=True)
	    optimizer = rep_op.minimize(total_loss, global_step=global_step)
		pred_class = tf.argmax(logit,axis = 1)
		init = tf.global_variables_initializer()
		
	sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init)

	epoch_loss_w1 = []
	epoch_loss_w2 = []
    test_acc_w1 = []
    test_acc_w2 = []		
	with sv.prepare_or_wait_for_session(server.target) as sess:

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
	        if FLAGS.task_index == 0:
	        	epoch_loss_w1.append(total_epoch_loss/int(x_train.shape[0]/batch_size))
	        else:
	        	epoch_loss_w2.append(total_epoch_loss/int(x_train.shape[0]/batch_size))
	        print("After epoch: ",epoch+1)
	        print("training_loss: ",total_epoch_loss/int(x_train.shape[0]/batch_size), ", training acc: ",total_acc/int(x_train.shape[0]/batch_size))
	        
	        pred_ = sess.run([pred_class],feed_dict = {x:x_test})
	        total_acc = get_acc(pred_,y_test)
	        if FLAGS.task_index == 0:
	        	test_acc_w1.append(total_acc/x_test.shape[0])
	        else:
	        	test_acc_w2.append(total_acc/x_test.shape[0])
	        print("test_acc: ",total_acc/x_test.shape[0])


		np.save("epoch_loss_stsynchro_w1",epoch_loss_w1)
		np.save("epoch_loss_stynchro_w2",epoch_loss_w2)
		np.save("test_acc_stynchro_w1",test_acc_w1)
		np.save("test_acc_stynchro_w2",test_acc_w2)

	sv.stop()




