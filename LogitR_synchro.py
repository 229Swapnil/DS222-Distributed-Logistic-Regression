from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
import h5py
import time

# cluster specificationl
parameter_servers = ["10.24.1.201:2227","10.24.1.202:2227"]
workers = [ "10.24.1.203:2227", "10.24.1.204:2227"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Depends on no. of workers starting from 0")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

y_train = np.load("y_train.npy").astype(np.float32)
y_test = np.load("y_test.npy")
h5f1 = h5py.File('x_train.h5','r')
x_train = h5f1['x_train'][:]
h5f2 = h5py.File('x_test.h5','r')
x_test = h5f2['x_test'][:]
h5f1.close()
h5f2.close()

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
n_epochs = 20
Reg = 0.005
# logs_path = "/home/swapnilgupta.229/Assignment2/"

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
		# tf.reset_default_graph()
		#Placeholder
		x = tf.placeholder(tf.float32,[None,x_train.shape[1]])
		y = tf.placeholder(tf.float32,[None,n_class])

		#Variables
		tf.set_random_seed(1)
		W = tf.get_variable(name = 'Weights',initializer = tf.random_normal([x_train.shape[1],n_class]))
		B = tf.get_variable(name = 'Bias',initializer = tf.random_normal([n_class]))
		global_step = tf.Variable(0, trainable=False)

		#Computation Ops
		logit = tf.matmul(x,W) + B
		sample_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y)
		loss = tf.reduce_mean(sample_loss)
		regularizer = tf.nn.l2_loss(W)
		total_loss = tf.reduce_mean(loss + Reg*regularizer)
		grad_op = tf.train.AdamOptimizer(learning_rate)
		rep_op = tf.train.SyncReplicasOptimizer(grad_op, replicas_to_aggregate=len(workers), total_num_replicas=len(workers), use_locking=True)
	    optimizer = rep_op.minimize(total_loss, global_step=global_step)  
	    init_token_op = rep_op.get_init_tokens_op()
	    chief_queue_runner = rep_op.get_chief_queue_runner()
		pred_class = tf.argmax(logit,axis = 1)
		init = tf.global_variables_initializer()

		
	sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init)
	
	epoch_loss_w1 = []
	epoch_loss_w2 = []
    test_acc_w1 = []
    test_acc_w2 = []
	with sv.prepare_or_wait_for_session(server.target) as sess:

		if FLAGS.task_index == 0:
			sv.start_queue_runners(sess, [chief_queue_runner])
			sess.run(init_token_op)
			
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


	np.save("epoch_loss_synchro_w1",epoch_loss_w1)
	np.save("epoch_loss_synchro_w2",epoch_loss_w2)
	np.save("test_acc_synchro_w1",test_acc_w1)
	np.save("test_acc_synchro_w2",test_acc_w2)




