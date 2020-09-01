# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:25:00 2020

@author: riyuecao
"""


import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import time
class BigModel:
    def __init__(self,args,model_type):
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.num_input = 784
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = 'big_model'
        self.temperature = args.temperature
        self.checkpoint_path = os.path.join(self.checkpoint_dir,self.checkpoint_file+'.ckpt')
        self.log_dir = os.path.join(args.log_dir,self.checkpoint_file)
        self.model_type = model_type
        self.epochs = args.epochs

        
        self.weights = {
            'wc1':tf.Variable(tf.random_normal([5,5,1,32]),name="%s_%s"%(self.model_type,'wc1')),
            'wc2':tf.Variable(tf.random_normal([5,5,32,64]),name="%s_%s"%(self.model_type,'wc2')),
            'wd1':tf.Variable(tf.random_normal([7*7*64,1024]),name="%s_%s"%(self.model_type,'wd1')),
            'out':tf.Variable(tf.random_normal([1024,self.num_classes]),name="%s_%s"%(self.model_type,'out'))} 
        
        self.biases = {
            'bc1':tf.Variable(tf.random_normal([32]),name="%s_%s"%(self.model_type,'bc1')),
            'bc2':tf.Variable(tf.random_normal([64]),name="%s_%s"%(self.model_type,'bc2')),
            'bd1':tf.Variable(tf.random_normal([1024]),name="%s_%s"%(self.model_type,'bd1')),
            'out':tf.Variable(tf.random_normal([self.num_classes]),name="%s_%s"%(self.model_type,'out'))} 
        
        self.build_model()
        self.saver = tf.train.Saver()
        
    def conv2d(self,x,w,b,strides=1):
        with tf.name_scope("%sconv2d"%(self.model_type)),tf.variable_scope("%smaxpool2d"%(self.model_type)):
            x = tf.nn.conv2d(x,w, strides=[1,strides,strides,1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)
    
    def maxpool2d(self,x,k=2):
        with tf.name_scope("%smaxpool2d"%(self.model_type)),tf.variable_scope("%smaxpool2d"%(self.model_type)):
            return tf.nn.max_pool(x,[1,k,k,1],[1,k,k,1], padding='SAME')
    
    def build_model(self):
        self.X = tf.placeholder(tf.float32,[None,self.num_input],name="%s_%s"%(self.model_type,'x_input'))
        self.Y = tf.placeholder(tf.float32,[None,self.num_classes],name="%s_%s"%(self.model_type,'y_input'))
        self.keep_prob = tf.placeholder(tf.float32,name="%s_%s"%(self.model_type,'keep_prob'))
        self.softmax_temperature = tf.placeholder(tf.float32,name="%s_%s"%(self.model_type,'temperature'))
        with tf.name_scope("%sinput_reshape"%(self.model_type)),tf.variable_scope("%sinput_reshape"%(self.model_type)):
            x = tf.reshape(self.X, [-1,28,28,1])
        
        with tf.name_scope("%sconv-maxpool"%(self.model_type)),tf.variable_scope("%sconv-maxpool"%(self.model_type)):
            conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
            conv1 = self.maxpool2d(conv1)
            
            conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
            conv2 = self.maxpool2d(conv2)
            
        with tf.name_scope("%sfully_connection"%(self.model_type)),tf.variable_scope("%sfully_connection"%(self.model_type)):
            fc1 = tf.reshape(conv2,[-1,self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']),self.biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1,self.keep_prob)
            
            logits = tf.add(tf.matmul(fc1, self.weights['out']),self.biases['out']) / self.softmax_temperature
        
        with tf.name_scope("%sprediction"%(self.model_type)),tf.variable_scope("%sprediction"%(self.model_type)):
            self.prediction = tf.nn.softmax(logits)
            
            self.accuarcy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.Y,1)),tf.float32))
        with tf.name_scope("%soptimization"%(self.model_type)),tf.variable_scope("%soptimization"%(self.model_type)):
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)
            
        with tf.name_scope("%ssummaries"%(self.model_type)),tf.variable_scope("%ssummaries"%(self.model_type)):
            tf.summary.scalar('loss', self.loss_op)
            tf.summary.scalar('accuarcy', self.accuarcy)
            
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        
        self.merged_summary = tf.summary.merge_all(scope=self.model_type)
        
        
    def start_session(self):
        self.sess = tf.Session()
    
    def close_session(self):
        self.sess.close()

    def evaluate(self,data_x,data_y,temperature,state='train'):
        if state=='train':
            pred = self.sess.run(self.prediction,feed_dict={self.X:data_x,self.keep_prob: self.dropoutprob, self.softmax_temperature: temperature})
            with tf.Session() as S:
                b_pred = tf.argmax(pred,1).eval()
                b_data_y = tf.argmax(data_y ,1).eval()
            precision = precision_score(b_data_y,b_pred,average='macro')
            recall = recall_score(b_data_y, b_pred,average='micro')
            f1 = f1_score(b_data_y, b_pred,average='micro')

        else:
            pred = self.sess.run(self.prediction,feed_dict={self.X:data_x,self.keep_prob: 1.0, self.softmax_temperature: temperature})
            with tf.Session() as S:
                b_pred = tf.argmax(pred, 1).eval()
                b_data_y = tf.argmax(data_y, 1).eval()
            precision = precision_score(b_data_y, b_pred, average='macro')
            recall = recall_score(b_data_y, b_pred, average='macro')
            f1 = f1_score(b_data_y, b_pred, average='macro')
        if state == 'test':
            print(confusion_matrix(b_data_y,b_pred))
        return precision,recall,f1,roc_auc_score(data_y,pred,average='macro')


    def train(self,dataset):
        self.sess.run(tf.global_variables_initializer())
        print('Starting Training')
        self.global_step = tf.Variable(0, trainable=False)
        train_data = dataset.get_train_data()
        num_steps = len(train_data.images)//self.batch_size
        validation_x, validation_y = dataset.get_validation_data()
        train_summary = tf.summary.FileWriter(self.log_dir,graph=self.sess.graph)
        max_accuarcy = 0

        for epoch in range(self.epochs):
            for step in tqdm(range(num_steps)):
                batch_x,batch_y = train_data.next_batch(self.batch_size,shuffle=False)

                _,summary = self.sess.run([self.train_op,self.merged_summary],
                                         feed_dict={self.X:batch_x,self.Y:batch_y,self.keep_prob:self.dropoutprob,
                                                    self.softmax_temperature:self.temperature})
            train_loss, train_acc = self.sess.run([self.loss_op, self.accuarcy],
                                      feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.0,
                                                 self.softmax_temperature: 1.0})
            precision,recall,f1,auc = self.evaluate(batch_x, batch_y, 1.0, state='train')

            val_loss, val_acc = self.sess.run([self.loss_op, self.accuarcy],
                                      feed_dict={self.X: validation_x, self.Y: validation_y, self.keep_prob: 1.0,
                                                 self.softmax_temperature: 1.0})
            val_precision,val_recall,val_f1,val_auc = self.evaluate(validation_x, validation_y, 1.0, state='val')

            print('Epoch'+str(epoch+1)+':Train_loss='+'{:.4f}'.format(train_loss)+',Train_acc='+'{:.4f}'.format(train_acc)+',Train_auc='+'{:.4f}'.format(auc)+
                  ',val_loss='+'{:.4f}'.format(val_loss)+',val_accuarcy='+'{:.4f}'.format(val_acc)+',val_auc='+'{:.4f}'.format(val_auc))
            if val_acc > max_accuarcy:
                save_path = self.saver.save(self.sess,self.checkpoint_path)
                max_accuarcy = val_acc
                print('Model_Checkpointed to %s'%save_path)


        train_summary.close()
        print('Traning Done!')


    def save_as_npy(self,dataset,batchsize):
        y_soft_target = []
        train_data = dataset.get_train_data()
        for step in range(len(train_data.images)//batchsize):
            batch_x, batch_y = train_data.next_batch(self.batch_size,shuffle=False)
            out = self.sess.run([self.prediction],
                                       feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.0,
                                                  self.softmax_temperature: 1.0})
            y_soft_target.append(out)
        soft_targets = np.c_[y_soft_target].reshape(len(train_data.images), 10)
        print(soft_targets.shape)
        np.save('teacher.npy',soft_targets)


    def predict(self,data_x,temperature):
        return self.sess.run(self.prediction,feed_dict={self.X:data_x,self.keep_prob: 1.0, self.softmax_temperature: temperature})
    
    def run_inference(self,dataset):
        test_images,test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuarcy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           self.keep_prob: 1.0,
                                                                           self.softmax_temperature: 1.0
                                                                           }))
        start = time.time()
        pred = self.predict(test_images,temperature=1.0)
        print('Time consumes: ' + str(time.time() - start) + ' seconds')
        test_precision,test_recall,test_f1,test_auc = self.evaluate(test_images, test_labels, 1.0, state='test')
        print('Testing Precsion:'+str(test_precision)+'\n'+'Testing Recall:'+str(test_recall)+'\n'+'Testing F1:'+str(test_f1)+'Testing AUC:'+str(test_auc))





    def load_model_from_file(self,path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
            
class SmallModel:
    def __init__(self,args,model_type):
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.hidden_number1 = 256
        self.hidden_number2 = 256
        self.num_input = 784
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = 'small_model'
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.temperature = args.temperature
        self.max_checkpoint_path = os.path.join(self.checkpoint_dir,self.checkpoint_file+'max')
        self.log_dir = os.path.join(args.log_dir,self.checkpoint_file)
        self.model_type = model_type
        self.soft_target = np.load('teacher.npy')


        
        self.weights = {
            'h1':tf.Variable(tf.random_normal([self.num_input,self.hidden_number1]),name="%s_%s"%(self.model_type,'h1')),
            'h2':tf.Variable(tf.random_normal([self.hidden_number1,self.hidden_number2]),name="%s_%s"%(self.model_type,'h2')),
            'out':tf.Variable(tf.random_normal([self.hidden_number2,self.num_classes]),name="%s_%s"%(self.model_type,'out')),
            'linear':tf.Variable(tf.random_normal([self.num_input,self.num_classes]),name="%s_%s"%(self.model_type,'linear'))} 
        
        self.biases = {
            'b1':tf.Variable(tf.random_normal([self.hidden_number1]),name="%s_%s"%(self.model_type,'bc1')),
            'b2':tf.Variable(tf.random_normal([self.hidden_number2]),name="%s_%s"%(self.model_type,'bc2')),
            'out':tf.Variable(tf.random_normal([self.num_classes]),name="%s_%s"%(self.model_type,'bd1')),
            'linear':tf.Variable(tf.random_normal([self.num_classes]),name="%s_%s"%(self.model_type,'out'))} 
        
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self):
        self.X = tf.placeholder(tf.float32,[None,self.num_input],name="%s_%s"%(self.model_type,'x_input'))
        self.Y = tf.placeholder(tf.float32,[None,self.num_classes],name="%s_%s"%(self.model_type,'y_input'))
        self.keep_prob = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, 'keep_prob'))
        self.flag = tf.placeholder(tf.bool,None,name="%s_%s"%(self.model_type,'flag'))
        self.soft_Y = tf.placeholder(tf.float32,[None,self.num_classes],name="%s_%s"%(self.model_type,'soft_y'))
        self.softmax_temperature = tf.placeholder(tf.float32,name="%s_%s"%(self.model_type,'softtemperature'))
        with tf.name_scope("%sfclayer"%(self.model_type)),tf.variable_scope("%sfclayer"%(self.model_type)):
            fc1 = tf.add(tf.matmul(self.X, self.weights['h1']),self.biases['b1'])
            fc2 = tf.add(tf.matmul(fc1, self.weights['h2']),self.biases['b2'])
            logits = tf.add(tf.matmul(fc2, self.weights['out']),self.biases['out'])
        
        with tf.name_scope("%sprediction"%(self.model_type)),tf.variable_scope("%sprediction"%(self.model_type)):
            self.prediction = tf.nn.softmax(logits)
            
            self.accuarcy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.Y,1)),tf.float32))
       
        with tf.name_scope("%soptimization"%(self.model_type)),tf.variable_scope("%soptimization"%(self.model_type)):
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
            self.total_loss = self.loss_op
            self.loss_soft_op = tf.cond(self.flag,true_fn=lambda:tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits/self.softmax_temperature, labels=self.soft_Y)),
                                        false_fn=lambda:0.0)
            #self.total_loss += tf.square(self.softmax_temperature) * self.loss_soft_op
            self.total_loss = 0.5*self.total_loss+0.5* self.loss_soft_op
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)
            
        with tf.name_scope("%ssummaries"%(self.model_type)),tf.variable_scope("%ssummaries"%(self.model_type)):
            tf.summary.scalar('loss_op', self.loss_op)
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar('accuarcy', self.accuarcy)
            
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        
        self.merged_summary = tf.summary.merge_all(scope=self.model_type)
        
        
    def start_session(self):
        self.sess = tf.Session()
    
    def close_session(self):
        self.sess.close()

    def evaluate(self,data_x,data_y,temperature,state='train'):
        if state=='train':
            pred = self.sess.run(self.prediction,feed_dict={self.X:data_x,self.keep_prob: self.dropoutprob, self.softmax_temperature: temperature})
            with tf.Session() as S:
                b_pred = tf.argmax(pred,1).eval()
                b_data_y = tf.argmax(data_y ,1).eval()
            precision = precision_score(b_data_y,b_pred,average='macro')
            recall = recall_score(b_data_y, b_pred,average='macro')
            f1 = f1_score(b_data_y, b_pred,average='macro')

        else:
            pred = self.sess.run(self.prediction,feed_dict={self.X:data_x,self.keep_prob: 1.0, self.softmax_temperature: temperature})
            with tf.Session() as S:
                b_pred = tf.argmax(pred, 1).eval()
                b_data_y = tf.argmax(data_y, 1).eval()
            precision = precision_score(b_data_y, b_pred, average='macro')
            recall = recall_score(b_data_y, b_pred, average='macro')
            f1 = f1_score(b_data_y, b_pred, average='macro')
        if state == 'test':
            print(confusion_matrix(b_data_y,b_pred))
        return precision,recall,f1,roc_auc_score(data_y,pred,average='micro')

    def train(self,dataset,teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True
        self.sess.run(tf.global_variables_initializer())
        
        train_data = dataset.get_train_data()
        validation_x,validation_y = dataset.get_validation_data()
        num_steps = len(train_data.images)//self.batch_size
        train_summary = tf.summary.FileWriter(self.log_dir,graph=self.sess.graph)
        max_accuarcy = 0
        print('Starting Training')
        for epoch in tqdm(range(self.epochs)):
            for i,step in enumerate(range(num_steps)):
                batch_x,batch_y = train_data.next_batch(self.batch_size,shuffle=False)
                soft_targets = batch_y
                if teacher_flag:
                    soft_targets = self.soft_target[i*self.batch_size:i*self.batch_size+self.batch_size,:]
                    #if i == 0:
                        #print(np.argmax(batch_y,1),np.argmax(soft_targets,1))
                _,summary = self.sess.run([self.train_op,self.merged_summary],
                                         feed_dict={self.X:batch_x,self.Y:batch_y,self.flag: teacher_flag,
                                                    self.soft_Y: soft_targets,
                                                    self.softmax_temperature:self.temperature})
            train_loss,train_acc = self.sess.run([self.loss_op,self.accuarcy],
                                         feed_dict={self.X:batch_x,self.Y:batch_y,self.flag: teacher_flag,self.soft_Y: soft_targets,
                                                    self.softmax_temperature:self.temperature})
            precision,recall,f1,auc = self.evaluate(batch_x, batch_y, 1.0, state='train')
            val_loss, val_acc = self.sess.run([self.loss_op, self.accuarcy],
                                              feed_dict={self.X: validation_x, self.Y: validation_y,
                                                         self.keep_prob: 1.0,
                                                         self.softmax_temperature: 1.0})
            val_precision,val_recall,val_f1,val_auc = self.evaluate(batch_x, batch_y, 1.0, state='val')
            print('Epoch'+str(epoch+1)+':Train_loss='+'{:.4f}'.format(train_loss)+',Train_acc='+'{:.4f}'.format(train_acc)+',Train_auc='+'{:.4f}'.format(auc)+
                  ',val_loss='+'{:.4f}'.format(val_loss)+',val_accuarcy='+'{:.4f}'.format(val_acc)+',val_auc='+'{:.4f}'.format(val_auc))

            if val_acc > max_accuarcy:
                save_path = self.saver.save(self.sess,self.checkpoint_path)
                max_accuarcy = val_acc
                print('Model_Checkpointed to %s'%save_path)


    def predict(self,data_x,temperature=1.0):
        return self.sess.run(self.prediction,feed_dict={self.X:data_x,self.flag: True, self.softmax_temperature: temperature})
    
    def run_inference(self,dataset):
        test_images,test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuarcy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           self.flag: True,
                                                                           self.softmax_temperature: 1.0
                                                                           }))
        start = time.time()
        pred = self.predict(test_images)
        print('Time consumes: ' + str(time.time() - start) + ' seconds')
        test_precision,test_recall,test_f1,test_auc = self.evaluate(test_images, test_labels, 1.0, state='test')
        print('Testing Precsion:'+str(test_precision)+'\n'+'Testing Recall:'+str(test_recall)+'\n'+'Testing F1:'+str(test_f1)+'Testing AUC:'+str(test_auc))


    def load_model_from_file(self,path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
              
