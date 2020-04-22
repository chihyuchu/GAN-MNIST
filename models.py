# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:19:32 2019

@author: Zhu Zhi Yu
"""

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class GenerativeAdversarialNetwork:
    def __init__(self,
                 input_shape,
                 vector_shape,
                 G_learning_rate=1e-4,
                 D_learning_rate=3e-4):

        self.input_shape  = input_shape
        self.vector_shape = vector_shape

        self.Dweights = None
        self.Dbiases = None
        self.Gweights = None
        self.Gbiases = None
        
        # reset default graph
        tf.reset_default_graph()
        # initialize new graph
        self.graph = tf.Graph()
        # build graph
        self.build(G_learning_rate,D_learning_rate)

        
    def build(self,G_learning_rate,D_learning_rate):
        # build graph
        
        with self.graph.as_default():
            with tf.device("/gpu:0"):
                ### inputs and targets for discriminator network
                self.real_X = tf.placeholder(dtype = tf.float32,\
                                             shape = (None,self.input_shape,self.input_shape))
                
                self.real_X_rs = tf.reshape(self.real_X,(-1,self.input_shape,self.input_shape,1))
                
                self.noise_vectors = tf.placeholder(dtype = tf.float32,\
                                                    shape = (None,self.vector_shape))
                
                
                
                self.fake_X = self.G_structure(inputs = self.noise_vectors)
                
                self.D_for_real = self.D_structure(inputs = self.real_X_rs)
                
                self.D_for_fake = self.D_structure(inputs = self.fake_X, reuse_variables = True)
                
                
                
                self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_for_real,
                                                                                     labels=tf.ones_like(self.D_for_real)))
                
                self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_for_fake,
                                                                                     labels=tf.zeros_like(self.D_for_fake)))
                
                self.D_loss =  self.D_loss_real + self.D_loss_fake
                
                self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_for_fake,
                                                                                     labels=tf.ones_like(self.D_for_fake)))
                
                optimizer = tf.train.AdamOptimizer
                
                self.tvars = tf.trainable_variables()
                self.d_vars = [var for var in self.tvars if 'D' in var.name]
                self.g_vars = [var for var in self.tvars if 'G' in var.name]
                
                self.train_op_for_D_fake = optimizer(D_learning_rate).minimize(self.D_loss_fake,var_list = self.d_vars)
                self.train_op_for_D_real = optimizer(D_learning_rate).minimize(self.D_loss_real,var_list = self.d_vars)
                
                #self.train_op_for_D = optimizer(D_learning_rate).minimize(self.D_loss,var_list=self.d_vars)
                
                self.train_op_for_G = optimizer(G_learning_rate).minimize(self.G_loss,var_list=self.g_vars)
                
                self.sess = tf.Session()
                
                ### Initialization  
                self.init_op = tf.global_variables_initializer()
                            
            
    def D_structure(self, inputs,reuse_variables = None):
        
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                if not (self.Dweights) and (not self.Dbiases):
                    kernel_size = 5
                    channel_num = (1,32,64)
                    fc_layer_shape = (7*7*64,1024,256,1)
                    bias_init = 0.0
                    weig_init = 0.02
                    
                    self.Dweights = {
                            'conv1':tf.Variable(tf.truncated_normal(shape=(kernel_size,kernel_size,\
                                                                           channel_num[0],channel_num[1]))*weig_init,name='Dw_conv1'),
                            'conv2':tf.Variable(tf.truncated_normal(shape=(kernel_size,kernel_size,\
                                                                           channel_num[1],channel_num[2]))*weig_init,name='Dw_conv1'),
                            'fc1':tf.Variable(tf.truncated_normal(shape=(fc_layer_shape[0],fc_layer_shape[1]))*weig_init,name='Dw_fc1'),
                            'fc2':tf.Variable(tf.truncated_normal(shape=(fc_layer_shape[1],fc_layer_shape[2]))*weig_init,name='Dw_fc2'),
                            'fc3':tf.Variable(tf.truncated_normal(shape=(fc_layer_shape[2],fc_layer_shape[3]))*weig_init,name='Dw_fc3'),
                            }
                    
                    self.Dbiases = {
                            'conv1':tf.Variable(tf.ones(shape=channel_num[1])*bias_init,name='Db_conv1'),
                            'conv2':tf.Variable(tf.ones(shape=channel_num[2])*bias_init,name='Db_conv2'),
                            'fc1':tf.Variable(tf.ones(shape=(fc_layer_shape[1]))*bias_init,name='Db_fc1'),
                            'fc2':tf.Variable(tf.ones(shape=(fc_layer_shape[2]))*bias_init,name='Db_fc2'),
                            'fc3':tf.Variable(tf.ones(shape=(fc_layer_shape[3]))*bias_init,name='Db_fc3'),
                            }
                
                ### structure
                conv1 = self.get_conv_layer(inputs,self.Dweights['conv1'],
                                            self.Dbiases['conv1'],stride=(1,1),
                                            activation=tf.nn.relu,pooling=True)
                
                conv2 = self.get_conv_layer(conv1,self.Dweights['conv2'],
                                            self.Dbiases['conv2'],stride=(1,1),
                                            activation=tf.nn.relu,pooling=True)
                
                flatten = self.get_flatten_layer(conv2)
                
                dense1 = self.get_dense_layer(flatten,self.Dweights['fc1'],
                                              self.Dbiases['fc1'],activation=tf.nn.relu)
                
                dense2 = self.get_dense_layer(dense1,self.Dweights['fc2'],
                                              self.Dbiases['fc2'],activation=tf.nn.relu)
                
                D_score = self.get_dense_layer(dense2,self.Dweights['fc3'],
                                            self.Dbiases['fc3'])
                
                return D_score
    
    def G_structure(self, inputs, reuse_variables=None):
        
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            if (not self.Gweights) and (not self.Gbiases):
                
                kernel_size = 5
                channel_num = (64,64,64,1)
                fc_layer_shape = (100,56*56)
                bias_init = 0.02
                weig_init = 0.02
                
                self.Gweights = {
                        'fc':tf.Variable(tf.truncated_normal(shape=(fc_layer_shape[0],fc_layer_shape[1]))*weig_init,name='Gw_fc'),
                        'conv1':tf.Variable(tf.truncated_normal(shape=(kernel_size,kernel_size,\
                                                                       channel_num[1],channel_num[0]))*weig_init,name='Gw_conv1'),
                        'conv2':tf.Variable(tf.truncated_normal(shape=(kernel_size,kernel_size,\
                                                                       channel_num[2],channel_num[1]))*weig_init,name='Gw_conv2'),
                        'conv3':tf.Variable(tf.truncated_normal(shape=(1,1,\
                                                                       channel_num[2],channel_num[3]))*weig_init,name='Gw_conv3')
                        }
                
                self.Gbiases = {
                        'fc':tf.get_variable('Gb_fc', [fc_layer_shape[1]],
                                             initializer=tf.truncated_normal_initializer(stddev=bias_init)),
                        'conv1':tf.get_variable('Gb_conv1', [channel_num[1]],
                                                initializer=tf.truncated_normal_initializer(stddev=bias_init)),
                        'conv2':tf.get_variable('Gb_conv2', [channel_num[2]],
                                                initializer=tf.truncated_normal_initializer(stddev=bias_init)),
                        'conv3':tf.get_variable('Gb_conv3', [channel_num[3]],
                                                initializer=tf.truncated_normal_initializer(stddev=bias_init))
                        }
             ### structure
            fc = self.get_dense_layer(inputs,self.Gweights['fc'],
                                      self.Gbiases['fc'],activation=tf.nn.relu)
            
            reshape = self.get_reshape_layer(fc,[7,7,64])
            
            output_shape1 = tf.stack([tf.shape(self.noise_vectors)[0],14,14,64])
            
            dconv1 = self.get_deconv_layer(reshape, self.Gweights['conv1'],
                                        self.Gbiases['conv1'],
                                        output_shape=output_shape1,stride=(2,2),
                                        activation=tf.nn.relu)
            
            dconv1_norm = tf.contrib.layers.batch_norm(dconv1, epsilon=1e-5, scope='Gb_conv1')
            
            output_shape2 = tf.stack([tf.shape(self.noise_vectors)[0],28,28,64])
            
            dconv2 = self.get_deconv_layer(dconv1_norm, self.Gweights['conv2'],
                                        self.Gbiases['conv2'],
                                        output_shape=output_shape2,stride=(2,2),
                                        activation=tf.nn.relu)
            
            dconv2_norm = tf.contrib.layers.batch_norm(dconv2, epsilon=1e-5, scope='Gb_conv2')
            
            fake_X = self.get_conv_layer(dconv2_norm, self.Gweights['conv3'],
                                        self.Gbiases['conv3'],
                                        stride=(1,1))
            
            return tf.sigmoid(fake_X)
                
                
    def get_conv_layer(self,input_layer,
                       weight,bias,stride=(1,1),
                       padding='SAME',activation=None,batch_norm=None,norm_scope=None,pooling=None):
        
        x = tf.add(
                tf.nn.conv2d(input_layer,
                             weight,
                             [1,stride[0],stride[1],1],
                             padding=padding) , bias)
        
        if activation:
            x = activation(x)
        
        if batch_norm and norm_scope:
            x = tf.contrib.layers.batch_norm(x, epsilon=1e-5,scope=norm_scope)
        
        if pooling:
            x = tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        return x
    
    def get_deconv_layer(self,input_layer,
                         weight,bias,
                         output_shape,stride=(1,1),padding='SAME',
                         activation=None):
        
        x = tf.add(
                tf.nn.conv2d_transpose(input_layer,
                                       weight,
                                       output_shape,
                                       [1,stride[0],stride[1],1],
                                       padding=padding), bias)
        
        if activation:
            x = activation(x)
        
        return x
    
    def get_dense_layer(self,input_layer,
                        weight,bias,activation=None):
        
        x = tf.add(
                tf.matmul(input_layer,weight), bias)
        
        if activation:
            x = activation(x)
        
        return x
    
    def get_flatten_layer(self,input_layer):
        shape = input_layer.get_shape().as_list()
        n = 1
        for s in shape[1:]:
            n = n*s
        
        x = tf.reshape(input_layer,shape = (-1,n))
        
        return x
    
    def get_reshape_layer(self,input_layer,shape):
        
        x = tf.reshape(input_layer,shape = (-1,shape[0],shape[1],shape[2]))
        
        return x
    
    def setup_tf_board(self):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        
            tf.summary.scalar('Generator_loss',self.G_loss)
            tf.summary.scalar('Discriminator_loss',self.D_loss)
            
            #image_for_tfboard = self.G_structure(self.noise_vectors,reuse_variables=True)
            
            #tf.summary.image('Generated_images',image_for_tfboard,5)
            
            merge = tf.summary.merge_all()
            logdir = "D:\logfile\GAN_MNIST"
            writer = tf.summary.FileWriter(logdir=logdir,graph=self.sess.graph)
    
    def fit(self, x, pre_train_iter = 300, total_iter = 10000,test_data=None,batch_size=None):
        
        N = x.shape[0]
        index = np.arange(N)
        
        # normalize x
        norm_x = np.zeros_like(x,dtype=np.float32)
        for i,image in enumerate(x):
            norm_x[i] = image/255
        
        self.D_pt_loss_curve = []
        self.D_loss_curve = []
        self.G_loss_curve = []
        self.fakeX_list = []
        
        if not batch_size:
            batch_size = 100
        
        #self.sess = tf.Session(graph = self.graph)
        self.sess.run(self.init_op)
        print('start training...')
        start_time = time.time()
        print("Pre-train stage")
        
        ### Pre-train
        
        for i in range(pre_train_iter):
            np.random.shuffle(index)
            batch_noise_vector = np.random.normal(size=(batch_size,self.vector_shape))
            batch_x = norm_x[index[0:batch_size]]
            _, __, dloss = self.sess.run([self.train_op_for_D_real,self.train_op_for_D_fake,
                                       self.D_loss],
                          feed_dict = {self.real_X:batch_x,
                                       self.noise_vectors:batch_noise_vector})
            if (i % 30 == 0):
                print("D loss:{:.4f}".format(dloss))
            
            # draw loss curve
            self.D_pt_loss_curve.append(dloss)
            
        print("Cycle train stage")
        for i in range(total_iter):
            
            np.random.shuffle(index)
            
            batch_x = norm_x[index[0:batch_size]]
            batch_noise_vector = np.random.normal(size=(batch_size,self.vector_shape))
            
            feed_dict = {self.real_X:batch_x,
                         self.noise_vectors:batch_noise_vector}
            
            ## Training discriminator
            _, __,dloss,gloss = self.sess.run([self.train_op_for_D_real,self.train_op_for_D_fake,
                                                    self.D_loss,
                                                    self.G_loss], feed_dict = feed_dict)
            
            ## Training generator
            batch_noise_vector = np.random.normal(size=(batch_size,self.vector_shape))
            _,fakeX = self.sess.run([self.train_op_for_G, self.fake_X],
                              feed_dict = {self.noise_vectors:batch_noise_vector})
            
            
            if (i % 100 == 0):
                print("Iteration {}/{}".format(i+100,total_iter))
                print("D loss:{}\nG loss:{}".format(dloss,gloss))
                plt.figure()
                plt.imshow(fakeX[0].reshape(28,28),cmap='Greys')
                plt.show()
                ## draw loss curve
                self.D_loss_curve.append(dloss)
                self.G_loss_curve.append(gloss)
                
                ## fake X in progress
                self.fakeX_list.append(fakeX[0])
                
        end_time = time.time()
        print("Total time collasped:{:.2f}s".format(end_time-start_time))
    
    def test_plot(self):
        noise_vec = np.random.normal(size=(9,self.vector_shape))
        fake = self.sess.run(self.fake_X,feed_dict={self.noise_vectors:noise_vec})
        fake = fake.reshape(9,28,28)
        plt.figure(1)
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(fake[i],cmap='gray')
            plt.box(False)
        
        plt.figure(2)
        plt.plot(self.D_loss_curve)