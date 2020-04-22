# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:22:44 2019

@author: Zhu Zhi Yu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

mndata = MNIST('MNIST_data')
images, labels = mndata.load_training()

images_array = np.zeros(shape=(60000,28,28))

for i in range(len(images)):
    images_array[i] = np.asarray(images[i]).reshape(28,28)
    
#%% GAN
from models import GenerativeAdversarialNetwork

tf.reset_default_graph()

gan = GenerativeAdversarialNetwork(input_shape=28,
                                   vector_shape=100,
                                   G_learning_rate=1e-4,
                                   D_learning_rate=3e-4)

gan.setup_tf_board()

gan.fit(images_array,pre_train_iter=300,total_iter=100000,batch_size = 50)

#%%
fake = gan.fakeX_list
fake = np.asarray(fake)
plt.imshow(fake[-1].reshape(28,28),cmap='gray')
plt.colorbar()
fake_ = np.concatenate((fake,fake,fake),axis=3)
fake_list = []
for i in range(fake_.shape[0]):
    fake_list.append(np.uint8(fake_[i]*255))

#%% write gif
#gan.test_plot()

#from array2gif import write_gif
    
#write_gif(fake_list,'GAN_result.gif',fps=5)
