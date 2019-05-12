#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# In[1]:


#!/usr/bin/env python


# In[1]:


'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from csv import reader

# import

from keras.layers import Lambda, Input, Dense
from keras.layers import Dropout
# keras lambda layer
from keras.models import Model
# from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import tensorflow as tf
# from keras.layers import Input, Dense
# from keras.models import Model
import csv
from os import listdir
from os.path import isfile, join
from scipy.misc import imread
# import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random as rd
# import cv2

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon


# In[2]:

# inputno = 0/0
# inp11 = 0
# out11 = 0

# def getnewloss(x2):

#     outputno = outputno + 1
#     return binary_crossentropy(x1,x2)


# def getloss(x1):
#     # output = outputs[inputno]
#     # inputindex = imindex[inputno]
#     outputno = 0
#     losses1 = tf.map_fn(lambda x : getnewloss(x),outputs)
#     inputno = inputno + 1
#     return tf.reduce_sum(losses1)

def classcrossentropy(inputs,outputs):
    loss = 0
    i = 0

    inputs = inputs[1]
    # outputs = 
    recon_loss = -tf.reduce_sum(
    inputs * tf.log(1e-5+outputs) + 
    (1-inputs) * tf.log(1e-5+1-outputs), 
    axis=1
)

    return recon_loss



    # print(tf.shape(inputs))
    # print(outputs.shape)
    # print(inputs[0].shape)
    # print(inputs[1].shape)
    
    # print(inputs.shape)
    #
    # return binary_crossentropy(inputs[1],outputs)
    
#     tf.map_fn(
#     fn,
#     elems,
#     dtype=None,
#     parallel_iterations=None,
#     back_prop=True,
#     swap_memory=False,
#     infer_shape=True,
#     name=None
# )
                                   
    # inp11 = inputs[0]
    # out11 = outputs                                                     
    # iminput = inputs[0]
    # imindex = inputs[1]
    # indices = [i for i, x in enumerate(my_list) if x == "whatever"]
    # outputs[:] = outputs[:].append(imindex)
    # for rec in outputs:
        # loss = rec
    # outputno = 0
    # inputno = 0
    # losses = tf.map_fn(lambda x : getloss(x),)

    # return tf.reduce_sum(losses,axis = 1)

    # inputs = inputs[0]

    # epsilon = 1e-10
    # epsilon = epsilon*np.ones((128,2048))

    # recon_loss = -tf.reduce_sum(
    #     inputs * tf.log(1e-10+outputs) + 
    #     (1-inputs) * tf.log(1e-10+1-outputs), 
    #     axis=1
    # )

    # recon_loss = tf.reduce_mean(recon_loss)

    # Latent loss
    # KL divergence: measure the difference between two distributions
    # Here we measure the divergence between 
    # the latent distribution and N(0, 1)
    # latent_loss = -0.5 * tf.reduce_sum(
    #     1 + z_log_sigma_sq - tf.square(z_mu) - 
    #     tf.exp(z_log_sigma_sq), axis=1)
    # latent_loss = tf.reduce_mean(latent_loss)

    # total_loss = recon_loss + latent_loss
    # return recon_loss

    # j = 0
    # # return binary_crossentropy(inputs[0],outputs)
    # inputim = inputs[0]
    # inputlabel = inputs[1]

    # for i in range(batch_size):
    #     # j = 0
    #     for j in range(batch_size):
    #         # j+= 1
    #         # input1 = inputim[i]
    #         input2 = inputim[j]
    #         k1 = inputlabel[i]
    #         k2 = inputlabel[j]
    #         output1 = outputs[i]
    #         # print(k1)
    #         # print(k2)
    #         # if(k1 == k2):
    #         loss = loss + binary_crossentropy(input2,output1)
    #     # i+= 1
    # return loss




    # print(inputs1.shape)
    # # if(runmode == "single"):
    # # return binary_crossentropy(inputs,outputs)
    

    # i = 0
    # for inputs in inputs1:
    #     outputs = outputs1[i]
    #     imlabel = 0
    #     i1 = 0
    #     for im1 in flowerlist:
    #         # if(im1 == inputs):
    #         if(np.array_equal(im1,inputs)):
    #             imlabel = labels[i1]
    #             break
    #         i1 = i1+1
    #     print(imlabel)
    # #     for key in dict1:    
    #     loss = 0
    #     i1 = 0
    #     for image1 in flowerlist:
    #         if(labels[i1] == imlabel):
    #             if(loss == 0):
    #                 loss = binary_crossentropy(image1,outputs)
    #             else:
    #                 loss = loss + binary_crossentropy(image1,outputs)
    #         i1 += 1
    #     # return binary_crossentropy(inputs,outputs)
    #     print(loss)
    #     print(binary_crossentropy(image1,outputs))
    #     i = i+1
    # return K.sum(loss)


# In[2]:


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[3]:


# In[3]:


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# In[6]:


# In[4]:
# folder = "jpg"
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[9]:


# In[17]:


# dict1 = {}
# mypath = "/home/anurag/projects/VAEProj/102flowers/jpg"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# sort(onlyfiles)
# onlyfiles.sort()
filename = 'flowersXception.csv'

featurestotal = load_csv(filename)
# labels = labels[0]    
# print(onlyfiles)
# i1 = 1
# seltol = 1300
#seltol = 10
# selected1 = 800
# for selected1 in [200]:
# samplpositive = 600
#samplpositive = 10
# featurestotal = []
# i1 = 0
# for image in onlyfiles:
#     if(image[-3:] != "jpg"):
#         continue
#     # i1 = i1 + 1
#     # print("efrffrwf")

#     im1 = mypath+"/"+ image
#     # print(im1)
#     # print(labels[i1])
#     # im = imread(im1,cv2.COLOR_BGR2GRAY)
#     im = imread(im1)
#     im = im/float(255)
#     # print(im)
#     res = cv2.resize(im, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
#     #res = cv2.resize(im, dsize=(500,500))
#     # res = im
#     # k1


#     k1 = np.reshape(res,-1)
#     # np.append(k1,labels[i1])
#     # print(k)
#     k1 = list(k1)
#     k1.append(labels[i1])
#     # print(k1)

# #     dict1[k1] = labels[i1] 
#     # print(k1.shape)
#         # print(i1)
#     print(i1)
#     featurestotal.append(k1)
#     i1 = i1+ 1
#     # if(i1 == 85):
#         # break



# flowerlist = []

rd.shuffle(featurestotal)
# flowerlist.extend(featurestotal)

# flowerlist contains the required entries



x_train = featurestotal[:8180]
x_test = featurestotal[8180:]


# x_train = featurestotal[:4000]
# x_test = featurestotal[4000:]


# x_train = featurestotal[:400]
# x_test = featurestotal[400:]


x_train = np.array(x_train,dtype = float)
x_test = np.array(x_test,dtype = float)

print(x_train)
print(x_test)


x_trainlabel = x_train[:,2048]
x_train = x_train[:,0:2048]
x_testlabel = x_test[:,2048]
x_test = x_test[:,0:2048]

print(x_testlabel)
print(x_trainlabel)


a = np.zeros(2048,dtype = float)
 # = np.repeat(a[:, :, np.newaxis], 3, axis=2)

x_trainlabelsum = np.repeat(a[np.newaxis,:], 102,axis = 0)
x_testlabelsum = np.repeat(a[np.newaxis,:], 102,axis = 0)
print(x_trainlabelsum[0].shape)

# x_testlabelsum = []


# In[18]:


trainnum = np.zeros(102)
testnum = np.zeros(102)

# In[18]:
trainonehotlabel = []
testonehotlabel = []


i = 0
for fq in x_trainlabel:
    image = x_train[i]
    fq = int(fq)
    # print(fq)
    fq = fq-1
#     print(fq)
#     print(x_trainlabelsum[fq].shape)
#     print(type(image))
#     print(image.shape)
#     print(x_trainlabelsum[fq])
# #     print(type(image))
#     print(image)
    
    x_trainlabelsum[fq] = np.add(x_trainlabelsum[fq],image)
    trainnum[fq] = trainnum[fq] + 1
    arr = np.zeros(102)
    arr[fq] = 1
    trainonehotlabel.append(arr[fq])

    i = i+1

i = 0
for fq in x_testlabel:
    fq = fq-1
    im = x_test[i]
    fq = int(fq)
    x_testlabelsum[fq] = np.add(x_testlabelsum[fq],im)
    testnum[fq] = testnum[fq] + 1

    arr = np.zeros(102)
    arr[fq] = 1
    trainonehotlabel.append(arr[fq])

    i = i+1


print(trainonehotlabel)
print(testonehotlabel)
exit()

x_train2 = np.copy(x_train)
x_test2 = np.copy(x_test)

i = 0
for d2 in x_trainlabel:
    d2 = int(d2)
    d2 = d2-1
    x_train2[i] = x_trainlabelsum[d2]/(float(trainnum[d2]))
    i = i+1

i = 0
for d2 in x_testlabel:
    d2 = int(d2)
    d2 = d2-1
    x_test2[i] = x_testlabelsum[d2]/(float(trainnum[d2]))
    i = i+1

# x_train

# x_trainlabel = np.array(x_trainlabel)
# x_testlabel = np.array(x_testlabel)

# x_train


# print(classcrossentropy(x_train[0],x_train[1]))
# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


# In[19]:


# image_size = x_train[0].shape[0]
# print(image_size)
original_dim = 2048
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# print(im)
# network parameters
input_shape = (original_dim, )
# print("input_shape")
# print(input_shape)
intermediate_dim = 512 # of original layers
intermediate_dim1 = 512 # of extra layers
classifierdim = 512 # of classifiers
classifieroutputdim = 102

batch_size = 128
latent_dim = 2
epochs = 300


# model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# VAE model = encoder + decoder
# build encoder model
input1 = Input(shape=input_shape, name='encoder_input_im')
input2 = Input(shape=input_shape, name='encoder_input_imsum')
input3 = Input(shape=input_shape,name='encoder_input_labels')
# inputs = Input(shape=1, name='encoder_input_label')
# drop1 = Dropout(0.2)(input1)
x = Dense(intermediate_dim, activation='relu')(input1)
x = Dense(intermediate_dim1, activation='relu')(x)

# x = Dense(intermediate_dim1, activation='relu')(x)
# drop1 = Dropout(0.2)(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

inps = [input1,input2]
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model

# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# encoder = Model(inputs = [input1,input2], [z_mean, z_log_var, z], name='encoder')
encoder = Model(inps, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_samples')

x = Dense(intermediate_dim, activation='relu')(latent_inputs)
# drop2 = Dropout(0.2)(x)
x = Dense(intermediate_dim1, activation='relu')(x)
# x = Dense(intermediate_dim1, activation='relu')(x)

outputs = Dense(original_dim, activation='sigmoid')(x)


# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

#building classifier models
classifier_inputs = Input(shape=(latent_dim,), name="class1")

x = Dense(classifierdim,activation = 'relu')(classifier_inputs)
# x = Dense(classifierdim,activation = 'relu')(x)  # additional layer
classifier_output = Dense(classifieroutputdim,activation = 'softmax')(x)

classifier = Model(classifier_inputs,classifier_output,name="classifier")

classifier.summary()

# instantiate VAE model
output1 = decoder(encoder(inps)[2])
output2 = classifier(encoder(inps)[2]) # classifing on smaple values
# output2 = classifier(encoder(inps)[0]) # classifing on z_means
outputs = [output1,output2]

vae = Model(inps, outputs, name='vae_mlp + classifier')


# inputs = Input(shape=input_shape, name='encoder_input')
# x = Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = Dense(latent_dim, name='z_mean')(x)
# z_log_var = Dense(latent_dim, name='z_log_var')(x)

# # use reparameterization trick to push the sampling out as input
# # note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# # instantiate encoder model
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# encoder.summary()
# # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# # build decoder model
# latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# x = Dense(intermediate_dim, activation='relu')(latent_inputs)
# outputs = Dense(original_dim, activation='sigmoid')(x)

# # instantiate decoder model
# decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()
# # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# # instantiate VAE model
# outputs = decoder(encoder(inputs)[2])
# vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    # dict1 = {}
    kllosswt = 1;
    parser = argparse.ArgumentParser()

    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    # data = (x_test, y_test)

# here1
    # VAE loss = mse_loss or xent_loss + kl_loss
    
    # runmode = input("single or class encoder ")
    # z33 = np.identity(1000)


    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = classcrossentropy(inps,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = kl_loss*-0.5*kllosswt
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    # plot_model(vae,
               # to_file='vae_mlp.png',
               # show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit([x_train,x_train2],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x_test,x_test2], None))
        vae.save_weights('vae_mlp_mnist.h5')



    # for i in test2
    # plot_results(models,
                 # data,
                 # batch_size=batch_size,
                 # model_name="vae_mlp")

    encoder, decoder = models
    # encoder, decoder = models
    x_test, y_test = [x_train,x_train2],x_trainlabel
    # os.makedirs(model_name, exist_ok=True)

    # filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    # plt.savefig(filename)
    plt.show()
# In[ ]:


# In[ ]:

