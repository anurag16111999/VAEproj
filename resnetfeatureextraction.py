from tqdm import tqdm 
# import tensorflow as tf 
# from keras.applications.resnet50 import ResNet50
# from keras.applications.xception import Xception, preprocess_input
# from keras.layers import Flatten, Input
# from keras.models import Model
from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os
# from google.cloud import storage
from io import BytesIO
import time
import cv2
import csv
from csv import reader

# import sort
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


start = time.time()

# base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
# input = Input(shape=(224,224,3),name = 'image_input')
# x = base_model(input)
# x = Flatten()(x)
# model = Model(inputs=input, outputs=x)
# weights = "imagenet"
# base_model = Xception(weights=weights)
# model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
# image_size = (299, 299)
# 
# base_model = ResNet50(weights=weights)
# model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
# image_size = (224, 224)

# model = ResNet50(weights='imagenet', pooling=max,input_shape=(224, 224, 3)) 
####### GENERATING FEATURES

# features1 = open('train_featues.txt', 'w+') #to save features
train_dir = '/home/anurag/projects/VAEProj/102flowers/jpg/'
train_files = os.listdir(train_dir)
train_files.sort()
# sort(train_files)
 
filename = 'FileName.csv'
labels = load_csv(filename)
labels = labels[0]    
# print(labels)
i = 0

start = time.time()
feat = []
i = 0;
for file in  tqdm(train_files):
    # try:
    # print()
    print(file)
    # print(i)
    # f = cv2.imread(os.path.join(train_dir, file))
    # print()
    # print(f)
    # f = f/float(255)
    # print(f)
    # print(f.shape)
    # res = cv2.resize(f, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
    # print(res.shape)
    # res = np.expand_dims(res,axis= 0)
    # print(res.shape)
    
    # res = (list(np.ndarray.flatten( ( (image.img_to_array (image.load_img(train_dir + file, target_size=(224, 224)))  ).astype(np.uint8)) ,'F') ) ).append(labels[i])
    res = image.load_img(train_dir + file, target_size=(224, 224))
    res = image.img_to_array(res)
    # print(res.shape)
    # res = np.expand_dims(res, axis=0) 
    # res = preprocess_input(res)
    # res = [list(xi) for xi in res]
    res = res.astype(np.uint8)
    # res = res.astype
    res = np.ndarray.flatten(res)
    # res 
    # res = [res]

    res = list(res)
    print(len(res))
    # features = model.predict(res)
    # print(type(res))

    # exit()
    # print(features)
    # print(features.shape)
    
    # features_reduce = features.squeeze()
    # features_reduce = list(features_reduce)
    # print(features_reduce)
    res.append(int(labels[i]))
    # print(features_reduce[2020:])
    # labels = labels + 1
    # feat.append(res)
    print(len(res))
    # break
    # print(str(len(feat)) + " <= lengthfeat")
    # print(str(len(feat[i])) + " <= lengthfeat")
    # with open('f__224_224.csv','a') as fd:
    #     fd.write(res)
    with open("flowersimages__224_224.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(res)
    
    i = i+1
    # features_reduce = kis 
    # print(features_reduce)
    # print(features_reduce.shape)
    # print(len)
    # exit()
    # break
    if(i == 350):
        break
    # features1.write(' '.join(str(x) for x in features.squeeze()) + '\n')
# except: # to skip not files in a wrong format
    # print("pass")
        # pass


# feat = np.array(feat)
# print(feat)
# np.save("foo.npy", feat)


# with open("flowersimages__224_224.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(feat)


# np.savetxt("flowers.csv", feat, delimiter=",")

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


# # print(i)
# end = time.time()
# print('\n\ntime spend: ' , (end - start)/60 , ' minutes \n\n')

# features1.close()