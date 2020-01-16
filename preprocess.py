# from keras.preprocessing.image import load_img,img_to_array
import os
import math
import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50,preprocess_input

X_train=[]
y_train = []

base_model = ResNet50(weights = 'imagenet',include_top = False)

emotion_folder = os.listdir('./Emotion')
for i in emotion_folder:
    # print(i)
    inner_folder = os.listdir('./Emotion/'+i+'/')
    print(inner_folder)
    for j in inner_folder:
        # print(j)/
        print('./Emotion/' + str(i) +'/' + str(j) + '/')
        txtfile = os.listdir('./Emotion/' + i +'/' + j + '/')
        # print(txtfile)
        if len(txtfile)!=0:
            with open('./Emotion/' + i +'/' + j + '/'+txtfile[0],'r') as file:
                label = int(file.read().split(' ')[3][0])
                print(label)
                # y_train.append(label)
            path = './cohn-kanade-images/' + i + '/' + j + '/'
            images1 = sorted(os.listdir(path),key=len)
            if '.DS_Store' in images1:
                images1.remove('.DS_Store')
            
            images = []
            # print(len(images1))
            if len(images1)>6:
                images = images1[-6:]
                # images = sorted(np.random.choice(images1,6))
                # for k in range(0,len(images1),int(len(images1)/7)):
                #     images.append(images1[k])
            else:
                images = images1
            # print(len(images))
            # print(images)
            imgs = []
            for image in images:
                # print(image)
                im = cv2.imread(path+ image)
                im = cv2.resize(im,(224,224),interpolation = cv2.INTER_AREA)
                # print(im.shape)
                im = np.expand_dims(im,axis=0)
                im = preprocess_input(im)
                # im = load_img(path + image)
                # im = img_to_array(im,data_format="channels_first")
                # imgs.append(im)
                # print(im.shape)
                feature = base_model.predict(im)
                # print(feature[0].shape)
                # print(feature.shape)
                # 1x7x7x2048
                imgs.append(feature[0])
            X_train.append(imgs)
            y_train.append(label)
            # imgs = np.array(imgs)
            # print(imgs.shape)
            # X_train.append(imgs)

X_train = np.array(X_train)
print(X_train.shape)
y_train = np.array(y_train)
print(y_train.shape)
np.save('X_train_resnet_vide.npy',X_train)
np.save('y_train_resnet_video.npy',y_train)

                
