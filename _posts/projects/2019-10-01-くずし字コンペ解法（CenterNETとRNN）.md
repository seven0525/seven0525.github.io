---
layout: post
title:  "くずし字コンペ解法(CenterNET/RNN)"
date:   2019-10-01
excerpt: "Kaggle出場したときのやつ"
project: true
tag:
- Machine Learning 
- Kaggle
comments: false
---

# 結果
- 106th/ 293 teams(2652 entries)
- Score 0.650

[詳しくはこちら](https://www.kaggle.com/c/kuzushiji-recognition/overview)


# 以下コード


## CenterNET用のデータ整形

```python
import numpy as np
import json
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import random
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import KFold,train_test_split
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from keras.models import Model
from keras.objectives import mean_squared_error
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
import os  

from keras.optimizers import Adam, RMSprop, SGD
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
```


```python
path_1="./data/train.csv"
path_2="./data/train_images/"
path_3="./data/test_images/"
path_4="./data/sample_submission.csv"
df_train=pd.read_csv(path_1)
#print(df_train.head())
#print(df_train.shape)
df_train=df_train.dropna(axis=0, how='any')#you can use nan data(page with no letter)
df_train=df_train.reset_index(drop=True)
#print(df_train.shape)

annotation_list_train=[]
category_names=set()

for i in range(len(df_train)):
  ann=np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,x,y,width,height for each picture
  category_names=category_names.union({i for i in ann[:,0]})

category_names=sorted(category_names)
dict_cat={list(category_names)[j]:str(j) for j in range(len(category_names))}
inv_dict_cat={str(j):list(category_names)[j] for j in range(len(category_names))}
#print(dict_cat)
  
for i in range(len(df_train)):
  ann=np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,left,top,width,height for each picture
  for j,category_name in enumerate(ann[:,0]):
    ann[j,0]=int(dict_cat[category_name])  
  ann=ann.astype('int32')
  ann[:,1]+=ann[:,3]//2#center_x
  ann[:,2]+=ann[:,4]//2#center_y
  annotation_list_train.append(["{}{}.jpg".format(path_2,df_train.loc[i,"image_id"]),ann])

print("sample image")
input_width,input_height=512, 512
img = np.asarray(Image.open(annotation_list_train[0][0]).resize((input_width,input_height)).convert('RGB'))
plt.imshow(img)
plt.show()

```

    sample image



![png](output_1_1.png)



```python
# get directory of test images
df_submission=pd.read_csv(path_4)
id_test=path_3+df_submission["image_id"]+".jpg"
```


```python
aspect_ratio_pic_all=[]
aspect_ratio_pic_all_test=[]
average_letter_size_all=[]
train_input_for_size_estimate=[]
resize_dir="resized/"
if os.path.exists(resize_dir) == False:os.mkdir(resize_dir)
for i in range(len(annotation_list_train)):
    with Image.open(annotation_list_train[i][0]) as f:
        width,height=f.size
        area=width*height
        aspect_ratio_pic=height/width
        aspect_ratio_pic_all.append(aspect_ratio_pic)
        letter_size=annotation_list_train[i][1][:,3]*annotation_list_train[i][1][:,4]
        letter_size_ratio=letter_size/area
    
        average_letter_size=np.mean(letter_size_ratio)
        average_letter_size_all.append(average_letter_size)
        train_input_for_size_estimate.append([annotation_list_train[i][0],np.log(average_letter_size)])#logにしとく
    

for i in range(len(id_test)):
    with Image.open(id_test[i]) as f:
        width,height=f.size
        aspect_ratio_pic=height/width
        aspect_ratio_pic_all_test.append(aspect_ratio_pic)


plt.hist(np.log(average_letter_size_all),bins=100)
plt.title('log(ratio of letter_size to picture_size))',loc='center',fontsize=12)
plt.show()
```


![png](output_3_0.png)


# Check Object Size
## Create Model


```python

category_n=1
import cv2
input_width,input_height=512, 512

def Datagen_sizecheck_model(filenames, batch_size, size_detection_mode=True, is_train=True,random_crop=True):
  x=[]
  y=[]
  
  count=0

  while True:
    for i in range(len(filenames)):
      if random_crop:
        crop_ratio=np.random.uniform(0.7,1)
      else:
        crop_ratio=1
      with Image.open(filenames[i][0]) as f:
        #random crop
        if random_crop and is_train:
          pic_width,pic_height=f.size
          f=np.asarray(f.convert('RGB'),dtype=np.uint8)
          top_offset=np.random.randint(0,pic_height-int(crop_ratio*pic_height))
          left_offset=np.random.randint(0,pic_width-int(crop_ratio*pic_width))
          bottom_offset=top_offset+int(crop_ratio*pic_height)
          right_offset=left_offset+int(crop_ratio*pic_width)
          f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))
        else:
          f=f.resize((input_width, input_height))
          f=np.asarray(f.convert('RGB'),dtype=np.uint8)          
        x.append(f)
      
      
      if random_crop and is_train:
        y.append(filenames[i][1]-np.log(crop_ratio))
      else:
        y.append(filenames[i][1])
      
      count+=1
      if count==batch_size:
        x=np.array(x, dtype=np.float32)
        y=np.array(y, dtype=np.float32)

        inputs=x/255
        targets=y       
        x=[]
        y=[]
        count=0
        yield inputs, targets



def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
  x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
  x_deep = BatchNormalization()(x_deep)   
  x_deep = LeakyReLU(alpha=0.1)(x_deep)
  x = Concatenate()([x_shallow, x_deep])
  x=Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
  x = BatchNormalization()(x)   
  x = LeakyReLU(alpha=0.1)(x)
  return x
  


def cbr(x, out_layer, kernel, stride):
  x=Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.1)(x)
  return x

def resblock(x_in,layer_n):
  x=cbr(x_in,layer_n,3,1)
  x=cbr(x,layer_n,3,1)
  x=Add()([x,x_in])
  return x  


#I use the same network at CenterNet
def create_model(input_shape, size_detection_mode=True, aggregation=True):
    input_layer = Input(input_shape)
    
    #resized input
    input_layer_1=AveragePooling2D(2)(input_layer)
    input_layer_2=AveragePooling2D(2)(input_layer_1)

    #### ENCODER ####

    x_0= cbr(input_layer, 16, 3, 2)#512->256
    concat_1 = Concatenate()([x_0, input_layer_1])

    x_1= cbr(concat_1, 32, 3, 2)#256->128
    concat_2 = Concatenate()([x_1, input_layer_2])

    x_2= cbr(concat_2, 64, 3, 2)#128->64
    
    x=cbr(x_2,64,3,1)
    x=resblock(x,64)
    x=resblock(x,64)
    
    x_3= cbr(x, 128, 3, 2)#64->32
    x= cbr(x_3, 128, 3, 1)
    x=resblock(x,128)
    x=resblock(x,128)
    x=resblock(x,128)
    
    x_4= cbr(x, 256, 3, 2)#32->16
    x= cbr(x_4, 256, 3, 1)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
 
    x_5= cbr(x, 512, 3, 2)#16->8
    x= cbr(x_5, 512, 3, 1)
    
    x=resblock(x,512)
    x=resblock(x,512)
    x=resblock(x,512)
    
    if size_detection_mode:
      x=GlobalAveragePooling2D()(x)
      x=Dropout(0.2)(x)
      out=Dense(1,activation="linear")(x)
    
    else:#centernet mode
    #### DECODER ####
      x_1= cbr(x_1, output_layer_n, 1, 1)
      x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
      x_2= cbr(x_2, output_layer_n, 1, 1)
      x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
      x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
      x_3= cbr(x_3, output_layer_n, 1, 1)
      x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n) 
      x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
      x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
      
      x_4= cbr(x_4, output_layer_n, 1, 1)

      x=cbr(x, output_layer_n, 1, 1)
      x= UpSampling2D(size=(2, 2))(x)#8->16 tconvのがいいか

      x = Concatenate()([x, x_4])
      x=cbr(x, output_layer_n, 3, 1)
      x= UpSampling2D(size=(2, 2))(x)#16->32
    
      x = Concatenate()([x, x_3])
      x=cbr(x, output_layer_n, 3, 1)
      x= UpSampling2D(size=(2, 2))(x)#32->64   128のがいいかも？ 
    
      x = Concatenate()([x, x_2])
      x=cbr(x, output_layer_n, 3, 1)
      x= UpSampling2D(size=(2, 2))(x)#64->128 
      
      x = Concatenate()([x, x_1])
      x=Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
      out = Activation("sigmoid")(x)
    
    model=Model(input_layer, out)
    
    return model
  
    


def model_fit_sizecheck_model(model,train_list,cv_list,n_epoch,batch_size=32):
    hist = model.fit_generator(
        Datagen_sizecheck_model(train_list,batch_size, is_train=True,random_crop=True),
        steps_per_epoch = len(train_list) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen_sizecheck_model(cv_list,batch_size, is_train=False,random_crop=False),
        validation_steps = len(cv_list) // batch_size,
        callbacks = [lr_schedule, model_checkpoint],#[early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 1
    )
    return hist

  

```


```python
K.clear_session()
model=create_model(input_shape=(input_height,input_width,3),size_detection_mode=True)
"""
# EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 10, verbose = 1)
# ModelCheckpoint
weights_dir = '/model_1/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(weights_dir + "val_loss{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,
                                      save_best_only = True, save_weights_only = True, period = 1)
# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)
"""
def lrs(epoch):
    lr = 0.0005
    if epoch>10:
        lr = 0.0001
    return lr

lr_schedule = LearningRateScheduler(lrs)
model_checkpoint = ModelCheckpoint("final_weights_step1.hdf5", monitor = 'val_loss', verbose = 1,
                                      save_best_only = True, save_weights_only = True, period = 1)
print(model.summary())
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 512, 512, 3)  0                                            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 256, 256, 16) 448         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 256, 256, 16) 64          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)       (None, 256, 256, 16) 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 256, 256, 3)  0           input_1[0][0]                    
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 256, 256, 19) 0           leaky_re_lu_1[0][0]              
                                                                     average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 128, 128, 32) 5504        concatenate_1[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 128, 128, 32) 128         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)       (None, 128, 128, 32) 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 128, 128, 3)  0           average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 128, 128, 35) 0           leaky_re_lu_2[0][0]              
                                                                     average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 64, 64, 64)   20224       concatenate_2[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 64, 64, 64)   256         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)       (None, 64, 64, 64)   0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 64, 64, 64)   36928       leaky_re_lu_3[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 64, 64, 64)   256         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)       (None, 64, 64, 64)   0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 64, 64, 64)   36928       leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 64, 64, 64)   256         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)       (None, 64, 64, 64)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 64, 64, 64)   36928       leaky_re_lu_5[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 64, 64, 64)   256         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)       (None, 64, 64, 64)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 64, 64, 64)   0           leaky_re_lu_6[0][0]              
                                                                     leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 64, 64, 64)   36928       add_1[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 64, 64, 64)   256         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)       (None, 64, 64, 64)   0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 64, 64, 64)   36928       leaky_re_lu_7[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 64, 64, 64)   256         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)       (None, 64, 64, 64)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 64, 64, 64)   0           leaky_re_lu_8[0][0]              
                                                                     add_1[0][0]                      
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 32, 32, 128)  73856       add_2[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 32, 32, 128)  512         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)       (None, 32, 32, 128)  0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 32, 32, 128)  147584      leaky_re_lu_9[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 32, 32, 128)  512         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 32, 32, 128)  147584      leaky_re_lu_10[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 32, 32, 128)  512         conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 32, 32, 128)  147584      leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 32, 32, 128)  512         conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 32, 32, 128)  0           leaky_re_lu_12[0][0]             
                                                                     leaky_re_lu_10[0][0]             
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 32, 32, 128)  147584      add_3[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 32, 32, 128)  512         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 32, 32, 128)  147584      leaky_re_lu_13[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 32, 32, 128)  512         conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, 32, 32, 128)  0           leaky_re_lu_14[0][0]             
                                                                     add_3[0][0]                      
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 32, 32, 128)  147584      add_4[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 32, 32, 128)  512         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 32, 32, 128)  147584      leaky_re_lu_15[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 32, 32, 128)  512         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_16 (LeakyReLU)      (None, 32, 32, 128)  0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, 32, 32, 128)  0           leaky_re_lu_16[0][0]             
                                                                     add_4[0][0]                      
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 16, 16, 256)  295168      add_5[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 16, 16, 256)  1024        conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_17 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_17[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 16, 16, 256)  1024        conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_18 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_18[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 16, 16, 256)  1024        conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_19 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_19[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 16, 16, 256)  1024        conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_20 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, 16, 16, 256)  0           leaky_re_lu_20[0][0]             
                                                                     leaky_re_lu_18[0][0]             
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 16, 16, 256)  590080      add_6[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 16, 16, 256)  1024        conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_21 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_21[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 16, 16, 256)  1024        conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_22 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, 16, 16, 256)  0           leaky_re_lu_22[0][0]             
                                                                     add_6[0][0]                      
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 16, 16, 256)  590080      add_7[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 16, 16, 256)  1024        conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_23 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_23[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 16, 16, 256)  1024        conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_24 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, 16, 16, 256)  0           leaky_re_lu_24[0][0]             
                                                                     add_7[0][0]                      
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 16, 16, 256)  590080      add_8[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 16, 16, 256)  1024        conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_25 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_25[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 16, 16, 256)  1024        conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_26 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, 16, 16, 256)  0           leaky_re_lu_26[0][0]             
                                                                     add_8[0][0]                      
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 16, 16, 256)  590080      add_9[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 16, 16, 256)  1024        conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_27 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_27[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, 16, 16, 256)  1024        conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_28 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, 16, 16, 256)  0           leaky_re_lu_28[0][0]             
                                                                     add_9[0][0]                      
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 8, 8, 512)    1180160     add_10[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, 8, 8, 512)    2048        conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_29 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 8, 8, 512)    2359808     leaky_re_lu_29[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, 8, 8, 512)    2048        conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_30 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 8, 8, 512)    2359808     leaky_re_lu_30[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, 8, 8, 512)    2048        conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_31 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 8, 8, 512)    2359808     leaky_re_lu_31[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, 8, 8, 512)    2048        conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_32 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, 8, 8, 512)    0           leaky_re_lu_32[0][0]             
                                                                     leaky_re_lu_30[0][0]             
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 8, 8, 512)    2359808     add_11[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, 8, 8, 512)    2048        conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_33 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 8, 8, 512)    2359808     leaky_re_lu_33[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, 8, 8, 512)    2048        conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_34 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    add_12 (Add)                    (None, 8, 8, 512)    0           leaky_re_lu_34[0][0]             
                                                                     add_11[0][0]                     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 8, 8, 512)    2359808     add_12[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, 8, 8, 512)    2048        conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_35 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 8, 8, 512)    2359808     leaky_re_lu_35[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, 8, 8, 512)    2048        conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_36 (LeakyReLU)      (None, 8, 8, 512)    0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    add_13 (Add)                    (None, 8, 8, 512)    0           leaky_re_lu_36[0][0]             
                                                                     add_12[0][0]                     
    __________________________________________________________________________________________________
    global_average_pooling2d_1 (Glo (None, 512)          0           add_13[0][0]                     
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 512)          0           global_average_pooling2d_1[0][0] 
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1)            513         dropout_1[0][0]                  
    ==================================================================================================
    Total params: 25,837,633
    Trainable params: 25,820,385
    Non-trainable params: 17,248
    __________________________________________________________________________________________________
    None



```python
train_list, cv_list = train_test_split(train_input_for_size_estimate, random_state = 111,test_size = 0.2)


learning_rate=0.0005
n_epoch=12
batch_size=32

model.compile(loss=mean_squared_error, optimizer=Adam(lr=learning_rate))
hist = model_fit_sizecheck_model(model,train_list,cv_list,n_epoch,batch_size)

#model.save_weights('final_weights_step1.h5')
model.load_weights('final_weights_step1.hdf5')
```


    ---------------------------------------------------------------------------


```python
predict = model.predict_generator(Datagen_sizecheck_model(cv_list,batch_size, is_train=False,random_crop=False),
                                  steps=len(cv_list) // batch_size)
target=[cv[1] for cv in cv_list]
plt.scatter(predict,target[:len(predict)])
plt.title('---letter_size/picture_size--- estimated vs target ',loc='center',fontsize=10)
plt.show()
```


    

```python
batch_size=1
predict_train = model.predict_generator(Datagen_sizecheck_model(train_input_for_size_estimate,batch_size, is_train=False,random_crop=False, ),
                                  steps=len(train_input_for_size_estimate)//batch_size)
```


    

```python
base_detect_num_h,base_detect_num_w=25,25
annotation_list_train_w_split=[]
for i, predicted_size in enumerate(predict_train):
  detect_num_h=aspect_ratio_pic_all[i]*np.exp(-predicted_size/2)
  detect_num_w=detect_num_h/aspect_ratio_pic_all[i]
  h_split_recommend=np.maximum(1,detect_num_h/base_detect_num_h)
  w_split_recommend=np.maximum(1,detect_num_w/base_detect_num_w)
  annotation_list_train_w_split.append([annotation_list_train[i][0],annotation_list_train[i][1],h_split_recommend,w_split_recommend])
for i in np.arange(0,1):
  print("recommended height split:{}, recommended width_split:{}".format(annotation_list_train_w_split[i][2],annotation_list_train_w_split[i][3]))
  img = np.asarray(Image.open(annotation_list_train_w_split[i][0]).convert('RGB'))
  plt.imshow(img)
  plt.show()
```


    
```python
category_n=1
output_layer_n=category_n+4
output_height,output_width=128,128

i=0

h_split=annotation_list_train_w_split[i][2]
w_split=annotation_list_train_w_split[i][3]
max_crop_ratio_h=1/h_split
max_crop_ratio_w=1/w_split
crop_ratio=np.random.uniform(0.5,1)
crop_ratio_h=max_crop_ratio_h*crop_ratio
crop_ratio_w=max_crop_ratio_w*crop_ratio

with Image.open(annotation_list_train_w_split[i][0]) as f:
        
        #random crop
        pic_width,pic_height=f.size
        f=np.asarray(f.convert('RGB'),dtype=np.uint8)
        top_offset=np.random.randint(0,pic_height-int(crop_ratio_h*pic_height))
        left_offset=np.random.randint(0,pic_width-int(crop_ratio_w*pic_width))
        bottom_offset=top_offset+int(crop_ratio_h*pic_height)
        right_offset=left_offset+int(crop_ratio_w*pic_width)
        img=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))

      
      
output_layer=np.zeros((output_height,output_width,(output_layer_n+category_n)))
for annotation in annotation_list_train_w_split[i][1]:

          x_c=(annotation[1]-left_offset)*(output_width/int(crop_ratio_w*pic_width))
          y_c=(annotation[2]-top_offset)*(output_height/int(crop_ratio_h*pic_height))
          width=annotation[3]*(output_width/int(crop_ratio_w*pic_width))
          height=annotation[4]*(output_height/int(crop_ratio_h*pic_height))
          
          top=np.maximum(0,y_c-height/2)
          left=np.maximum(0,x_c-width/2)
          bottom=np.minimum(output_height,y_c+height/2)
          right=np.minimum(output_width,x_c+width/2)
          
          if top>=output_height or left>=output_width or bottom<=0 or right<=0:#random crop(エリア外の除去)
            continue
          width=right-left
          height=bottom-top
          x_c=(right+left)/2
          y_c=(top+bottom)/2
          
        
        
          category=0#not classify
          heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)
                            *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))
          output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:])
          output_layer[int(y_c//1),int(x_c//1),category_n+category]=1
          output_layer[int(y_c//1),int(x_c//1),2*category_n]=y_c%1#height offset
          output_layer[int(y_c//1),int(x_c//1),2*category_n+1]=x_c%1
          output_layer[int(y_c//1),int(x_c//1),2*category_n+2]=height/output_height
          output_layer[int(y_c//1),int(x_c//1),2*category_n+3]=width/output_width

fig, axes = plt.subplots(1, 3,figsize=(15,15))
axes[0].set_axis_off()
axes[0].imshow(img)
axes[1].set_axis_off()
axes[1].imshow(output_layer[:,:,1])
axes[2].set_axis_off()
axes[2].imshow(output_layer[:,:,0])
plt.show()
```


    

```python
category_n=1
output_layer_n=category_n+4
output_height,output_width=128,128

def Datagen_centernet(filenames, batch_size):
  x=[]
  y=[]
  
  count=0

  while True:
    for i in range(len(filenames)):
      h_split=filenames[i][2]
      w_split=filenames[i][3]
      max_crop_ratio_h=1/h_split
      max_crop_ratio_w=1/w_split
      crop_ratio=np.random.uniform(0.5,1)
      crop_ratio_h=max_crop_ratio_h*crop_ratio
      crop_ratio_w=max_crop_ratio_w*crop_ratio
      
      with Image.open(filenames[i][0]) as f:
        
        #random crop
        
        pic_width,pic_height=f.size
        f=np.asarray(f.convert('RGB'),dtype=np.uint8)
        top_offset=np.random.randint(0,pic_height-int(crop_ratio_h*pic_height))
        left_offset=np.random.randint(0,pic_width-int(crop_ratio_w*pic_width))
        bottom_offset=top_offset+int(crop_ratio_h*pic_height)
        right_offset=left_offset+int(crop_ratio_w*pic_width)
        f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))
        x.append(f)      

      output_layer=np.zeros((output_height,output_width,(output_layer_n+category_n)))
      for annotation in filenames[i][1]:
        x_c=(annotation[1]-left_offset)*(output_width/int(crop_ratio_w*pic_width))
        y_c=(annotation[2]-top_offset)*(output_height/int(crop_ratio_h*pic_height))
        width=annotation[3]*(output_width/int(crop_ratio_w*pic_width))
        height=annotation[4]*(output_height/int(crop_ratio_h*pic_height))
        top=np.maximum(0,y_c-height/2)
        left=np.maximum(0,x_c-width/2)
        bottom=np.minimum(output_height,y_c+height/2)
        right=np.minimum(output_width,x_c+width/2)
          
        if top>=(output_height-0.1) or left>=(output_width-0.1) or bottom<=0.1 or right<=0.1:#random crop(out of picture)
          continue
        width=right-left
        height=bottom-top
        x_c=(right+left)/2
        y_c=(top+bottom)/2

        
        category=0#not classify, just detect
        heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)
                            *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))
        output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:])
        output_layer[int(y_c//1),int(x_c//1),category_n+category]=1
        output_layer[int(y_c//1),int(x_c//1),2*category_n]=y_c%1#height offset
        output_layer[int(y_c//1),int(x_c//1),2*category_n+1]=x_c%1
        output_layer[int(y_c//1),int(x_c//1),2*category_n+2]=height/output_height
        output_layer[int(y_c//1),int(x_c//1),2*category_n+3]=width/output_width
      y.append(output_layer)  
    
      count+=1
      if count==batch_size:
        x=np.array(x, dtype=np.float32)
        y=np.array(y, dtype=np.float32)

        inputs=x/255
        targets=y       
        x=[]
        y=[]
        count=0
        yield inputs, targets

def all_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    alpha=2.
    beta=4.

    heatmap_true_rate = K.flatten(y_true[...,:category_n])
    heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])
    heatmap_pred = K.flatten(y_pred[...,:category_n])
    heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))
    offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))
    sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))
    
    all_loss=(heatloss+1.0*offsetloss+5.0*sizeloss)/N
    return all_loss

def size_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))
    return (5*sizeloss)/N

def offset_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))
    return (offsetloss)/N
  
def heatmap_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    alpha=2.
    beta=4.

    heatmap_true_rate = K.flatten(y_true[...,:category_n])
    heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])
    heatmap_pred = K.flatten(y_pred[...,:category_n])
    heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))
    return heatloss/N

  
def model_fit_centernet(model,train_list,cv_list,n_epoch,batch_size=32):
    hist = model.fit_generator(
        Datagen_centernet(train_list,batch_size),
        steps_per_epoch = len(train_list) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen_centernet(cv_list,batch_size),
        validation_steps = len(cv_list) // batch_size,
        callbacks = [lr_schedule],#early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 1
    )
    return hist
```


```python
K.clear_session()
model=create_model(input_shape=(input_height,input_width,3),size_detection_mode=False)

def lrs(epoch):
    lr = 0.001
    if epoch >= 20: lr = 0.0002
    return lr

lr_schedule = LearningRateScheduler(lrs)

"""

# EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 60, verbose = 1)
# ModelCheckpoint
weights_dir = '/model_2/'

if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(weights_dir + "val_loss{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,
                                      save_best_only = True, save_weights_only = True, period = 3)
# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)
"""
model.load_weights('final_weights_step1.h5',by_name=True, skip_mismatch=True)

print(model.summary())
```


```python
train_list, cv_list = train_test_split(annotation_list_train_w_split, random_state = 111,test_size = 0.2)#stratified split is better

learning_rate=0.001
n_epoch=30
batch_size=32
model.compile(loss=all_loss, optimizer=Adam(lr=learning_rate), metrics=[heatmap_loss,size_loss,offset_loss])
hist = model_fit_centernet(model,train_list,cv_list,n_epoch,batch_size)

model.save_weights('final_weights_step2.h5')
```


    

```python
pred_in_h=512
pred_in_w=512
pred_out_h=int(pred_in_h/4)
pred_out_w=int(pred_in_w/4)

for i in np.arange(0,1):
  img = np.asarray(Image.open(cv_list[i][0]).resize((pred_in_w,pred_in_h)).convert('RGB'))
  predict=model.predict((img.reshape(1,pred_in_h,pred_in_w,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))
  heatmap=predict[:,:,0]

  fig, axes = plt.subplots(1, 2,figsize=(15,15))
  axes[0].set_axis_off()
  axes[0].imshow(img)
  axes[1].set_axis_off()
  axes[1].imshow(heatmap)
  plt.show()
```


![png](output_14_0.png)



```python
from PIL import Image, ImageDraw

def NMS_all(predicts,category_n,score_thresh,iou_thresh):
  y_c=predicts[...,category_n]+np.arange(pred_out_h).reshape(-1,1)
  x_c=predicts[...,category_n+1]+np.arange(pred_out_w).reshape(1,-1)
  height=predicts[...,category_n+2]*pred_out_h
  width=predicts[...,category_n+3]*pred_out_w

  count=0
  for category in range(category_n):
    predict=predicts[...,category]
    mask=(predict>score_thresh)
    #print("box_num",np.sum(mask))
    if mask.all==False:
      continue
    box_and_score=NMS(predict[mask],y_c[mask],x_c[mask],height[mask],width[mask],iou_thresh)
    box_and_score=np.insert(box_and_score,0,category,axis=1)#category,score,top,left,bottom,right
    if count==0:
      box_and_score_all=box_and_score
    else:
      box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
    count+=1
  score_sort=np.argsort(box_and_score_all[:,1])[::-1]
  box_and_score_all=box_and_score_all[score_sort]
  #print(box_and_score_all)

 
  _,unique_idx=np.unique(box_and_score_all[:,2],return_index=True)
  #print(unique_idx)
  return box_and_score_all[sorted(unique_idx)]
  
def NMS(score,y_c,x_c,height,width,iou_thresh,merge_mode=False):
  if merge_mode:
    score=score
    top=y_c
    left=x_c
    bottom=height
    right=width
  else:
    #flatten
    score=score.reshape(-1)
    y_c=y_c.reshape(-1)
    x_c=x_c.reshape(-1)
    height=height.reshape(-1)
    width=width.reshape(-1)
    size=height*width
    
    
    top=y_c-height/2
    left=x_c-width/2
    bottom=y_c+height/2
    right=x_c+width/2
    
    inside_pic=(top>0)*(left>0)*(bottom<pred_out_h)*(right<pred_out_w)
    outside_pic=len(inside_pic)-np.sum(inside_pic)
    #if outside_pic>0:
    #  print("{} boxes are out of picture".format(outside_pic))
    normal_size=(size<(np.mean(size)*10))*(size>(np.mean(size)/10))
    score=score[inside_pic*normal_size]
    top=top[inside_pic*normal_size]
    left=left[inside_pic*normal_size]
    bottom=bottom[inside_pic*normal_size]
    right=right[inside_pic*normal_size]
  

    

  #sort  
  score_sort=np.argsort(score)[::-1]
  score=score[score_sort]  
  top=top[score_sort]
  left=left[score_sort]
  bottom=bottom[score_sort]
  right=right[score_sort]
  
  area=((bottom-top)*(right-left))
  
  boxes=np.concatenate((score.reshape(-1,1),top.reshape(-1,1),left.reshape(-1,1),bottom.reshape(-1,1),right.reshape(-1,1)),axis=1)
  
  box_idx=np.arange(len(top))
  alive_box=[]
  while len(box_idx)>0:
  
    alive_box.append(box_idx[0])
    
    y1=np.maximum(top[0],top)
    x1=np.maximum(left[0],left)
    y2=np.minimum(bottom[0],bottom)
    x2=np.minimum(right[0],right)
    
    cross_h=np.maximum(0,y2-y1)
    cross_w=np.maximum(0,x2-x1)
    still_alive=(((cross_h*cross_w)/area[0])<iou_thresh)
    if np.sum(still_alive)==len(box_idx):
      print("error")
      print(np.max((cross_h*cross_w)),area[0])
    top=top[still_alive]
    left=left[still_alive]
    bottom=bottom[still_alive]
    right=right[still_alive]
    area=area[still_alive]
    box_idx=box_idx[still_alive]
  return boxes[alive_box]#score,top,left,bottom,right



def draw_rectangle(box_and_score,img,color):
  number_of_rect=np.minimum(500,len(box_and_score))
  
  for i in reversed(list(range(number_of_rect))):
    top, left, bottom, right = box_and_score[i,:]

    
    top = np.floor(top + 0.5).astype('int32')
    left = np.floor(left + 0.5).astype('int32')
    bottom = np.floor(bottom + 0.5).astype('int32')
    right = np.floor(right + 0.5).astype('int32')
    #label = '{} {:.2f}'.format(predicted_class, score)
    #print(label)
    #rectangle=np.array([[left,top],[left,bottom],[right,bottom],[right,top]])

    draw = ImageDraw.Draw(img)
    #label_size = draw.textsize(label)
    #print(label_size)
    
    #if top - label_size[1] >= 0:
    #  text_origin = np.array([left, top - label_size[1]])
    #else:
    #  text_origin = np.array([left, top + 1])
    
    thickness=4
    if color=="red":
      rect_color=(255, 0, 0)
    elif color=="blue":
      rect_color=(0, 0, 255)
    else:
      rect_color=(0, 0, 0)
      
    
    if i==0:
      thickness=4
    for j in range(2*thickness):#薄いから何重にか描く
      draw.rectangle([left + j, top + j, right - j, bottom - j],
                    outline=rect_color)
    #draw.rectangle(
    #            [tuple(text_origin), tuple(text_origin + label_size)],
    #            fill=(0, 0, 255))
    #draw.text(text_origin, label, fill=(0, 0, 0))
    
  del draw
  return img
            
  
def check_iou_score(true_boxes,detected_boxes,iou_thresh):
  iou_all=[]
  for detected_box in detected_boxes:
    y1=np.maximum(detected_box[0],true_boxes[:,0])
    x1=np.maximum(detected_box[1],true_boxes[:,1])
    y2=np.minimum(detected_box[2],true_boxes[:,2])
    x2=np.minimum(detected_box[3],true_boxes[:,3])
    
    cross_section=np.maximum(0,y2-y1)*np.maximum(0,x2-x1)
    all_area=(detected_box[2]-detected_box[0])*(detected_box[3]-detected_box[1])+(true_boxes[:,2]-true_boxes[:,0])*(true_boxes[:,3]-true_boxes[:,1])
    iou=np.max(cross_section/(all_area-cross_section))
    #argmax=np.argmax(cross_section/(all_area-cross_section))
    iou_all.append(iou)
  score=2*np.sum(iou_all)/(len(detected_boxes)+len(true_boxes))
  print("score:{}".format(np.round(score,3)))
  return score

                



for i in np.arange(0,5):
  #print(cv_list[i][2:])
  img=Image.open(cv_list[i][0]).convert("RGB")
  width,height=img.size
  predict=model.predict((np.asarray(img.resize((pred_in_w,pred_in_h))).reshape(1,pred_in_h,pred_in_w,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))
  
  box_and_score=NMS_all(predict,category_n,score_thresh=0.3,iou_thresh=0.4)

  #print("after NMS",len(box_and_score))
  if len(box_and_score)==0:
    continue

  true_boxes=cv_list[i][1][:,1:]#c_x,c_y,width_height
  top=true_boxes[:,1:2]-true_boxes[:,3:4]/2
  left=true_boxes[:,0:1]-true_boxes[:,2:3]/2
  bottom=top+true_boxes[:,3:4]
  right=left+true_boxes[:,2:3]
  true_boxes=np.concatenate((top,left,bottom,right),axis=1)
    
  heatmap=predict[:,:,0]
 
  print_w, print_h = img.size
  #resize predocted box to original size
  box_and_score=box_and_score*[1,1,print_h/pred_out_h,print_w/pred_out_w,print_h/pred_out_h,print_w/pred_out_w]
  check_iou_score(true_boxes,box_and_score[:,2:],iou_thresh=0.5)
  img=draw_rectangle(box_and_score[:,2:],img,"red")
  img=draw_rectangle(true_boxes,img,"blue")
  
  fig, axes = plt.subplots(1, 2,figsize=(15,15))
  #axes[0].set_axis_off()
  axes[0].imshow(img)
  #axes[1].set_axis_off()
  axes[1].imshow(heatmap)#, cmap='gray')
  #axes[2].set_axis_off()
  #axes[2].imshow(heatmap_1)#, cmap='gray')
  plt.show()

```


    
```python
def split_and_detect(model,img,height_split_recommended,width_split_recommended,score_thresh=0.3,iou_thresh=0.4):
  width,height=img.size
  pred_in_w,pred_in_h=512,512
  pred_out_w,pred_out_h=128,128
  category_n=1
  maxlap=0.5
  height_split=int(-(-height_split_recommended//1)+1)
  width_split=int(-(-width_split_recommended//1)+1)
  height_lap=(height_split-height_split_recommended)/(height_split-1)
  height_lap=np.minimum(maxlap,height_lap)
  width_lap=(width_split-width_split_recommended)/(width_split-1)
  width_lap=np.minimum(maxlap,width_lap)

  if height>width:
    crop_size=int((height)/(height_split-(height_split-1)*height_lap))#crop_height and width
    if crop_size>=width:
      crop_size=width
      stride=int((crop_size*height_split-height)/(height_split-1))
      top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]
      left_list=[0]
    else:
      stride=int((crop_size*height_split-height)/(height_split-1))
      top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]
      width_split=-(-width//crop_size)
      stride=int((crop_size*width_split-width)/(width_split-1))
      left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]

  else:
    crop_size=int((width)/(width_split-(width_split-1)*width_lap))#crop_height and width
    if crop_size>=height:
      crop_size=height
      stride=int((crop_size*width_split-width)/(width_split-1))
      left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]
      top_list=[0]
    else:
      stride=int((crop_size*width_split-width)/(width_split-1))
      left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]
      height_split=-(-height//crop_size)
      stride=int((crop_size*height_split-height)/(height_split-1))
      top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]
  
  count=0

  for top_offset in top_list:
    for left_offset in left_list:
      img_crop = img.crop((left_offset, top_offset, left_offset+crop_size, top_offset+crop_size))
      predict=model.predict((np.asarray(img_crop.resize((pred_in_w,pred_in_h))).reshape(1,pred_in_h,pred_in_w,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))
  
      box_and_score=NMS_all(predict,category_n,score_thresh,iou_thresh)#category,score,top,left,bottom,right
      
      #print("after NMS",len(box_and_score))
      if len(box_and_score)==0:
        continue
      #reshape and offset
      box_and_score=box_and_score*[1,1,crop_size/pred_out_h,crop_size/pred_out_w,crop_size/pred_out_h,crop_size/pred_out_w]+np.array([0,0,top_offset,left_offset,top_offset,left_offset])
      
      if count==0:
        box_and_score_all=box_and_score
      else:
        box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
      count+=1
  #print("all_box_num:",len(box_and_score_all))
  #print(box_and_score_all[:10,:],np.min(box_and_score_all[:,2:]))
  if count==0:
    box_and_score_all=[]
  else:
    score=box_and_score_all[:,1]
    y_c=(box_and_score_all[:,2]+box_and_score_all[:,4])/2
    x_c=(box_and_score_all[:,3]+box_and_score_all[:,5])/2
    height=-box_and_score_all[:,2]+box_and_score_all[:,4]
    width=-box_and_score_all[:,3]+box_and_score_all[:,5]
    #print(np.min(height),np.min(width))
    box_and_score_all=NMS(box_and_score_all[:,1],box_and_score_all[:,2],box_and_score_all[:,3],box_and_score_all[:,4],box_and_score_all[:,5],iou_thresh=0.5,merge_mode=True)
  return box_and_score_all


print("test run. 5 image")
all_iou_score=[]
for i in np.arange(0,5):
  img=Image.open(cv_list[i][0]).convert("RGB")
  box_and_score_all=split_and_detect(model,img,cv_list[i][2],cv_list[i][3],score_thresh=0.3,iou_thresh=0.4)
  if len(box_and_score_all)==0:
    print("no box found")
    continue
  true_boxes=cv_list[i][1][:,1:]#c_x,c_y,width_height
  top=true_boxes[:,1:2]-true_boxes[:,3:4]/2
  left=true_boxes[:,0:1]-true_boxes[:,2:3]/2
  bottom=top+true_boxes[:,3:4]
  right=left+true_boxes[:,2:3]
  true_boxes=np.concatenate((top,left,bottom,right),axis=1)

  

 
  print_w, print_h = img.size
  iou_score=check_iou_score(true_boxes,box_and_score_all[:,1:],iou_thresh=0.5)
  all_iou_score.append(iou_score)
  """
  img=draw_rectangle(box_and_score_all[:,1:],img,"red")
  img=draw_rectangle(true_boxes,img,"blue")
  
  fig, axes = plt.subplots(1, 2,figsize=(15,15))
  #axes[0].set_axis_off()
  axes[0].imshow(img)
  #axes[1].set_axis_off()
  axes[1].imshow(heatmap)#, cmap='gray')

  plt.show()
  """
print("average_score:",np.mean(all_iou_score))
```

    test run. 5 image



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-20-e5e324a5c730> in <module>()
         79 for i in np.arange(0,5):
         80   img=Image.open(cv_list[i][0]).convert("RGB")
    ---> 81   box_and_score_all=split_and_detect(model,img,cv_list[i][2],cv_list[i][3],score_thresh=0.3,iou_thresh=0.4)
         82   if len(box_and_score_all)==0:
         83     print("no box found")


    IndexError: list index out of range



```python
from tqdm import tqdm
count=0
crop_dir="/crop_letter/"
if os.path.exists(crop_dir) == False:os.mkdir(crop_dir)

train_input=[]
pic_count=0
for ann_pic in tqdm(annotation_list_train):
  pic_count+=1
  with Image.open(ann_pic[0]) as img:
    for ann in ann_pic[1]:#cat,center_x,center_y,width,height for each picture
      cat=ann[0]
      c_x=ann[1]
      c_y=ann[2]
      width=ann[3]
      height=ann[4]
      save_dir=crop_dir+str(count)+".jpg"
      img.crop((int(c_x-width/2),int(c_y-height/2),int(c_x+width/2),int(c_y+height/2))).save(save_dir)
      train_input.append([save_dir,cat])
      count+=1
                 
```


   


```python
df_unicode_translation=pd.read_csv("./data/unicode_translation.csv")
unicode=df_unicode_translation["Unicode"].values
char=df_unicode_translation["char"].values
dict_translation={unicode[i]:char[i] for i in range(len(unicode))}

i=0
img = np.asarray(Image.open(train_input[i][0]).resize((32,32)).convert('RGB'))
name = dict_translation[inv_dict_cat[str(train_input[i][1])]]
print(name)
plt.imshow(img)
plt.show()
  
```


    

```python
input_height,input_width=32,32

def Datagen_for_classification(filenames, batch_size, is_train=True,random_crop=True):
  x=[]
  y=[]
  
  count=0

  while True:
    for i in range(len(filenames)):
      if random_crop:
        crop_ratio=np.random.uniform(0.8,1)
      else:
        crop_ratio=1
      with Image.open(filenames[i][0]) as f:
        
        #random crop
        if random_crop and is_train:
          pic_width,pic_height=f.size
          f=np.asarray(f.convert('RGB'),dtype=np.uint8)
          top_offset=np.random.randint(0,pic_height-int(crop_ratio*pic_height))
          left_offset=np.random.randint(0,pic_width-int(crop_ratio*pic_width))
          bottom_offset=top_offset+int(crop_ratio*pic_height)
          right_offset=left_offset+int(crop_ratio*pic_width)
          f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))
        else:
          f=f.resize((input_width, input_height))
          f=np.asarray(f.convert('RGB'),dtype=np.uint8)          
        x.append(f)
      
        y.append(int(filenames[i][1]))
      count+=1
      if count==batch_size:
        x=np.array(x, dtype=np.float32)
        y=np.identity(len(dict_cat))[y].astype(np.float32)

        inputs=x/255
        targets=y       
        x=[]
        y=[]
        count=0
        yield inputs, targets
        
def create_classification_model(input_shape, n_category):
    input_layer = Input(input_shape)#32
    x=cbr(input_layer,64,3,1)
    x=resblock(x,64)
    x=resblock(x,64)
    x=cbr(input_layer,128,3,2)#16
    x=resblock(x,128)
    x=resblock(x,128)
    x=cbr(input_layer,256,3,2)#8
    x=resblock(x,256)
    x=resblock(x,256)
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.2)(x)
    out=Dense(n_category,activation="softmax")(x)#sigmoid???catcrossていぎ
    
    classification_model=Model(input_layer, out)
    
    return classification_model
      
def model_fit_classification(model,train_list,cv_list,n_epoch,batch_size=32):
    hist = model.fit_generator(
        Datagen_for_classification(train_list,batch_size, is_train=True,random_crop=True),
        steps_per_epoch = len(train_list) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen_for_classification(cv_list,batch_size, is_train=False,random_crop=False),
        validation_steps = len(cv_list) // batch_size,
        #callbacks = [early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 1
    )
    return hist
```


```python
K.clear_session()
input_height,input_width=32,32
model=create_classification_model(input_shape=(input_height,input_width,3),n_category=len(dict_cat))
"""

# EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 60, verbose = 1)
# ModelCheckpoint
weights_dir = './model_3/'

if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(weights_dir + "val_loss{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,
                                      save_best_only = True, save_weights_only = True, period = 1)
# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)
"""

print(model.summary())
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 16, 16, 256)  7168        input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 16, 16, 256)  1024        conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 16, 16, 256)  1024        conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_12[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 16, 16, 256)  1024        conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, 16, 16, 256)  0           leaky_re_lu_13[0][0]             
                                                                     leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 16, 16, 256)  590080      add_5[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 16, 16, 256)  1024        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 16, 16, 256)  590080      leaky_re_lu_14[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 16, 16, 256)  1024        conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)      (None, 16, 16, 256)  0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, 16, 16, 256)  0           leaky_re_lu_15[0][0]             
                                                                     add_5[0][0]                      
    __________________________________________________________________________________________________
    global_average_pooling2d_1 (Glo (None, 256)          0           add_6[0][0]                      
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 256)          0           global_average_pooling2d_1[0][0] 
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 4212)         1082484     dropout_1[0][0]                  
    ==================================================================================================
    Total params: 3,455,092
    Trainable params: 3,452,532
    Non-trainable params: 2,560
    __________________________________________________________________________________________________
    None



```python
train_list, cv_list = train_test_split(train_input, random_state = 111,test_size = 0.2)
learning_rate=0.005
n_epoch=10
batch_size=64

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate),metrics=["accuracy"])
hist = model_fit_classification(model,train_list,cv_list,n_epoch,batch_size)

model.save_weights('final_weights_step3.h5')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-25-73f6432eaa0e> in <module>()
    ----> 1 train_list, cv_list = train_test_split(train_input, random_state = 111,test_size = 0.2)
          2 learning_rate=0.005
          3 n_epoch=10
          4 batch_size=64
          5 


    NameError: name 'train_input' is not defined



```python
for i in range(3):
  img = np.asarray(Image.open(train_input[i][0]).resize((32,32)).convert('RGB'))
  predict=np.argmax(model.predict(img.reshape(1,32,32,3)/255),axis=1)[0]
  name = dict_translation[inv_dict_cat[str(predict)]]
  print(name)
  plt.imshow(img)
  plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-26-f24a1b351731> in <module>()
          1 for i in range(3):
    ----> 2   img = np.asarray(Image.open(train_input[i][0]).resize((32,32)).convert('RGB'))
          3   predict=np.argmax(model.predict(img.reshape(1,32,32,3)/255),axis=1)[0]
          4   name = dict_translation[inv_dict_cat[str(predict)]]
          5   print(name)


    NameError: name 'train_input' is not defined



```python
from tqdm import tqdm

K.clear_session()
print("loading models...")
model_1=create_model(input_shape=(512,512,3),size_detection_mode=True)
#model_1.load_weights('final_weights_step1.h5')
model_1.load_weights('./results/final_weights_step1.hdf5')

model_2=create_model(input_shape=(512,512,3),size_detection_mode=False)
model_2.load_weights('./results/final_weights_step2.h5')

model_3=create_classification_model(input_shape=(32,32,3),n_category=len(dict_cat))
model_3.load_weights('./results/final_weights_step3.h5')


def pipeline(i,print_img=False):
  # model1: determine how to split image
  if print_img: print("model 1")
  img = np.asarray(Image.open(id_test[i]).resize((512,512)).convert('RGB'))
  predicted_size=model_1.predict(img.reshape(1,512,512,3)/255)
  detect_num_h=aspect_ratio_pic_all_test[i]*np.exp(-predicted_size/2)
  detect_num_w=detect_num_h/aspect_ratio_pic_all_test[i]
  h_split_recommend=np.maximum(1,detect_num_h/base_detect_num_h)
  w_split_recommend=np.maximum(1,detect_num_w/base_detect_num_w)
  if print_img: print("recommended split_h:{}, split_w:{}".format(h_split_recommend,w_split_recommend))

  # model2: detection
  if print_img: print("model 2")
  img=Image.open(id_test[i]).convert("RGB")
  box_and_score_all=split_and_detect(model_2,img,h_split_recommend,w_split_recommend,score_thresh=0.3,iou_thresh=0.4)#output:score,top,left,bottom,right
  if print_img: print("find {} boxes".format(len(box_and_score_all)))
  print_w, print_h = img.size
  if (len(box_and_score_all)>0) and print_img: 
    img=draw_rectangle(box_and_score_all[:,1:],img,"red")
    plt.imshow(img)
    plt.show()

  # model3: classification
  count=0
  if (len(box_and_score_all)>0):
    for box in box_and_score_all[:,1:]:
      top,left,bottom,right=box
      img_letter=img.crop((int(left),int(top),int(right),int(bottom))).resize((32,32))#大き目のピクセルのがいいか？
      predict=(model_3.predict(np.asarray(img_letter).reshape(1,32,32,3)/255))
      predict=np.argmax(predict,axis=1)[0]
      code=inv_dict_cat[str(predict)]
      c_x=int((left+right)/2)
      c_y=int((top+bottom)/2)
      if count==0:
        ans=code+" "+str(c_x)+" "+str(c_y)
      else:
        ans=ans+" "+code+" "+str(c_x)+" "+str(c_y)
      count+=1
  else:
    ans=""
  return ans

_=pipeline(0,print_img=True)

#I'm sorry. Not nice coding. Time consuming.
for i in tqdm(range(len(id_test))):
  ans=pipeline(i,print_img=False)
  df_submission.set_value(i, 'labels', ans)
      
df_submission.to_csv("./results/submission.csv",index=False)

```

    loading models...
    model 1
    recommended split_h:[[1.9653429]], split_w:[[1.2797313]]
    model 2
    find 365 boxes



![png](output_23_1.png)


    
      0%|          | 0/4150 [00:00<?, ?it/s][A/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:63: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
    
      0%|          | 1/4150 [00:04<5:41:04,  4.93s/it][A
      0%|          | 2/4150 [00:08<4:59:41,  4.33s/it][A
      0%|          | 3/4150 [00:12<4:47:50,  4.16s/it][A
      0%|          | 4/4150 [00:17<5:02:34,  4.38s/it][A
      0%|          | 5/4150 [00:22<5:06:19,  4.43s/it][A
      0%|          | 6/4150 [00:26<5:07:25,  4.45s/it][A
      0%|          | 7/4150 [00:30<5:03:13,  4.39s/it][A
      0%|          | 8/4150 [00:37<5:24:02,  4.69s/it][A
      0%|          | 9/4150 [00:41<5:19:17,  4.63s/it][A
      0%|          | 10/4150 [00:46<5:21:59,  4.67s/it][A
      0%|          | 11/4150 [00:51<5:23:30,  4.69s/it][A/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:3118: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py:85: RuntimeWarning: invalid value encountered in true_divide
      ret = ret.dtype.type(ret / rcount)
    
      0%|          | 12/4150 [00:52<5:03:54,  4.41s/it][A
      0%|          | 13/4150 [00:57<5:03:43,  4.41s/it][A
      0%|          | 14/4150 [01:02<5:06:41,  4.45s/it][A


   