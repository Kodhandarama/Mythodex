# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:01:07 2019

@author: 91948
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:16:19 2019

@author: 91948
"""

import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
import random
import cv2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.keras.models import save_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.applications import InceptionResNetV2,VGG16,MobileNet,Xception
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Activation, Flatten
import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import plot_model
from tensorflow.keras.models import save_model





CATEGORY = dict()
if not 'PATH' in globals():
    PATH = dict()

PATH["Train_Weapons"] = 'D:\\New\\Weapons'   ## weapons.model
PATH["Train_Actions2"] = 'D:\\New\\Actions2' ## actions2.model
PATH["Train_Animals"] = 'D:\\New\\Animals' ## animals.model
PATH["Train_Charecters"] = 'D:\\New\\Charecters' ## Charecters.model
PATH["Train_Scene"] = 'D:\\New\\Scene' ## Charecters.model
PATH["Train_Gender"] = 'D:\\New\\Gender' ## gender.model

'''
PATH["Test"] = 'D:\DataSets\Gods\Hanuman_train'
PATH["checkpoint"] = 'D:\CAP'
PATH["train_captions"] = "D:\DataSets\Captions.txt"
PATH["test_captions"] = "D:\DataSets\Test_Caption.txt"
'''


def get_imgs (path):
    result = list()
    for file in os.listdir(path):
        if (file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')):
            result.append(file)
        else :
            print("File not compatible (type error)",file)
    return result

#images_train = get_imgs(PATH['Train'])

def load_image(path, size=None,show = False):
    try:
        img = Image.open(path)
        if show :
            plt.imshow(img)
            plt.show()
    except Exception as e:
        print(e)
        print("File not found")
        pass
    if not size is None:
        try :
            img = img.resize(size=size, resample=Image.LANCZOS)
        except Exception as e:
            print('Erorr:',e)
            try:
                print('using cv2')
                img = cv2.imread(path)
                img = cv2.resize(img(size,size))
                return img
            except Exception as e:
                print('idk',e)
    img = np.array(img)
    img = img / 255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img



if not 'image_model' in globals():
    image_model = Xception(include_top=True, weights='imagenet')
    #image_model.summary()
    transfer_layer = image_model.get_layer('avg_pool')
    image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

    img_size = K.int_shape(image_model.input)[1:3]
    print('img size:',img_size)



    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    print('Output tensor',transfer_values_size)



#image_model_transfer.summary()


def process_images(data_dir, filenames, batch_size=1024):
    error = 0
    num_images = len(filenames)
    #print(num_images)
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

   
    start_index = 0

    while start_index < num_images:
        end_index = start_index + batch_size

        
        if end_index > num_images:
            end_index = num_images

        current_batch_size = end_index - start_index

        for i, filename in enumerate(filenames[start_index:end_index]):
            error+=1
            path = os.path.join(data_dir, filename)
            #print(path)
            print(error)
            try:
                img = load_image(path, size=img_size)
                image_batch[i] = img
            except Exception as e:
                print('File error skiped',e)
                continue

        transfer_values_batch = \
            image_model_transfer.predict(image_batch[0:current_batch_size])

        transfer_values[start_index:end_index] = \
            transfer_values_batch[0:current_batch_size]

        start_index = end_index

    #print()
    #print("GAY",len(transfer_values))
    return transfer_values





def plot_scatter(x,y,number_of_categories):
    # Create a color-map with a different color for each class.
    
    from sklearn.decomposition import PCA
    
    pcs = PCA(n_components = 2)
    t_new = pcs.fit_transform(x)
    
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, number_of_categories))
    y_new = np.asarray(y,dtype = np.int16)
    # Get the color for each sample.
    colors = cmap[y_new]

    # Extract the x- and y-values.
    x_p = t_new[:, 0]
    y_p = t_new[:, 1]
    plt.scatter(x_p, y_p, color=colors)
    plt.show()





def predict (path , show = True):
    print('Original Image:')
    img = load_image(path,img_size,show = show)
    img_batch = np.expand_dims(img,axis=0)
    transfer_values = image_model_transfer.predict(img_batch)
    prediction = model.predict([transfer_values])
    pre_list = list(prediction[0])
    pre_index = pre_list.index(max(pre_list))
    if 0 and max(pre_list) < 0.50:
        print(pre_index)
        return None
    print(pre_list)
    print(CATEGORY)
    if show:
        print(CATEGORY[pre_index])
    return pre_index



def cache(cache_path, fn, *args, **kwargs):
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.
        #print('big no no')
        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)
        print("- Data saved to cache-file: " + cache_path)

    return obj



def process_images_cache(path):
    number_of_categories = 0
    train_data = list()
    sex = list()
    categories = os.listdir(path)
    for category in categories:
        if not category.endswith('.txt'):
            sex.append(category)
    categories = sex
    for category in categories: 
        number_of_categories+=1
        class_num = categories.index(category)
        CATEGORY[class_num] = category
        path_category = os.path.join(path,category)
        cache_path = os.path.join(path_category, str(category) +".pkl")
        images_train = get_imgs(path_category)
        print(category,len(images_train))
        transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=path_category,
                            filenames=images_train)
        for transfer_value in transfer_values:
            train_data.append([transfer_value,class_num])
    return train_data,number_of_categories



def train_and_save(cat):
    train_data,number_of_categories = process_images_cache(PATH[cat])
    random.shuffle(train_data)
    save_path = cat.split('_')[-1:][0] + ".model"
    save_path = os.path.join('D:\\ML\\Models',save_path)

    x = list()
    y = list() 

    for i,j in train_data:
        x.append(i)
        y.append(j)
        
    x = np.asarray(x,dtype = np.float16)
    y = np.asarray(y,dtype = np.float16)
        #print(len(transfer_values[0]

    #input_size = Input(shape = ( len(transfer_values[0]), ))
    #output_size = Dense(16)(input_size)

    #model = Model(inputs=input_size,outputs = output_size)

    #model.compile(optimizer = "") 

    print('number of categories:',number_of_categories)


    model = Sequential()
    model.add(Flatten())
    model.add(Dense(2048 ,activation='sigmoid'))
    model.add(Dense(512,activation='sigmoid'))
    model.add(Dense(512,activation='sigmoid'))
    model.add(Dense(number_of_categories,activation='sigmoid'))
    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    #model.summary()
    #model.plot_model('model.jpg')
    print('Compiled')


    sex = model.fit(x[:int(len(x)*0.75)],y[:int(len(y)*0.75)],epochs=25,shuffle = True)
    evalsex = model.evaluate(x[int(len(x)*0.75):],y[int(len(y)*0.75):])

    #model.summary()
    
    # use sex / evalsex for acuu checking
    plot_scatter(x,y,number_of_categories)
    plt.plot(sex.history['loss'])
    #plt.plot(evalsex.history['loss'])
    plt.show()
    plt.plot(sex.history['sparse_categorical_accuracy'])
    plt.plot(evalsex[0])
    #plt.plot(evalsex.history['sparse_categorical_accuracy'])
    plt.show()
    if (input('Wanna Save [Y/N]') == 'Y'):
        save_model(model,save_path)
    return train_data
def show_transfer_value (train_data):
    for img in train_data:
        img = img.reshape((64,32))
        plt.imshow(img)
        plt.show()
        
  

def train ():
    num_models = len(PATH)
    i=0
    for path in PATH:
        train_and_save(path)
        i+=1
        print(i,' out of ',num_models)
    print("All models are up to date!!")  


    
'''
train_and_save('Train_Animals')      
train_and_save('Train_Actions2')      
train_and_save('Train_Weapons')      
train_and_save('Train_Scene')
train_and_save('Train_Animals')    
train_and_save('Train_Charecters')    
'''


#train_and_save('Train_Charecters')
#train_and_save('Train_Gender') 
      





'''



'''



