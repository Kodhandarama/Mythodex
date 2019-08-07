# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:14:39 2019

@author: 91948
"""

import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
import time
import cv2
from tensorflow.python.keras import backend as K
from imageai.Detection import ObjectDetection
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.applications import InceptionResNetV2,VGG16,MobileNet,Xception
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Activation, Flatten
import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import plot_model
from tensorflow.keras.models import load_model,save_model
import string
import random
letters = string.ascii_letters
feed_back = False

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
                img = cv2.resize(img,(size,size))
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


    print('Transfer Modle has been made')
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    print('Output tensor',transfer_values_size)     



#image_model_transfer.summary()



def cache(cache_path, fn, *args, **kwargs):
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)
        print("- Data loaded from cache-file: " + cache_path)
    else:
        obj = fn(*args, **kwargs)
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)
        print("- Data saved to cache-file: " + cache_path)

    return obj


PATH = dict()
PATH['Update'] = 'E:\\Update\\New'
PATH['Categories'] = 'D:\\New'
PATH['Models'] = "D:\\ML\\Models"
def get_cats(path):
    CATEGORIES = dict()
    for obj in os.listdir(path):
        CATEGORIES[obj] = list()
        for new_obj in os.listdir(os.path.join(path,obj)):
            CATEGORIES[obj].append(new_obj)
    return CATEGORIES
       
CATEGORIES = get_cats(PATH["Categories"])
 
def load_models (path):
    result = dict()
    for cat in CATEGORIES.keys():
        result[cat] = load_model(os.path.join(path,cat+'.model')) 
    return result

models = load_models(PATH["Models"])
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("D:\\ML\\Models\\resnet50_coco_best_v2.0.1.h5")
detector.loadModel()




def split_imgs(path,show = False):
    detections,_ = detector.detectObjectsFromImage(input_image=path, output_image_path='.\\', extract_detected_objects=True)
    img = load_image(path,None,show = show)
    rois = list()
    rois_img = list()
    for detection in detections:
        rois.append(detection['box_points'])
        x = detection['box_points'][0]
        y = detection['box_points'][1]
        width = abs(detection['box_points'][2] - x)
        height = abs(detection['box_points'][3]  - y)
        rois_img.append(img[y:y+height , x:x+width, :])
    if show :
        for roi in rois:
            plt.imshow(roi)
            plt.show() 
    return rois_img


    

def predict_all (path = None , show = True, roi = None):
    result = dict()
    print('Original Image:')
    if path :
        img = load_image(path,img_size,show = show)
    elif not type(roi) == 'NoneType' :
        img = roi
        try :
            img = img.resize(size=img_size, resample=Image.LANCZOS)
        except Exception as e:
            print('PIL failed ',e)
            try:
                print('retrying using cv2')
                img = cv2.resize(img,(img_size[0],img_size[0]))
            except Exception as e:
                print(e)
                return False
        if show:
            plt.imshow(roi)
            plt.show()
    else :
        print("Error")
        return False
    img_batch = np.expand_dims(img,axis=0)
    transfer_values = image_model_transfer.predict(img_batch)
    print("Transfer Values:",transfer_values.shape)
    for cat in CATEGORIES.keys():
        model = models[cat]
        pre = list(model.predict(transfer_values)[0])
        pre_idx = pre.index(max(pre))
        if max(pre) < 0.15:
            print(cat,' prediction : Not Sure but i think:',CATEGORIES[cat][pre_idx], sep = ' ')
            result[cat] = (CATEGORIES[cat][pre_idx],False)
            continue
        #print(CATEGORIES[cat],pre,sep = '\n')
        print(cat,' prediction :', CATEGORIES[cat][pre_idx])
        result[cat] = (CATEGORIES[cat][pre_idx],True)
    return transfer_values,result



def prection (path,show= True,feed_back = False):
    print("Entire Image:")
    start_time = time.time()
    transfer_values,result = predict_all(path,show = show)
    #print('For whole img',result)
    rois = split_imgs(path, show = not show)# can improve
    print("For each region of interest")
    i =0
    for roi in rois:
       if roi.shape[0] < 25 or roi.shape[1] < 25:
           #print('roi too small')
           continue
       if show:
           plt.imshow(roi)
           plt.show() 
       try:
           img = roi.resize(size=img_size, resample=Image.LANCZOS)
       except:
           try:
               img = cv2.resize(roi,(img_size[0],img_size[0]))
           except Exception as e:
               print("Image failed",e)
               return False
       img_batch = np.expand_dims(img,axis=0)
       transfer_values = image_model_transfer.predict(img_batch)


       pre = list(models['Animals'].predict(transfer_values)[0])
       pre_idx = pre.index(max(pre))
       if max(pre) < 0.085:
           print('Animals prediction : Not Sure but i think:',CATEGORIES['Animals'][pre_idx], sep = ' ')
           result['Animals_roi_'+str(i)] = (CATEGORIES['Animals'][pre_idx],False)
       else:
           print('Animals prediction :', CATEGORIES['Animals'][pre_idx]) 
           result['Animals_roi_'+str(i)] = (CATEGORIES['Animals'][pre_idx],True)
       if feed_back:
           if (input("Right or wrong:") == 'N') :
               char = input("Enter the animals:")
               path_new = os.path.join(PATH['Update'],'Animals')
               path_new = os.path.join(path_new,char)
               img_name = ''.join(random.choice(letters) for i in range(10))+'.jpg'
               new_roi = roi * 255.0
               try :
                   cv2.imwrite(os.path.join(path_new,img_name),new_roi)
               except Exception as e:
                   print("Failed Savign",e)
       if not (CATEGORIES['Animals'][pre_idx] == 'Humans' or CATEGORIES['Animals'][pre_idx] == 'Monkeys' or CATEGORIES['Animals'][pre_idx] =='Eagles'):
           continue
       
       pre = list(models['Charecters'].predict(transfer_values)[0])
       pre_idx = pre.index(max(pre))
       if max(pre) < 0.085:
           print('Charecter prediction : Not Sure but i think:',CATEGORIES['Charecters'][pre_idx], sep = ' ')
           result['Charecters_roi_'+str(i)] = (CATEGORIES['Charecters'][pre_idx],False)
       else:
           print('Charecter prediction :', CATEGORIES['Charecters'][pre_idx]) 
           result['Charecters_roi_'+str(i)] = (CATEGORIES['Charecters'][pre_idx],True)
       
       if feed_back:  
           if (input("Right or wrong:") == 'N' and feed_back):
               char = input("Enter the charecter:")
               path_new = os.path.join(PATH['Update'],'Charecters')
               path_new = os.path.join(path_new,char)
               img_name = ''.join(random.choice(letters) for i in range(10))+'.jpg'
               new_roi = roi * 255.0
               try :
                   cv2.imwrite(os.path.join(path_new,img_name),new_roi)
               except Exception as e:
                   print('Failed Saving',e)
       
        
       
        
       pre = list(models['Gender'].predict(transfer_values)[0])
       pre_idx = pre.index(max(pre))
       if max(pre) < 0.085:
           print('gender prediction : Not Sure but i think:',CATEGORIES['Gender'][pre_idx], sep = ' ')
           result['Gender_roi_'+str(i)] = (CATEGORIES['Gender'][pre_idx],False)
       else:
           print('gender prediction :', CATEGORIES['Gender'][pre_idx]) 
           result['Gender_roi_'+str(i)] = (CATEGORIES['Gender'][pre_idx],False)
    
       if feed_back:  
           if (input("Right or wrong:") == 'N' and feed_back):
               char = input("Enter the charecter:")
               path_new = os.path.join(PATH['Update'],'Gender')
               path_new = os.path.join(path_new,char)
               img_name = ''.join(random.choice(letters) for i in range(10))+'.jpg'
               new_roi = roi * 255.0
               try :
                   cv2.imwrite(os.path.join(path_new,img_name),new_roi)
               except Exception as e:
                   print('Failed Saving',e)
       
       i=i+1
    print('Elapsed Time:',time.time() - start_time)

    return result   

#prection('D:\\DataSets\\Scenes\\10.jpg')
#prection('D:\\Yolo\\hello.jpg') 
'''
prection('D:\\DataSets\\Scenes\\5.jpg')
for file in os.listdir('D:\\DataSets\\Scenes')[35:55]:
    path = os.path.join('D:\\DataSets\\Scenes',file)
    prection(path)
    print('\n\n')
    
''' 
    
    
    
    
'''
def predict (path , show = True):
    print('Original Image:')
    img = load_image(path,img_size,show = show)
    img_batch = np.expand_dims(img,axis=0)
    transfer_values = image_model_transfer.predict(img_batch)
    

    
''' 