# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:48:41 2019

@author: 91948
"""

import numpy as np
import os
import matplotlib as plt
import cv2 
import random
import pickle




print("Versions:",np.__version__,plt.__version__,cv2.__version__)


training_data=[]

Datadir = "D:\DataSets\Gods"
Category = ["Shiva","Ganesha"]

img_size=int(input("Enter the img size:"))
def get_data():
    for category in Category:
        path=os.path.join(Datadir,category)
        class_num = Category.index(category)
        print(path)
        for d in os.listdir(path):    
            try:
                img=cv2.imread(os.path.join(path,d),cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img,(img_size,img_size))
                try:
                    plt.imshow(img,cmap="gray")
                    plt.show()
                except Exception as e:
                    pass
                training_data.append([img,class_num])
            except Exception as e:
                print("Found a broken image",e)
                
                
get_data()
print("Len of training data:",len(training_data))
random.shuffle(training_data)

x=[]
y=[]

for img,label in training_data:
    x.append(img)
    y.append(label)
    
    
x=np.array(x).reshape(-1,img_size,img_size,1)


name_x=input("Name the data file (.pickle):")
name_y=input("Name the label file (.pickle):")

pickle_out=open(name_x,"wb")
pickle.dump(x,pickle_out)
pickle_out.close()



pickle_out=open(name_y,"wb")
pickle.dump(y,pickle_out)
pickle_out.close()

print("All Done!!!!")




