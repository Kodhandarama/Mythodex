# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:37:56 2019

@author: 91948
"""
import os
import collections
from transfervalues_new import train
from Scene_mapping import predict_caption

'''   Update the Models and set up setup   '''

if input('Retrain and Update Models? [Y/N]') == 'Y':
    train()
else:
    print('Models are not upto date')
from Text_Ramayana import finalize,remove_duplicates
from predictor_multi import prection,split_imgs,load_image



#feed_back = input('Do you want to impliment Feedback [Y\N]')
feed_back = input('Feed back [Y/N]')
feed_back = True if feed_back == 'Y' else False


numarator = 0
denominator = 0


#'D:\\DataSets\\Scenes\\10.jpg'

def predict(path,feed_back = False):
    global numarator
    global denominator
    '''            Nueral Network Model           '''
    result = prection(path,feed_back = feed_back)
    #print(prediction)
    char_gender = list()
    char_prob = list()
    prediction_strict = list ()
    prediction_linient = list()
   
    '''                  For UI use                           '''
    pics = list()
    img = load_image(path)
    img_new = img * 255.0
    rois = split_imgs(path)
    pics = list(map(lambda x:x*255.0,rois))
    pics.insert(0,img_new)
    result_text = dict()
    
    for sub_pre in result:
        try:
            j = [int(s) for s in sub_pre.split('_') if s.isdigit()]
            if len(j) == 0:
                j = -1
            else:
                j = j[0]
        except :
            pass
        if j not in result_text.keys():
            result_text[j] = list()
        result_text[j].append((sub_pre,result[sub_pre]))
    
    UI = list()
    text_part = list()
    for key in result_text:
        text_part.append(result_text[key])
    UI = list(zip(pics,text_part)) ####  use UI for UI
    
    
    
    ''' processing the out put of the nueral network '''
    
    for sub_pre in result:
        if 'Charecter' in sub_pre:
            temp = list()
            temp.append(result[sub_pre][0])
            char_prob.append((result[sub_pre][0],result[sub_pre][1]))
        if 'Gender' in sub_pre:
            char_gender.append((temp[0],result[sub_pre][0]))
        if not('Charecter' in sub_pre or 'Gender' in sub_pre):
            if result[sub_pre][1]:
                prediction_strict.append(result[sub_pre][0])
            prediction_linient.append(result[sub_pre][0])
            
    predict_scene_at_a_glance = result['Actions2']
    #print('all_predicions_lineient',all_predicions_lineient)
    
    '''                 Graph Model              '''
    
    char_no_duplicates = remove_duplicates(char_gender)
    char_no_duplicates_strict = list()
    for char,prob in char_prob:
        if [item for item, count in collections.Counter(char_prob).items() if count == 1]:
            char_no_duplicates_strict.append((char,prob))
    #print(char_no_duplicates)
    if not len(char_no_duplicates[0][0]) == 1:
        char_no_duplicates = list(map(lambda x:x[0],char_no_duplicates))
    char_list = list(map(lambda x:x[0],char_no_duplicates_strict)) 
    for char in char_no_duplicates:
        if not char in char_list:
           char_no_duplicates_strict.append((char,True)) 
    char_no_duplicates_strict = finalize(char_no_duplicates_strict)
    char_no_duplicates_strict = list(set(map(lambda x : x[0],char_no_duplicates_strict)))
    #print('sending to scene mapping',set(char_no_duplicates))
    #print('Char no duplicate strict',char_no_duplicates_strict)
    for char in char_no_duplicates:
        prediction_linient.append(char)
    for char in char_no_duplicates_strict:
        prediction_strict.append(char)
    
    
    
    ''' Find all the Captions '''
    print(prediction_linient)
    print(prediction_strict)
    final_solution = predict_caption(prediction_linient)
    final_solution = predict_caption(prediction_strict)
    UI.append(final_solution)
    return UI



"""

for file in os.listdir('E:\\Test_Images')[15:35]:
    try:
        predict(os.path.join('E:\\Test_Images',file),feed_back = feed_back)
    except Exception as e:
        print("Internal file error",e)




"""
'''



for file in os.listdir('D:\\DataSets\\Scenes')[:15]:
    try:
        predict(os.path.join('D:\\DataSets\\Scenes',file),feed_back = feed_back)
    except Exception as e:
        print("Internal file error",e)

'''
#print(numarator/denominator)





