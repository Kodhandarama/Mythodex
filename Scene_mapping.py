# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:39:42 2019

@author: Devika
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 07:36:55 2019

@author: Devika
"""
import os
def predict_caption (input_list):
    elements=[]
    similarities=dict()#Has all the similaritie scores between input and individual files
    high=["Shabari","Vishwamitra","Ahalya","Eagles","Deer","Kumbhakarna"]
    low=["forests","Beach",'Mountain']
    key=["Rama","Ravana"]
    intermediate=["Fire"]
    path=r"D:\\ML\\Scenes"
    for filename in enumerate(os.listdir(path)):
        new_path=path+"\\"+filename[1]
        f=open(new_path,"r")
        lines=f.readlines()
        classes=lines[len(lines)-4:len(lines)]#To obtain the last 4 lines which contain the classes
        #print(classes)
        for category in classes:#eg:- Characters/Landscapes
            for j in range(0,len(category)):
                if category[j]==":":
                    index=j
            if category[len(category)-1]=="\n":
                category=category[index+1:len(category)-1]
            else:
                category=category[index+1:len(category)]
            temp=category.split(",")
            for i in temp:
                if i!="":
                    elements.append(i)
        #Similarity calculation
        numerator=0
        denominator=1
        count=0
        """for i in key:
            if i in elements:
                count+=1
        if count==2:
            elements.append("Bow_and_arrow")
            numerator+=4
            #print("MODIFIED")"""
            
        trial=["Sita","Fire"]
        final=["Ravana","Bow_and_arrow","Rama"]
        Lanka=["Fire","Mace"]
        specifics=["The_final_battle.txt","Trial_by_fire.txt","Fire_in_lanka.txt"]
        if final[0] in input_list and final[1] in input_list:
            input_list=trial
            input_list.extend(["Bow_and_arrow","Horse"])
            numerator+=100
            print("BATTLE")
            similarity=numerator/denominator
            similarities["The_final_battle.txt"]=similarity 
            break
            count+=1
        elif trial[0] in input_list and trial[1] in input_list:
            input_list=trial
            input_list.extend(["Rama","Lakshmana"])
            numerator+=100
            #print("TRIAL")
            count+=1
            similarity=numerator/denominator
            similarities["Trial_by_fire.txt"]=similarity
            break
        
        elif Lanka[0] in input_list and Lanka[1] in input_list:
            input_list=trial
            input_list.extend(["Hanuman","Ravana","Monkeys"])
            numerator+=100
            #print("LANKA")
            count+=1
            similarity=numerator/denominator
            similarities["Fire_in_lanka.txt"]=similarity
            break
        
        if count==0:
            for i in input_list:
                denominator+=1.0
                if i in elements:
                    numerator+=1.0
                    if i in high:
                        numerator+=3
                    if i in intermediate:
                        numerator+=2
                    if i in low:
                        numerator+=1
            for i in elements:
                if i in elements and i not in input_list:
                    denominator+=0.7
                    if i in high:
                        denominator+=2
        count=0
        
        
        
        
        
        similarity=numerator/denominator
        similarities[filename[1]]=similarity 
            
        elements=[]
                   
                
        f.close()
    ranked_list=sorted(similarities, key=lambda x: similarities[x])[::-1]#Sorting
    print("The scene is most likely:")
    print(ranked_list[:3])
    path+="\\"+ranked_list[0]
    #print(path)
    f=open(path,"r")
    text=f.read()
    for i in range(0,len(text)):
        if text[i]=="*":
            index=i
    text=text[0:index]
    print(text)
    f.close()
    return ranked_list[:3]