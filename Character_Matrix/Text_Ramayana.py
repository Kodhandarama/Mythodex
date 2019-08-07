# -*- coding: cp1252 -*-
#TO EXTRACT NAMES OF CHARACTERS(WHICH ARE IN BOLD) FROM .docx file
from docx import *
import time
import numpy as np
from scipy.stats import zscore
import itertools

start = time.time()
document = Document("Ramayana.docx")
bolds=[]
for para in document.paragraphs:
    for run in para.runs:
        if run.bold :
            word=run.text
            if word!="":
                if word[0].isupper():
                    
                    bolds.append(word)
#print(*bolds,sep="\n")
    
sno=1
count=0  
character=dict()
unique=set(bolds)
#print(unique)   
for i in unique:
    for j in bolds:
        if i==j:
            count+=1
    if i!="":
        character[i]=count
    count=0;
    
ranked_list=sorted(character, key=lambda x: character[x])[::-1]
for i in ranked_list:
    #print(sno,")",i,character[i])
    sno+=1



#print(ranked_list[0:14])

#TO NAVIGATE WORDS AND SENTENCES IN .TXT FILE WHICH HAS THE SAME CONTENTS AS THE .DOCX FILE
f=open("Ramayana.txt","r")
text=f.read()
sentences=text.split(".")
relation=[]#contains a list of tuples of characters appearing together
rel=[]#contains two characters that appear together
c=0
for sentence in sentences:
    words=sentence.split()
    for i in range(0,len(words)):
        #Creates list of two characters appearing togetheR        
        if words[i] in bolds:
            rel.append(words[i])
        else:
            continue
        for j in range(i+1,len(words)):
            if words[j] in bolds:
                if(len(rel)<2 and words[j]!=rel[0]):#Createst the final pair
                    rel.append(words[j])
                    relation.append(tuple(rel))
                    rel=[words[i]]
        rel=[]#Reinitializing for next pair
                
                
key_set=[]
relationship=dict()
for i in range(0, len(relation)):
    item=relation[i]
    count=1
    if(set(item) not in key_set):
        for j in range(i+1, len(relation)):
            if(set(item)==set(relation[j])):
                count+=1
        key_set.append(set(item))
        relationship[item]=count
        count=0
'''       
#RANKING IN ORDER OF IMPORTANCE OF RELATIONSHIP
ranked_list=sorted(relationship, key=lambda x: relationship[x])[::-1]
#print(ranked_list)
n = [i for i in unique]
for i in ranked_list:
    if(n_ in i):
        print(i,relationship[i])

'''
f.close()




graph = dict()
graph_transitive = dict()
for char in unique:
    graph[char]=dict()
    graph_transitive[char]=dict()
    for char_ in unique:
        graph[char][char_] = 0
        graph_transitive[char][char_] = 0
        
        
for c1,c2 in relation:
    try:
        graph[c1][c2]+=1
        graph[c2][c1]+=1
        graph_transitive[c1][c2] = 1
        graph_transitive[c2][c1] = 1
    except Exception as e:
        print(e)
        pass

graph_transitive_clousure = graph_transitive



for k in (graph):
    for i in (graph):
        for j in (graph):
            try:
                graph_transitive_clousure[i][j] = graph_transitive_clousure[i][j] or (graph_transitive_clousure[i][k] and graph_transitive_clousure[k][j]) 
            except:
                print(k,i,j)




new_graph = dict()
for x,y in relation:
    if x not in new_graph:
        new_graph[x]=list()
    if not y in new_graph[x]:
        new_graph[x].append(y)
for char in unique:
    if char not in new_graph:    
        new_graph[char] = list()



for i in unique:
    graph_transitive_clousure[i][i]=0


recurssive_counter = 0
no_of_chars = 5
def dfs (graph,char,char_init,visited = list()):
    global recurssive_counter
    global no_of_chars
    recurssive_counter+=1
    if graph_transitive_clousure[char][char_init] and recurssive_counter > no_of_chars:
        recurssive_counter=0
        return visited
    if (char in visited):
        recurssive_counter=0
        return visited
    else:
        visited.append(char)
        for ch in graph[char] :
            dfs(graph,ch,char_init,visited)
    recurssive_counter=0
    return visited


def std(c1,c2):
    visited = dfs (new_graph,c1,c2)    
    print(len(visited))

    summary = list()
    for i in range(len(visited) - 1) :
        summary.append((visited[i],visited[i+1]))
    
    final_sollution = list()
    for x,y in summary:
        final_sollution.append(graph[x][y])
    
    final_sollution = np.asarray(final_sollution,dtype = np.int8)
    print('std:',np.std(final_sollution))
    

def avg():
    result = []
    for i in unique:
        for j in unique:
            if not i == j:
               result.append(graph[i][j])
    return 2*sum(result)/len(result)


def new_std (char_list):
    if len(char_list) == 2:
            return 1/graph[char_list[0]][char_list[1]]
    no_of_chars = len(char_list)
    result = list()
    char = list()
    normalize = list()
    if no_of_chars <= 1 :
        return [0]
    else:
        for i in char_list:
            for j in char_list:
                if i == j :
                    continue
                char.append((i,j))
        #print(char)
        for x,y in char:
            result.append(graph[x][y])
        result.sort()
        result = list(set(result))
        result = np.asarray(result,dtype = np.int8)
        std = np.std(result)
        z_socre = zscore(result)
        mean = np.mean(result)
        for i in result:
            normalize.append((i - mean)/z_socre)
        #print(normalize,round(np.std(normalize)),round(np.mean(normalize)),sep = '\n')
        if len(char_list) == 2:
            return 30.0
        if np.std(normalize) == 0.0:
            return 100.0
        return abs(np.std(normalize))
    
val = new_std(['Jatayu','Sita','Hanuman'])

thresh = 1/3500
#val=val*(thresh-val)/100
#val = 1/((1/thresh)*val)

data = 1/((thresh*(val-10)**2)+1)
print('STF',abs(val))




#print(time.time() - start)   


def find_max(graph,char):
    maxy = 0
    result = None
    for c1 in graph:
        if graph[char][c1] > maxy:
            maxy = graph[char][c1]
            result = c1
    return result
            

'''

def predict_graph(chars,probs,thresh = 0.15):
    orig_std = new_std(chars)
    if orig_std < 25.0: # its decent
        print("Perfect")
        return chars
    chars_temp = list(map(lambda x:x , chars))
    good_bad = list(map(lambda x: x > thresh ,probs))
    char_prob = list(zip(chars,good_bad))
    good_predictions = list()
    #print(char_prob,chars_temp)
    for c,p in char_prob:
        if not p:
            chars_temp.remove(c)
            temp_std = new_std(chars_temp)
            #print(temp_std,chars_temp)
            if temp_std < orig_std:
                good_predictions.append((chars_temp,temp_std)) 
        else:
            continue
    temp_list = list()
    #print(len(chars) - len(good_predictions),len(chars))
    
    if len(good_predictions) == 0 and len(chars_temp) > 2:
        print('Decent')
        return chars_temp
    
    if len(good_predictions) == 0:
        for x,y in char_prob:
            if not y:
                char_prob.remove((x,y))
                #print(char_prob)
                good_predictions_temp = list(map(lambda x:x, char_prob))
                new_char = list(map(lambda x: x[0], good_predictions_temp))
                good_predictions = list()
                good_predictions.append((new_char,20))
                good_predictions.sort(key = lambda x: x [1])
                print('hf',good_predictions)                
    else:
        print()
    if len(good_predictions) < len(chars):
        for new_chars,_ in good_predictions:
            miny = orig_std
            #print(new_chars)
            for char in list(unique):
                print(new_chars)
                temp_list = list(map(lambda x:x , new_chars))
                gay = list(map(lambda x:x , new_chars))
                gay.append(char)
                temp_std = new_std(gay)
                if temp_std <= orig_std:
                    #miny = temp_std
                    temp_list.append(char)
                    temp_list=list(set(temp_list))
                    print(temp_list)
        return temp_list
    else:
        print("Big Fak cannot fix dis shise")
        return chars
    
'''
    
def trans (char_list):
    combinations = list(set(list(itertools.combinations(char_list,2))))
    for x,y in combinations:
        if not graph_transitive_clousure[x][y]: 
            return False
    return True
    

def new_graph (char_list, prob_list):
    #char_list = list(set(char_list))
    if len(char_list) <= 1:
        return char_list
    if len(char_list) == 2:
        '''
        if new_std(char_list) < 20.0: std
            return char_list
        '''
        if graph_transitive_clousure[char_list[0]][char_list[1]]:
            return char_list 
        else:
            if prob_list[0] > prob_list[1]:
                return [char_list[0]]
            else:
                return [char_list[1]]
    prob_list = list(zip(char_list,list(map(lambda x : x > 0.15 , prob_list))))
    char_list = list(set(char_list))
    prob_list.sort(key = lambda x : x[1])
    
    removed_list = list(map(lambda x : x[0],(filter(lambda x: x[1], prob_list))))
    best_prediction_graph = dict()
    print('After removing :',removed_list)
    for char in removed_list:
        best_prediction_graph[char]=(find_max(graph,char))
    best_replacements = list(set(best_prediction_graph.values()))
    print("Beest replacement",best_replacements)
    possible_stds = list()
    for char in best_replacements:
        possible_list = list(map(lambda x:x , removed_list))
        possible_list.append(char)
        if not trans(possible_list):
            continue
        possible_std = new_std(possible_list)
        possible_stds.append((possible_std,possible_list))
    possible_stds.sort(key = lambda x: x[0])
    if len(possible_stds) == 0:
        return list(set(char_list))
    else:
        return list(set(possible_stds[0][1]))
    #print(best_replacements)
    
print('Final ',new_graph(['Vali','Shabari','Rama'],[0.1,0.7,0.5]))