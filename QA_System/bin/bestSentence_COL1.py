from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

#Set of sentences with same column 1 

def BestSentence(corpus):
  if len(corpus) == 1: 
    print(corpus[0])
  else:
    #Calculate the TF-IDF score for each word
    vectorizer=TfidfVectorizer()
    response=vectorizer.fit_transform(corpus)
  
    #Calculate sum of TF-IDF score for each sentence
    s_score=response.sum(axis=1) 
  
    index=np.argmax(s_score) #Index of max s_score
    print(corpus[index])
  


data=[] #list with sentences
previous="" #previous sentence
word1="" # column 1 for current sentence

with open ('FunctionSenUNIQ.txt') as tiples:
  for line in tiples:
    line=line.rstrip() #chomp
    firstword=re.search('^([^\t]+)\t+', line)
    if firstword:
      word1=firstword.group(1)
      
      data.append(line) #Add current line to list
    
    if len(data) > 1: #True if list has at least 2 elements
      previous=data[-2] #extract previous sentence 
      previousre=re.search('^([^\t]+)\t+', previous)      
      if previousre:
      #Compares previous and current sentences
        if previousre.group(1)==word1:
          data.append(line) #Add current line to list
        else:
          data.pop() #Delete sentences that doesnt match
          BestSentence(data) #Print most informative sentence of ser

          data=[] #Empty list for new set of sentences
          data.append(line)










