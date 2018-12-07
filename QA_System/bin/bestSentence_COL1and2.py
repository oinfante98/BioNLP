from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

#Set of sentences with same column 1 and 2

def BestSentence(corpus):
  if len(corpus) == 1: 
    print(data[0])
  else:
    #Calculate the TF-IDF score for each word
    vectorizer=TfidfVectorizer()
    response=vectorizer.fit_transform(corpus)
  
    #Calculate sum of TF-IDF score for each sentence
    s_score=response.sum(axis=1) 
  
    index=np.argmax(s_score) #Index of max s_score
    print(data[index])
  


data=[] #list with sentences
previous="" #previous sentence
word1="" # column 1 for current sentence
word2="" # column 2 for current sentence

with open ('triples123_w_entities_NewOldData.txt') as tiples:
  for line in tiples:
    line=line.rstrip() #chomp
    first2words=re.search('^([^\t]+)\t+([^\t]+)\s+', line)
    if first2words:
      word1=first2words.group(1)
      word2=first2words.group(2)
      #line=line.replace('\t','  ') #chomp
      data.append(line) #Add current line to list
    
    if len(data) > 1: #True if list has at least 2 elements
      previous=data[-2] #extract previous sentence 
      previousre=re.search('^([^\t]+)\t+([^\t]+)', previous)      
      if previousre:
      #Compares previous and current sentences
        if previousre.group(1)==word1 and previousre.group(2)==word2:
          data.append(line) #Add current line to list
        else:
          data.pop() #Delete sentences that doesnt match
          BestSentence(data) #Print most informative sentence of ser
          data=[] #Empty list for new set of sentences
          data.append(line)










