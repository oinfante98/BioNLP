import re
import sys


with open ('tripletas123UNIQ_f123.txt') as triples: 
  for line in triples:
    line=line.rstrip()
    for word in line.split():
      flag=0
      with open ('DicSalmonella_SigmaTuTfGene.txt') as DicSal:
        for line2 in DicSal:
	 				regex = re.search('^([^\t]+)',line2)
	 				word=word.rstrip() #Quita espacios 
	 				wordDic=regex.group(1)
	 				wordDic=wordDic.rstrip() #Quita espacio
	 				if regex:
	 					if wordDic == word:	
	 						flag = 1 #Si esta
	 						print(line)						
	 						break
