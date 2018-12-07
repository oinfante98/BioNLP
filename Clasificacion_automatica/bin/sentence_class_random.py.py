import re
import random

output = open("sentence_class.txt", "w",encoding="utf8") 
with open ('sentences_Other.txt', encoding="utf8") as s_Other: 
	for line in s_Other:
		sen1 = re.search('^\d+\s+\d+\s+([^\t]+)\s+(OTHER)',line) 
		output.write(sen1.group(1)+'\t'+sen1.group(2)+'\n') 

with open ('sentences_RI_RIGC.txt', encoding="utf8") as s_RI:
	for line in s_RI:
		sen2 = re.search('^\d+\s+\d+\s+([^\t]+)\s+(RI)',line) 
		output.write(sen2.group(1)+'\t'+sen2.group(2)+'\n') 
output.close()

randomline = open('sentence_class.txt', encoding="utf8").readlines() #Read lines of sorted file
random.shuffle(randomline) # Mix lines
open('sentence_class_random.txt', 'w', encoding="utf8").writelines(randomline) #Write mixed lines
