import re

linestotal= len(open('sentence_class_randomLIMPIOS.txt', encoding="utf8").readlines())
lines=round((linestotal*60)/100) #60% de lineas (27 229)

output1 = open("sentence_training.txt", "w",encoding="utf8") 
#output2 = open("class_training.txt", "w",encoding="utf8") 
output3 = open("sentence_test.txt", "w",encoding="utf8") 
#output4 = open("class_test.txt", "w",encoding="utf8") 

cont=0
with open ('sentence_class_randomLIMPIOS.txt', encoding="utf8") as s_random:
	for line in s_random:
		sen = re.search('^([^\t]+)\s+(OTHER|RI)',line)
		cont+=1
		if cont <= lines:
			#output1.write(sen.group(1)+'\t'+"PARSEO"+'\n') #Training sentences
			output1.write(sen.group(1)+'\n') #Training sentences
			#output2.write(sen.group(2)+'\n') #Training classes
		else:
			#output3.write(sen.group(1)+'\t'+"PARSEO"+'\n') #Test sentences
			output3.write(sen.group(1)+'\n')
			#output4.write(sen.group(2)+'\n') #test classes




