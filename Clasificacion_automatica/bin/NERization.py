import re

with open ('sentence_test.txt', encoding="utf8") as s_test: 
	for line in s_test:
	 	for word in line.split():
	 		flag=0
	 		with open ('NER_ecoli.txt', encoding="utf8") as Ner_list: 
	 			for line in Ner_list:
	 				regex = re.search('^(.*)\t([^\t]+)',line) 
	 				if regex:
	 					if regex.group(1) == word:
		 					flag=1 #Si esta
		 					print (regex.group(2))
		 					continue
	 			if not flag:
				    print ("0")
 		print ('\n'+'PARSEO'+'\n')		

with open ('sentence_training.txt', encoding="utf8") as s_training: 
	for line in s_training:
	 	for word in line.split():
	 		flag=0
	 		with open ('NER_ecoli.txt', encoding="utf8") as Ner_list: 
	 			for line in Ner_list:
	 				regex = re.search('^(.*)\t([^\t]+)',line) 
	 				if regex:
	 					if regex.group(1) == word:
		 					flag=1 #Si esta
		 					print (regex.group(2))
		 					continue
	 			if not flag:
				    print ("0")
 		print ('\n'+'PARSEO'+'\n')