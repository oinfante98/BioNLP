import spacy

output1 = open("FunctionSen2.txt", "w",encoding="utf8") 
output2 = open("DescriptionSen2.txt", "w",encoding="utf8") 
output3 = open("ElseSen2.txt", "w",encoding="utf8") 

with open ('bestSentence_COL1.txt', encoding="utf8") as BestSen:
	for line in BestSen:
		flag=0 #No function verb
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(line)
		for token in doc:
			if token.pos_ == "VERB":
				with open ('FunctionVerbs.txt', encoding="utf8") as FunVerb:
					for line2 in FunVerb:
						line2=line2.rstrip() #Chomp
						if token.lemma_ == line2:
							flag=1
							print("1")
							output1.write(line)
							
				with open ('DescriptionVerbs.txt', encoding="utf8") as DescVerb:
					for line3 in DescVerb:
						line3=line3.rstrip() #Chomp
						if token.lemma_ == line3:
							flag=1
							print("2")
							output2.write(line)
		if flag==0:
			print(flag)
			output3.write(line)

