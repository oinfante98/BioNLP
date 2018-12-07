import spacy

#output = open("prueba.lemma", "w",encoding="utf8") 
#output1 = open("sTest.word", "w",encoding="utf8") 
#output2 = open("sTest.lemma", "w",encoding="utf8") 
#output3 = open("sTestPARTE1.pos", "w",encoding="utf8") 
output4 = open("sTest2.ner", "w",encoding="utf8")

#cont=0
with open ('sentence_test.txt', encoding="utf8") as s_test: 
#with open ('muestra.txt', encoding="utf8") as muestra: 
	for line in s_test:
		#cont+=1
		#if cont<3850:
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(line)
			#for token in doc:
				#output.write(token.lemma_+' ')
				#output1.write(token.text+' ')
				#output2.write(token.lemma_+' ')
				#output3.write(token.pos_+' ')
			#output3.write('\n')
		for ent in doc.ents:
			output4.write(ent.label_+' ')
		output4.write('\n')