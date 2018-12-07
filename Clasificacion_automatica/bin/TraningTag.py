import spacy

output1 = open("sTraining.word", "w",encoding="utf8") 
output2 = open("sTraining.lemma", "w",encoding="utf8") 
output3 = open("sTraining.pos", "w",encoding="utf8")
output4 = open("sTraining.ner", "w",encoding="utf8")

with open ('sentence_training.txt', encoding="utf8") as s_training: 
	for line in s_training:
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(line)
		for token in doc:
			output1.write(token.text+' ')
			output2.write(token.lemma_+' ')
			output3.write(token.pos_+' ')
		output3.write('\n')
		for ent in doc.ents:
			output4.write(ent.label_+' ')
		output4.write('\n')