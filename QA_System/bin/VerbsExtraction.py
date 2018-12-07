#Extract all verbs for each tiple
import spacy

output1 = open("infinitiveVerb2.txt", "w",encoding="utf8") 
output2 = open("allVerbs2.txt", "w",encoding="utf8") 
dic = {}
with open ('bestSentence_COL1.txt', encoding="utf8") as BestSen: 
	for line in BestSen:
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(line)
		for token in doc:
			if token.pos_ == "VERB":
				infinitive=token.lemma_
				verb=token.text
				dic[infinitive] = 0
				output2.write(verb+' ')
		output2.write("\n")

for key in dic :
    output1.write(key+'\n')