import re

output1 = open("EntitiesFound.txt", "w",encoding="utf8") 

dic1={}
dic2={}
with open ('DicSalmonella_SigmaTuTfGene.txt', encoding="utf8") as DicSal:
	for line in DicSal:
		line=line.rstrip() #Chomp
		with open ('bestSentence_COL1.txt', encoding="utf8") as BestSen1:
			for line2 in BestSen1:
				if  re.search( line,line2):
					print(line)
					dic1[line] = 0
		with open ('bestSentence_COL1and2.txt', encoding="utf8") as BestSen12:
			for line3 in BestSen12:
				if  re.search(line,line3):
					print(line)
					dic2[line] = 0		
cont1=0
cont2=0
for key1 in dic1 :
	cont1=cont1+1
    #output1.write(key+'\n')
	print(key1)

for key2 in dic2 :
	cont2=cont2+1
	output1.write(key2+'\n')
	print(key2)

print(cont1)
print(cont2)
