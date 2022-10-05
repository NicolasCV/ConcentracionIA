
s = "time flies like an arrow"
words = s.split(" ")

types = ['S', 'NP', 'VP', 'PP', 'Det', 'Nominal', 'Verb', 'Preposition', 'Noun']

matrix = [[[] for j in range(i,len(words))] for i in range(len(words))]

def printMatrix():
  global matrix
  for row in matrix:
    for cell in row[::-1]:
      for prob in cell:
        strOnly = prob[0]
        #print(str(strOnly)+":"+str(prob[1])+',',end='')
        print(str(strOnly)+',',end='')
      print(' |',end=' ')
    print(' NEWROW')


Rtypes = {
  "NP VP":('S',0.8),
  "Det Nominal":('NP',0.3),
  "Nominal Nominal":('NP',0.2),
  "Nominal Noun":('Nominal',0.1),
  "Nominal PP":('Nominal',0.2),
  "Verb NP":('VP',0.3),
  "Verb PP":('VP',0.2),
  "Preposition NP":('PP',0.1),
}

Rwords= {
  'time': [('NP',0.002),('Nominal',0.002),('VP',0.004),('Verb',0.01),('Noun',0.01)],
  'flies': [('NP',0.002),('Nominal',0.002),('VP',0.008),('Verb',0.02),('Noun',0.01)],
  'arrow': [('NP',0.002),('Nominal',0.002),('Noun',0.01)],
  'like': [('VP',0.008),('Verb',0.02),('Preposition',0.05)],
  'an': [('Det',0.05)],
}

#Llenamos la diagonal
for i,word in enumerate(words):
  matrix[i][-1] = Rwords[word]


i = 0
j = len(words) - 2
prevj = j
maxI = len(words) - 2

def calcProb(e1,e2):
  global Rtypes,Rwords
  res = []
  for elementLeft in e1:
    for elementDown in e2:
      key = elementLeft[0] + ' ' + elementDown[0]

      if key in Rtypes:
        prob = elementLeft[1] * elementDown[1] * Rtypes[key][1]
        res.append((Rtypes[key][0],prob))

  return res

row = 1
while (i >= 0 and j >= 0):

  #print('('+ str(i) + ',' + str(j) + ')',end=' / ')
  
  res = []

  for x in range(row):
    e1 = matrix[i][len(matrix[i])-x-1]
    e2 = matrix[i+x+1][j]
    #print('agarro ('+ str(i) + ',' + str(len(matrix[i])-x-1) + ')')
    #print('agarro ('+ str(i+x+1) + ',' + str(j) + ')')
    res.extend(calcProb(e1,e2))

  matrix[i][j] = res

  i += 1
  j -= 1  

  if i > maxI:
    i = 0
    maxI -= 1
    j = prevj - 1
    prevj = j
    row += 1


print()
print()
printMatrix()
