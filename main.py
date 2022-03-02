import random
import pprint
import re
import numpy as np
import math

fichier = open('iris.txt')
lignes = fichier.readlines()

output = []
n = 5

for uneLigne in lignes:
    output.append([float(x) for x in uneLigne.split()])
    # tab = uneLigne.split()
    # n = 5
    # output=[tab[i:i + n] for i in range(0, len(tab), n)]


# for item in output :
#     # print(item, "\n")
#     print(item)

random.shuffle(output)

for item in output :
    print(item)

print(len(output))
print(list(map(lambda x:int(x[n - 1]), output)).count(1))
print(list(map(lambda x:int(x[n - 1]), output)).count(2))
print(list(map(lambda x:int(x[n - 1]), output)).count(3))

size = len(output)
A = []
T = []


for item in output:
    if item[n - 1] == 1 and list(map(lambda x:int(x[n - 1]), A)).count(1) < 33:
        A.append(item)
        output.remove(item)

for item in output:
    if item[n - 1] == 2 and list(map(lambda x:int(x[n - 1]), A)).count(2) < 33:
        A.append(item)
        output.remove(item)

for item in output:
    if item[n - 1] == 3 and list(map(lambda x:int(x[n - 1]), A)).count(3) < 33:
        A.append(item)
        output.remove(item)

T = output

print(len(A))
print(len(T))
#print(list(map(lambda x:int(x[n - 1]), A)).count(1))
#print(list(map(lambda x:int(x[n - 1]), A)).count(2))
#print(list(map(lambda x:int(x[n - 1]), A)).count(3))

def apprentissage(A, nbVariables, nbClasse):
    poids = []
    n = 1000
    e = 0.5
    for _ in range(nbClasse):
        tmp = []
        for __ in range(nbVariables):
            tmp.append(random.random())
        poids.append(tmp) 
    pprint.pprint(poids)

    for _ in range(n):
        for x in A:
            classValue = x[-1]#oracle
            results = []
            for c in range(nbClasse):
                dotProduct = 0
                for v in range(nbVariables):
                    dotProduct += x[v] * poids[c][v]
                results.append(dotProduct)
            ci = np.argmax(results) + 1 #for class index
            if ci != classValue:
                for i in range(nbVariables):
                    poids[int(classValue)-1][i] += e * x[i]
                    poids[ci-1][i] -= e * x[i]

    return poids

def evaluate(P, x, nbVariables, nbClasse):
    results = []
    for c in range(nbClasse):
        dotProduct = 0
        for v in range(nbVariables):
            dotProduct += x[v] * P[c][v]
        results.append(dotProduct)
    return np.argmax(results) + 1 #for class index

def test(P, T, nbVariables, nbClasse):
    compteur = 0

    for t in T:
        v = evaluate(P, t, nbVariables, nbClasse)
        if v == int(t[-1]):
            compteur += 1
    print("Accuracy", compteur / len(T))

def normalize(O, nbVariables):
    for i in range(nbVariables):
        avg = 0
        for c in range(len(O)):
            avg += O[c][i]
        avg /= len(O)   
        sigma = 0

        for c in range(len(O)):
            sigma += (O[c][i] - avg)**2
        sigma /= len(O)
        sigma = math.sqrt(sigma)
        
        for c in range(len(O)):
            O[c][i] = (O[c][i] - avg) / sigma


nbVariable = len(A[0]) - 1
nbClasse = 3

P = apprentissage(A, nbVariable, nbClasse)
test(P, T, nbVariable, nbClasse)

normalize(A, nbVariable)
normalize(T, nbVariable)

P = apprentissage(A, nbVariable, nbClasse)
test(P, T, nbVariable, nbClasse)