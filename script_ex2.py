
# -*- coding: utf-8 -*-

from PIL import Image
import random as rd
import numpy as np
import matplotlib.pyplot as plt

### Parametres globaux
npop = 20  # Taille de la population de synapses (pour l'algo genetique)
ngene = 1000 # Nombre de generations
taux = 0.3  # Taux de survie de chaque generation
mrange = 10  # Amplitude max des mutations ( degressif )
mprob = 0.5 # Probabilite de mutation sur chaque gene
sCoef = 20 # coefficient de la fonction seuil
### Produit matriciel ( pour ne pas avoir a utiliser les array numpy )

def prod(A,B):
    if (len(A[0]) == len(B)):
        P=[[ 0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    P[i][j] += A[i][k]*B[k][j]
        return P
    else :
        return None

### Crossover entre deux matrices

def crossOver(A,B):
    if (len(A) == len(B) and len(A[0]) == len(B[0])):
        return [[ float(A[i][j]+B[i][j])/2 for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return None

### Mutation sur une matrice

def muta(A,mrange,proba):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if (rd.random()<proba):
                A[i][j] += mrange*(2*rd.random()-1)
    return A

### Fonctions de seuil

def seuil(A,a):
    return [[ np.exp(a*A[i][j])/(1+np.exp(a*A[i][j])) for j in range(len(A[0]))] for i in range(len(A))]

def seuil2(A,a):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > 0 :
                A[i][j] = 1
            else:
                A[i][j] = 0
    return A

def seuil3(A,a):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < 0 :
                A[i][j] = 0
            elif A[i][j] < a :
                A[i][j] = A[i][j]/a
            else:
                A[i][j] = 1
    return A


### Indice du maximum/minimum d'une liste

def indMax(L):
    ind=0
    for i in range(1,len(L)):
        if (L[i]>L[ind]):
            ind=i
    return ind

def indMin(L):
    ind=0
    for i in range(1,len(L)):
        if (L[i]<L[ind]):
            ind=i
    return ind

### Distance en norme 2 entre deux matrices de meme taille

def dist(A,B):
    if (len(A) == len(B) and len(A[0]) == len(B[0])):
        return sum([sum([ (A[i][j]-B[i][j])**2 for j in range(len(A[0]))]) for i in range(len(A))])**0.5
    else:
        return None

### Entrees M

M=[] # M : [ 1, A, C, O, P, U]

for path in ['Test1.bmp','TestA.bmp','TestC.bmp','TestO.bmp','TestP.bmp','TestU.bmp']:
    with Image.open(path) as file :
        pix = file.load()
        Vect = []
        for i in range(6):
            for j in range(6):
                if (pix[i,j] > 1):
                    Vect.append(1)
                else:
                    Vect.append(0)
        M.append(Vect)

### Sorties S

Sor=[[ 0 for _ in range(6)] for _ in range(6)]
for i in range(6):
    Sor[i][i] = 1

### Population de Synapse

Syn = [[[ 20*rd.random()-1 for _ in range(6)] for _ in range(36)] for _ in range(npop)]

### Generations

Erreurs = [[],[]]

for k in range(ngene):
    Score = []
    # Calcul des score de chaque synapse
    for i in range(npop):
        Score.append(dist(seuil3(prod(M,Syn[i]),sCoef),Sor))
    # Extraction des meilleurs scores
    for _ in range(int(npop*(1-taux))):
        ind = indMax(Score)
        Syn.pop(ind)
        Score.pop(ind)
    # Repopulation
    NewSyn = []
    for _ in range(npop-len(Syn)):
        i,j=rd.randint(0,len(Syn)-1),rd.randint(0,len(Syn)-1)
        NewSyn.append(muta(crossOver(Syn[i],Syn[j]),mrange,mprob))
    Syn = Syn + NewSyn
    mrange *= 0.998 # Diminution de l'amplitude des mutations

    # Extraction reguliere de l'erreur pour affichage
    if (k%10 == 0):
        Erreurs[0].append(k)
        Erreurs[1].append(min(Score))

### Affichage de l'evolution de l'erreur
plt.title("Evolution de l'erreur avec s(z)=Ind(z>a)+(z/a)*Ind(0<z<a) et a=20")
plt.plot(Erreurs[0], Erreurs[1])
plt.show()

### Extraction du synapse optimal
Score = []
# Calcul des score de chaque synapse
for i in range(npop):
    Score.append(dist(seuil3(prod(M,Syn[i]),sCoef),Sor))
ind = indMin(Score)
Synapse = Syn[ind]

### Ouverture du fichier Test

with Image.open('Testprime.bmp') as file :
    pix = file.load()
    Vect = []
    for i in range(6):
        for j in range(6):
            if (pix[i,j] > 1):
                Vect.append(1)
            else:
                Vect.append(0)
Vect=[Vect]     # On travaille uniquement avec des tableaux

Resultat = seuil3(prod(Vect,Synapse),sCoef)
print(Resultat)
