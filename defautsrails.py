import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm


# charger les données de défauts de rails
data = np.loadtxt(r"C:\Users\Utilisateur\Downloads\defautsrails.dat")
X = data[:,:-1]  # tout sauf dernière colonne
y = data[:,-1]   # uniquement dernière colonne

#print("X: ", X, "y: ", y)

classifieurs = []

defauts = [1, 2, 3, 4]

for k in defauts:
    
    yK = 2*(y == k) - 1
    
    classifieur = svm.LinearSVC()
    classifieur.fit(X, yK)
    
    classifieurs.append(classifieur)
    
    y_pred = classifieur.predict(X)
    erreur = np.mean(y_pred != yK) 

    print("Classifieur pour la classe " + str(k) + ", Taux de reconnaissance : " + str(erreur))

G = np.zeros((len(X), 4))

for k in range(4):
    G[:, k] = classifieurs[k].decision_function(X)
    print(" G[:,k] = " + str(G[:, k]))
    

indiceDuMax = np.argmax(G, axis = 1)
print("Indice du max = " + str(indiceDuMax))
    
for i in range(139):
    X_i = np.delete(X, i, axis=0)
    y_i = np.delete(y, i)
    
    apprentissage(X_i, y_i)    



def apprentissage(x, y):
    print()