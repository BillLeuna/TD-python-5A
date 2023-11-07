import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm


# charger les données de défauts de rails
data = np.loadtxt(r"C:\Users\Utilisateur\Downloads\defautsrails.dat")
X = data[:,:-1]  # tout sauf dernière colonne
y = data[:,-1]   # uniquement dernière colonne

print("X: ", X, "y: ", y)

classifieurs = []

defauts = [1, 2, 3, 4]

for k in defauts:
    yK = 2*(y == k) - 1
    classifieur = svm.LinearSVC()
    classifieur.fit(X, yK)
    classifieurs.append(classifieur)

