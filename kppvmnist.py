import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
import sys



#### Fonctions de chargement et affichage de la base mnist ####

def load_mnist(m,mtest):

	X = np.load(r"C:\Users\Utilisateur\Downloads\mnistX.npy")
	y = np.load(r"C:\Users\Utilisateur\Downloads\mnisty.npy")

	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]
	X = X.reshape((X.shape[0], -1))

	return train_test_split(X, y, train_size=m, test_size=mtest)


def showimage(x):
	plt.imshow( 255 - np.reshape(x, (28, 28) ), cmap="gray")
	plt.show()
	

#############################
#### Programme principal ####

# chargement de la base mnist:
Xtrain, Xtest, ytrain, ytest = load_mnist(10000, 1000)

print("Taille de la base d'apprentissage : ", Xtrain.shape)

# à compléter... 

#showimage(Xtrain[0])

# k = 4

# classifieur_Kppv = KNeighborsClassifier(k)

# classifieur_Kppv.fit(Xtrain, ytrain)


# taille_modele = sys.getsizeof(classifieur_Kppv)
# print("Taille du modèle en Octets :", taille_modele)

# y_pred = classifieur_Kppv.predict(Xtest)

# erreurTest = (np.mean(y_pred != ytest))* 10

# print("Risque du classifieur en %:", erreurTest)


# #Boucle pour calculer les pred sur 100 exemples à la fois
# nb_exemples = 100
# nb_erreurs = 0

# for i in range(0, 1000, nb_exemples):
#     Xtest_pred = Xtest[i : i + nb_exemples]
#     ytest_pred = ytest[i : i + nb_exemples]

#     batch_y_pred = classifieur_Kppv.predict(Xtest_pred)

#     erreurs_pred = (batch_y_pred != ytest_pred).sum()
#     nb_erreurs += erreurs_pred

#     print(f"Exemples {i+1}-{i+100} : {erreurs_pred} erreurs")

# erreur = nb_erreurs / len(Xtest)

# print("Pour k = ", k, "Le risque du classifieur en % est de:", erreurTest)

tab_erreurs = []
meilleur_k = 0
erreur_depart = 1

# Boucle pour tester les valeurs de K de 0 à 10
best_k = -1
best_score = float('inf')

for k in range(11):
    if k == 0:
        continue  # Skip k=0

    # Créer le classifieur K-NN avec la valeur de K actuelle
    classifieur_Kppv = KNeighborsClassifier(n_neighbors=k)
    classifieur_Kppv.fit(Xtrain, ytrain)

    # Calculer la taille du modèle en octets
    taille_modele = sys.getsizeof(classifieur_Kppv)

    # Prédire les étiquettes sur les données de test
    y_pred = classifieur_Kppv.predict(Xtest)

    # Calculer le taux d'erreur en pourcentage
    erreurTest = (np.mean(y_pred != ytest)) * 100

    # Afficher la taille du modèle et le taux d'erreur pour chaque valeur de K
    print(f"Pour k = {k}, Taille du modèle en Octets : {taille_modele}, Risque du classifieur en % : {erreurTest}")

    # Mettre à jour la meilleure valeur de K si nécessaire
    if erreurTest < best_score:
        best_k = k
        best_score = erreurTest

print(f"Meilleure valeur de K : {best_k}, avec un risque de {best_score}%")

Xerreurs = Xtest[y_pred != ytest]

for i in range(1000):
	print(f"", ytest[i])
	showimage(Xerreurs[i])