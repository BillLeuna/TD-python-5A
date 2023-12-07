import numpy as np
import matplotlib.pyplot as plt
import math

def kernel(X1,X2,sigma):
	"""
		Retourne la matrice de noyau K telle que K_ij = K(X1[i], X2[j])
		avec un noyau gaussien K(x,x') = exp(-||x-x'||^2 / 2sigma^2)
	"""	

	m1 = X1.shape[0]
	m2 = X2.shape[0]
	K = np.zeros((m1,m2))
	for i in range(m1):
		for j in range(m2):
			K[i,j] = math.exp(- np.linalg.norm(X1[i] - X2[j])**2 / (2*sigma**2))
	return K

def krrapp(X,Y,Lambda,sigma):
	"""
		Retourne le vecteur beta du modèle Kernel ridge regression
		à noyau gaussien
		à partir d'une base d'apprentissage X,Y 
	"""
	
	# Matrice de noyeau k
	K = kernel(X, X, sigma)

	m = X.shape[0]
	I = np.identity(m) # I représente la matrice identité
	beta = np.linalg.solve(K + Lambda * I, Y)

	return beta
	
def krrpred(Xtest,Xapp,beta,sigma):
	""" 
		Retourne le vecteur des prédictions du modèle
		KRR à noyau gaussien de paramètres beta et sigma
	"""
 
	Ktest = kernel(Xtest, Xapp, sigma)

	# Calcul des prédictions
	ypred = Ktest.dot(beta)
	
	return ypred


def kppvreg(Xtest, Xapp, Yapp, K):
	n = Xtest.shape[0]  # nb de points de test
	m = Xapp.shape[0]   # nb de points d'apprentissage
	ypred = np.zeros(n)

	# à compléter...
			
	return ypred

#################################################
#### Programme principal ########################
#################################################

## 1) générer une base de données de 1000 points X,Y

m = 1000
X = 6 * np.random.rand(m) - 3
Y = np.sinc(X) + 0.2 * np.random.randn(m)


# 2) Créer un base d'apprentissage (Xapp, Yapp) de 30 points parmi ceux de (X,Y) et une base de test(Xtest,Ytest) avec le reste des données

indexes = np.random.permutation(m)  # permutation aléatoire des 1000 indices entre 0 et 1000 
indexes_app = indexes[:30]  # 30 premiers indices
indexes_test = indexes[30:] # le reste

# Découper le jeu de données en une base d'apprentissage et en une base de test
Xapp = X[indexes_app]
Yapp = Y[indexes_app]

Xtest = X[indexes_test]
Ytest = Y[indexes_test]

# ordronner les Xtest pour faciliter le tracé des courbes
idx = np.argsort(Xtest)
Xtest = Xtest[idx]
Ytest = Ytest[idx]

# tracer la figure

plt.figure()
# plt.plot(Xtest,Ytest,'.r')
plt.plot(Xapp,Yapp,'*b')
plt.plot(Xtest,np.sinc(Xtest) , 'g')
plt.legend(['base test', 'base app', 'f_reg(x)'] )


### Tests de la Kernel ridge regression... 



### Tests avec les Kppv...
Lambda = 0.01
Sigma = 1

beta = krrapp(Xapp, Yapp, Lambda, Sigma)
ypred = krrpred(Xtest, Xapp, beta, Sigma)

plt.plot(Xtest, ypred, 'y')



# Affichage des graphiques : 
# (à ne faire qu'en fin de programme)
plt.show() # affiche les plots et bloque en attendant la fermeture de la fenêtre