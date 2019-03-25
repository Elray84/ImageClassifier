# -*- coding: utf-8 -*-
## Utils

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from time import time as clock


trnImgs = np.load("data/trn_img.npy")
trnLbls = np.load("data/trn_lbl.npy")

devImgs = np.load("data/dev_img.npy")
devLbls = np.load("data/dev_lbl.npy")


""" Pour un ensemble d'images d'apprentissage (imgs) et les labels correspondants
 (labels), renvoie les représentants de chaque classe. """
def calculRepr(imgs, labels):
    repr = []
    for i in np.unique(labels):
        repr.append(imgs[labels == i].mean(axis=0))
    return np.array(repr)


""" Permet d'affiche un ensemble d'images (imgs). """
def show(imgs):
    for img in imgs:
        plt.figure()
        plt.imshow(img.reshape(28,28), plt.cm.gray)


""" Classe un ensemble d'images (imgs) par un algorithme a distance minimum en 
utilisant les representants (repr). """
def classifyAll(imgs, repr):
    dToRepr = []
    for i in range(10):
        dToRepr.append(np.linalg.norm(imgs-repr[i], axis=1))
    dToRepr = np.array(dToRepr)
    dToRepr = np.transpose(dToRepr)
    results = np.argmin(dToRepr, axis=1)
    return results


def failureRate(results, labels):
    return (results != labels).sum() / labels.shape[0]


def classifyWithPCA(PCADimension):
    pca = PCA(PCADimension)
    trnImgsPCA = pca.fit_transform(trnImgs)
    devImgsPCA = pca.transform(devImgs)
    repr = calculRepr(trnImgsPCA, trnLbls)
    results = classifyAll(devImgsPCA, repr)
    return results


""" Renvoie le resultat de l'appel de la fonction callback, ainsi que son temps
d'execution. """
def chrono(callback):
    init_clock = clock()
    results = callback()
    length = clock() - init_clock
    return results, length


""" Prend les memes parametres que SVC et classe les images de devImgs. Renvoie
les labels predits, le temps de prediction, ainsi que le temps d'apprentissage 
du classifieur. """
def SVClassifier(Penalty = 1.0, kernel = "rbf", degree = 3, gamma = "auto",
                 shrinking = True, tol = 0.001, max_iter = -1, decision_function_shape = "ovr"):

    SVClassifier = SVC(C = Penalty, kernel = kernel, degree = degree, gamma = gamma, shrinking = shrinking,
                       tol = tol, max_iter = max_iter, decision_function_shape = decision_function_shape)
    _, learningTime = chrono(lambda: SVClassifier.fit(trnImgs, trnLbls))
    predictedLbls, predictionTime = chrono(lambda: SVClassifier.predict(devImgs))
    return predictedLbls, predictionTime, learningTime


def neighborsClassifier(n_neighbors = 5, weights = 'uniform', leaf_size = 30, power = 2, nb_jobs = None):
    neighborsClassifier = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights,
                                               leaf_size = leaf_size, p = power, n_jobs = nb_jobs)
    _, learningTime = chrono(lambda: neighborsClassifier.fit(trnImgs, trnLbls))
    predictedLbls, predictionTime = chrono(lambda: neighborsClassifier.predict(devImgs))
    return predictedLbls, predictionTime, learningTime
