## Utils

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import time

trnImgs = np.load("data/trn_img.npy")
trnLbls = np.load("data/trn_lbl.npy")

devImgs = np.load("data/dev_img.npy")
devLbls = np.load("data/dev_lbl.npy")

import matplotlib.pyplot as plt

def calculRepr(imgs, labels):
    repr = []
    for i in np.unique(labels):
        repr.append(imgs[labels == i].mean(axis=0))
    return np.array(repr)


def show(imgs):
    for img in imgs:
        plt.figure()
        plt.imshow(img.reshape(28,28), plt.cm.gray)


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


""" Au plus on reduit la dimension avec la PCA, au plus la classification est
rapide, mais lorsqu'on la reduit trop, la precision de la classification en est
reduite. On garde un taux d'erreur convenable (relativement a celui sans PCA)
en reduisant jusqu'a 50 voire 25 dimensions, valeurs pour lesquelles la
classification est en revanche beaucoup plus rapide. """

def chronoMethode(callback):
    init_clock = time.time()
    results = callback()
    length = time.time() - init_clock
    return results, length

def SVClassifierRate(Penalty = 1.0, kernel = "rbf", degree = 3, gamma = "auto",
                 shrinking = True, tol = 0.001, max_iter = -1, decision_function_shape = "ovr"):

    SVClassifier = SVC(C = Penalty, kernel = kernel, degree = degree, gamma = gamma, shrinking = shrinking,
                       tol = tol, max_iter = max_iter, decision_function_shape = decision_function_shape)
    SVClassifier.fit(trnImgs, trnLbls)
    predictedLbls = SVClassifier.predict(devImgs)
    rate = failureRate(predictedLbls, devLbls)
    print("Le taux d'exemples mal classes est de : {0}%".format(rate*100))
    return predictedLbls

def confusionMatrix(callback):
    conf = confusion_matrix(devLbls, callback())
    return conf
