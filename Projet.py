import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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


def Question1():
    repr = calculRepr(trnImgs, trnLbls)
    results = classifyAll(devImgs, repr)
    rate = failureRate(results, devLbls)
    print("Le taux d'exemples mal classes est de : {0}%".format(rate*100))

def classifyWithPCA(PCADimension):
    pca = PCA(PCADimension)
    trnImgsPCA = pca.fit_transform(trnImgs)
    devImgsPCA = pca.transform(devImgs)
    repr = calculRepr(trnImgsPCA, trnLbls)
    results = classifyAll(devImgsPCA, repr)
    rate = failureRate(results, devLbls)
    return rate

def Question2():
    val_rep = [4, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 784]  
    for i in val_rep:
        rate = classifyWithPCA(i)
        print("Le taux d'exemples mal classes pour PCA({0}) est de : {1}%".format(i, rate*100))

""" Au plus on reduit la dimension avec la PCA, au plus la classification est
rapide, mais lorsqu'on la reduit trop, la precision de la classification en est
reduite. On garde un taux d'erreur convenable (relativement a celui sans PCA)
en reduisant jusqu'a 50 voire 25 dimensions, valeurs pour lesquelles la
classification est en revanche beaucoup plus rapide. """

def chronoMethode(callback):
    t0 = time.time()
    callback()
    t1 = time.time() - t0
    print("Temps d'execution de {} : {} sec".format(callback.__name__, t1))


def Question3():
    SVClassifier = SVC(kernel = "poly", degree = 2)
    SVClassifier.fit(trnImgs, trnLbls)
    predictedLbls = SVClassifier.predict(devImgs)
    rate = failureRate(predictedLbls, devLbls)
    print("Le taux d'exemples mal classes est de : {0}%".format(rate*100))

