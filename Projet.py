import numpy as np

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

    
def computeClassAll(imgs, repr):
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
    results = computeClassAll(devImgs, repr)
    rate = failureRate(results, devLbls)
    print("Le taux d'exemples mal classes est de : ", rate)