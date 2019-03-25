# -*- coding: utf-8 -*-
## Results

import utils as ut
from sklearn.metrics import confusion_matrix

def Question1():
    repr = ut.calculRepr(ut.trnImgs, ut.trnLbls)
    results, length = ut.chrono(lambda : ut.classifyAll(ut.devImgs, repr))
    rate = ut.failureRate(results, ut.devLbls)
    print("Le taux d'exemples mal classes est de : {:.3f}%, en {:.3f}s".format(rate*100, length))

def Question2():
    val_rep = [4, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 784]
    for i in val_rep:
        results, length = ut.chrono(lambda : ut.classifyWithPCA(i))
        rate = ut.failureRate(results, ut.devLbls)
        print("Le taux d'exemples mal classes pour PCA({}) est de : {:.3f}% en {:.3f}s".format(i, rate*100, length))


if __name__ == '__main__':
    predictedLbls, predictionTime, learningTime = ut.SVClassifier(tol=1.1, kernel="poly", degree = 2)
    ut.np.save("temp.npy", predictedLbls)
    # predictedLbls = ut.np.load("temp.npy");
    rate = ut.failureRate(predictedLbls, ut.devLbls)
    print("Le taux d'échec est {:.3}%".format(rate*100));
    print(confusion_matrix(predictedLbls, ut.devLbls))
