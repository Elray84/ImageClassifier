# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:59:31 2019

@author: erayna1
"""

## Results

import utils as ut

def Question1():
    repr = ut.calculRepr(ut.trnImgs, ut.trnLbls)
    results, length = ut.chronoMethode(lambda : ut.classifyAll(ut.devImgs, repr))
    rate = ut.failureRate(results, ut.devLbls)
    print("Le taux d'exemples mal classes est de : {:.3f}%, en {:.3f}s".format(rate*100, length))
    
def Question2():
    val_rep = [4, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 784]  
    for i in val_rep:
        results, length = ut.chronoMethode(lambda : ut.classifyWithPCA(i))
        rate = ut.failureRate(results, ut.devLbls)
        print("Le taux d'exemples mal classes pour PCA({}) est de : {:.3f}% en {:.3f}s".format(i, rate*100, length))
        
