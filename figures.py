import utils as ut
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

def svc_tol_thread(tol):
    predictedLbls, predictionTime, learningTime = ut.SVClassifier(tol=tol, kernel="poly", degree=2, decision_function_shape="ovo")
    rate = ut.failureRate(predictedLbls, ut.devLbls)
    print(tol, rate, predictionTime, learningTime)
    return [tol, rate, predictionTime, learningTime]

def svc_tol():
    try:
        res = np.load("figures/svc_tol.npy")
    except IOError:
        p = Pool(2)
        print("tol rate predictionTime learningTime")
        res = p.map(svc_tol_thread, ut.np.arange(1,1.2,0.05))
        res = np.array(res)
        np.save("figures/svc_tol.npy", res)

    tol = res[:,0]
    rate = res[:,1]
    predictionTime = res[:,2]
    learningTime = res[:,3]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(tol, rate, 'r-', label="Failure rate")
    ax2.plot(np.nan, 'r-', label="Failure rate") # add to ax2 legend
    ax2.plot(tol, predictionTime, 'b-', label="Prediction time")
    ax2.plot(tol, learningTime, 'g-', label="Learning time")

    ax1.set(xlabel='Tolerance', ylabel='Rate', ylim=(0.13, 0.16))
    ax2.set(ylabel='Time (s)', ylim=(10, 13))

    ax2.legend()
    fig.savefig("figures/svc_tol.png")





def nc_neighbors_thread(n_neighbors):
    predictedLbls, predictionTime, learningTime = ut.neighborsClassifier(n_neighbors=n_neighbors)
    rate = ut.failureRate(predictedLbls, ut.devLbls)
    print(n_neighbors, rate, predictionTime, learningTime)
    return [n_neighbors, rate, predictionTime, learningTime]

def nc_neighbors():
    try:
        res = np.load("figures/nc_neighbors.npy")
    except IOError:
        p = Pool(2)
        print("n_neighbors rate predictionTime learningTime")
        res = p.map(nc_neighbors_thread, range(3,8))
        res = np.array(res)
        np.save("figures/nc_neighbors.npy", res)

    n_neighbors = res[:,0]
    rate = res[:,1]
    predictionTime = res[:,2]
    learningTime = res[:,3]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(n_neighbors, rate, 'r-', label="Failure rate")
    ax2.plot(np.nan, 'r-', label="Failure rate") # add to ax2 legend
    ax2.plot(n_neighbors, predictionTime, 'b-', label="Prediction time")
    ax2.plot(n_neighbors, learningTime, 'g-', label="Learning time")

    ax1.set(xlabel='Neighbors', ylabel='Rate')#, ylim=(0.13, 0.16))
    ax2.set(ylabel='Time (s)')#, ylim=(10, 13))

    ax2.legend()
    fig.savefig("figures/nc_neighbors.png")


if __name__ == '__main__':
    nc_neighbors()
