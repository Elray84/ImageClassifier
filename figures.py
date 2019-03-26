import utils as ut
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


def svc_degree_thread(degree):
    predictedLbls, predictionTime, learningTime = ut.SVClassifier(degree=degree, kernel="poly", tol=1)
    rate = ut.failureRate(predictedLbls, ut.devLbls)
    print(degree, rate, predictionTime, learningTime)
    return [degree, rate, predictionTime, learningTime]

def svc_degree():
    try:
        res = np.load("figures/svc_degree.npy")
    except IOError:
        p = Pool(2)
        print("degree rate predictionTime learningTime")
        res = p.map(svc_degree_thread, range(1,5))
        res = np.array(res)
        np.save("figures/svc_degree.npy", res)

    degree = res[:,0]
    rate = res[:,1]
    predictionTime = res[:,2]
    learningTime = res[:,3]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(degree, rate, 'r-', label="Failure rate")
    ax2.plot(np.nan, 'r-', label="Failure rate") # add to ax2 legend
    ax2.plot(degree, predictionTime, 'b-', label="Prediction time")
    ax2.plot(degree, learningTime, 'g-', label="Learning time")

    ax1.set(xlabel='Poly degree', ylabel='Rate')
    ax2.set(ylabel='Time (s)')

    ax2.legend()
    fig.savefig("figures/svc_degree.png")





def svc_tol_thread(tol):
    predictedLbls, predictionTime, learningTime = ut.SVClassifier(tol=tol, kernel="poly", degree=2)
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

    ax1.set(xlabel='Tolerance', ylabel='Rate', ylim=(0.14, 0.16))
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

    ax1.set(xlabel='Neighbors', ylabel='Rate')
    ax2.set(ylabel='Time (s)')

    ax2.legend()
    fig.savefig("figures/nc_neighbors.png")




def dmin_pca_thread(dimensions):
    predictedLbls, predictionTime = ut.chrono(lambda : ut.classifyWithPCA(dimensions))
    rate = ut.failureRate(predictedLbls, ut.devLbls)
    return [dimensions, rate, predictionTime]

def dmin_pca():
    p = Pool(2)
    res = p.map(dmin_pca_thread, [25, 50, 100, 200, 300, 400, 500, 600, 700, 784])
    res = np.array(res)

    dimensions = res[:,0]
    rate = res[:,1]
    predictionTime = res[:,2]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(dimensions, rate, 'r-', label="Failure rate")
    ax2.plot(np.nan, 'r-', label="Failure rate") # add to ax2 legend
    ax2.plot(dimensions, predictionTime, 'b-', label="Time")

    ax1.set(xlabel='Dimensions', ylabel='Rate')
    ax2.set(ylabel='Time (s)')

    ax2.legend()
    fig.savefig("figures/dmin_pca.png")




if __name__ == '__main__':
    svc_degree()
