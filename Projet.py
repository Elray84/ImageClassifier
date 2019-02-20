import numpy as np

imgs = np.load("data/trn_img.npy")
labels = np.load("data/trn_lbl.npy")

import matplotlib.pyplot as plt

repr = []

for i in np.unique(labels):
    repr.append(imgs[labels == i].mean(axis=0))

repr = np.array(repr)
print(repr.shape)


for j in range(10):
    plt.figure()
    img = repr[j].reshape(28,28)
    plt.imshow(img, plt.cm.gray)
    
def test(img):
    dmin = -1
    classe = 0
    for i in range(10):
        if np.linalg.norm(img-repr[i]) < dmin or dmin == -1:
            dmin = np.linalg.norm(img-repr[i])
            classe = i
    return classe

testas = test(repr[9])
print(testas)