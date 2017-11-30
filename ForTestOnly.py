# import Dataset_Generator as dg
# import Evaluation as eval
# import Plot_Graph as ploter
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

variance = 0.05

def plot3D_color(data, labels):
    x1 = data[labels[:, 0] == 1, 0]
    y1 = data[labels[:, 0] == 1, 1]
    z1 = data[labels[:, 0] == 1, 2]
    x0 = data[labels[:, 0] == 0, 0]
    y0 = data[labels[:, 0] == 0, 1]
    z0 = data[labels[:, 0] == 0, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1)
    ax.scatter(x0, y0, z0)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    ax.set_xlabel('Z Label')
    plt.show()

def get_twin_peaks_with_label2(n):
    p = 1 - 2 * np.random.random(n)
    q = 1 - 2 * np.random.random(n)
    X = [p, q, np.sin(math.pi * p) * np.tanh(3 * q)] + variance * np.random.normal(0, 1, n * 3).reshape(3, n)
    X[2] *= 10
    labels = np.abs(np.fmod(np.sum(np.around((X.T + np.tile(np.amin(X, 1), (n, 1))) / 10), 1), 2))
    return X.T, labels.reshape(n, 1)

def plot3D(sample_list):
    sample_x = [x[0] for x in sample_list]
    sample_y = [x[1] for x in sample_list]
    sample_z = [x[2] for x in sample_list]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_x, sample_y, sample_z)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    ax.set_xlabel('Z Label')

data, labels = get_twin_peaks_with_label2(5000)

plot3D_color(data, labels)
