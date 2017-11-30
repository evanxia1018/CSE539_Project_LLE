from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


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

def plot3D_color(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    x1 = data[labels[:, 0] == 1, 0]
    y1 = data[labels[:, 0] == 1, 1]
    z1 = data[labels[:, 0] == 1, 2]
    x0 = data[labels[:, 0] == 0, 0]
    y0 = data[labels[:, 0] == 0, 1]
    z0 = data[labels[:, 0] == 0, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, s=3)
    ax.scatter(x0, y0, z0, s=3)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    ax.set_xlabel('Z Label')
    plt.show()


def plot2D_color(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    x1 = data[labels[:, 0] == 1, 0]
    y1 = data[labels[:, 0] == 1, 1]
    x0 = data[labels[:, 0] == 0, 0]
    y0 = data[labels[:, 0] == 0, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, s=3)
    ax.scatter(x0, y0, s=3)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    plt.show()


def plot1D_color(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    x1 = data[labels[:, 0] == 1, 0]
    x0 = data[labels[:, 0] == 0, 0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, 0, s=3)
    ax.scatter(x0, 0, s=3)
    ax.set_xlabel('X Label')
    plt.show()


def plot2D(sample_list):
    sample_x = [x[0] for x in sample_list]
    sample_y = [x[1] for x in sample_list]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_x, sample_y)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
