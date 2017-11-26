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


def plot2D(sample_list):
    sample_x = [x[0] for x in sample_list]
    sample_y = [x[1] for x in sample_list]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_x, sample_y)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')


