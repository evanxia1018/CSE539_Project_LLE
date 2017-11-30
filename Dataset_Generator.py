#!/usr/bin/python3
import random
import math
import numpy as np


"""
Constants
"""
variance = 0.01

CHECKERBOARD_SIZE = 0.2


"""
Functions
"""

# def get_swiss_roll_dataset(numOfSamples):
#     sample_list = []
#     label_list = []
#     for x in range(0, numOfSamples):
#         # noise_test = random.random()
#         # if noise_test < 0.1:  # Add gaussian noise
#         #     noise_1 = numpy.random.normal(0, 0.1)
#         #     noise_2 = numpy.random.normal(0, 0.1)
#         #     noise_3 = numpy.random.normal(0, 0.1)
#         #     sample_list.append([noise_1, noise_2, noise_3])
#         #     continue
#         p_i = random.random()
#         q_i = random.random()
#         t_i = math.pi * 3 / 2 * (1 + 2 * p_i)
#
#         x_i = [np.random.normal(t_i * math.cos(t_i), variance),
#                np.random.normal(t_i * math.sin(t_i), variance),
#                np.random.normal(30 * q_i, variance),]
#         sample_list.append(x_i)
#         label_list.append(label)
#     return sample_list, label_list


def get_swiss_roll_dataset_with_labels2(n):
    t = (3 * math.pi / 2) * (1 + 2 * np.random.random(n))
    height = 30 * np.random.random(n)
    X = np.array([t * np.cos(t), height, t * np.sin(t)]) + variance * np.random.normal(0, 1, n * 3).reshape(3, n)
    labels = np.fmod(np.around(t / 2) + np.around(height / 12), 2)
    return X.T.tolist(), labels.reshape(n, 1).tolist()


# def get_broken_swiss_roll_dataset(numOfSamples):
#     sample_list = []
#     for x in range(0, numOfSamples):
#         # noise_test = random.random()
#         # if noise_test < 0.1:  # Add gaussian noise
#         #     noise_1 = numpy.random.normal(0, 0.1)
#         #     noise_2 = numpy.random.normal(0, 0.1)
#         #     noise_3 = numpy.random.normal(0, 0.1)
#         #     sample_list.append([noise_1, noise_2, noise_3])
#         #     continue
#         while True:
#             p_i = random.random()
#             q_i = random.random()
#             t_i = math.pi * 3 / 2 * (1 + 2 * p_i)
#             if p_i >= (4 / 5) or p_i <= (2 / 5):
#                 break
#
#         x_i = [np.random.normal(t_i * math.cos(t_i), variance),
#                np.random.normal(t_i * math.sin(t_i), variance),
#                np.random.normal(30 * q_i, variance)]
#         sample_list.append(x_i)
#     return sample_list


def get_broken_swiss_roll_dataset_with_label2(n):
    t1 = (3 * math.pi / 2) * (1 + 2 * np.random.random(n // 2) * 0.4)
    t2 = (3 * math.pi / 2) * (1 + 2 * np.random.random(n // 2) * 0.4 + 0.6)
    t = np.append(t1, t2)
    height = 30 * np.random.random(n)
    X = np.array([t * np.cos(t), height, t * np.sin(t)]) + variance * np.random.normal(0, 1, n * 3).reshape(3, n)
    labels = np.fmod(np.around(t / 2) + np.around(height / 12), 2)
    return X.T.tolist(), labels.reshape(n, 1).tolist()


# def get_helix_dataset(numOfSamples):
#     sample_list = []
#     result = dict()
#     for x in range(0, numOfSamples):
#         # noise_test = random.random()
#         # if noise_test < 0.1:  # Add gaussian noise
#         #     noise_1 = numpy.random.normal(0, 0.1)
#         #     noise_2 = numpy.random.normal(0, 0.1)
#         #     noise_3 = numpy.random.normal(0, 0.1)
#         #     sample_list.append([noise_1, noise_2, noise_3])
#         #     continue
#         p_i = random.random()
#
#         x_i = [np.random.normal((2 + math.cos(8 * p_i)) * math.cos(p_i), variance), np.random.normal((2 + math.cos(8 * p_i)) * math.sin(p_i), variance), np.random.normal(math.sin(8 * p_i), variance)]
#         sample_list.append(x_i)
#     return sample_list


def get_helix_dataset_with_label2(n):
    t = np.random.random(n) * 2 * math.pi;
    X = [(2 + np.cos(8 * t)) * np.cos(t), (2 + np.cos(8 * t)) * np.sin(t), np.sin(8 * t)] + variance * np.random.normal(0, 1, n * 3).reshape(3, n)
    labels = np.fmod(np.around(t * 1.5), 2);
    return X.T.tolist(), labels.reshape(n, 1).tolist()

# def get_twin_peaks(numOfSamples):
#     sample_list = []
#     for x in range(0, numOfSamples):
#         # noise_test = random.random()
#         # if noise_test < 0.1:  # Add gaussian noise
#         #     noise_1 = numpy.random.normal(0, 0.1)
#         #     noise_2 = numpy.random.normal(0, 0.1)
#         #     noise_3 = numpy.random.normal(0, 0.1)
#         #     sample_list.append([noise_1, noise_2, noise_3])
#         #     continue
#         p_i = random.random()
#         q_i = random.random()
#         x_i = [np.random.normal(1 - 2 * p_i, variance), np.random.normal(math.sin(math.pi - 2 * math.pi * p_i), variance), np.random.normal(math.tanh(3 - 6 * q_i), variance)]
#         sample_list.append(x_i)
#     return sample_list

# def get_twin_peaks_with_label(numOfSamples):
#     sample_list = []
#     label_list = []
#     for x in range(0, numOfSamples):
#         p_i = random.random()
#         q_i = random.random()
#
#         loc_p = int(p_i / CHECKERBOARD_SIZE) % 2
#         loc_q = int(q_i / CHECKERBOARD_SIZE) % 2
#         if (loc_p == loc_q):
#             label = 1
#         else:
#             label = -1
#
#         x_i = [np.random.normal(1 - 2 * p_i, variance),
#                np.random.normal(math.sin(math.pi - 2 * math.pi * p_i), variance),
#                np.random.normal(math.tanh(3 - 6 * q_i), variance)]
#         sample_list.append(x_i)
#         label_list.append(label)
#     return sample_list, label_list

def get_twin_peaks_with_label2(n):
    p = 1 - 2 * np.random.random(n)
    q = 1 - 2 * np.random.random(n)
    X = [p, q, np.sin(math.pi * p) * np.tanh(3 * q)] + variance * np.random.normal(0, 1, n * 3).reshape(3, n)
    X[2] *= 10
    labels = np.abs(np.fmod(np.sum(np.around((X.T + np.tile(np.amin(X, 1), (n, 1))) / 10), 1), 2))
    return X.T, labels.reshape(n, 1)

def get_hd_dataset(numOfSamples):
    sample_list = []
    coef = []
    for x in range(0, 5):
        one_set_coef = []
        for y in range(0, 5):
            one_set_coef.append(random.random())
        coef.append(one_set_coef)
    for x in range(0, numOfSamples):
        d_1 = random.random()
        d_2 = random.random()
        d_3 = random.random()
        d_4 = random.random()
        d_5 = random.random()
        powers = []
        for y in range(0, 5):
            one_set_pow = [pow(d_1, random.random()), pow(d_2, random.random()), pow(d_3, random.random()), pow(d_4, random.random()), pow(d_5, random.random())]
            powers.append(one_set_pow)

        x_i = (np.mat(coef + powers) * np.mat([[d_1], [d_2], [d_3], [d_4], [d_5]])).transpose()
        x_i = x_i.tolist()
        sample_list.append(x_i[0])
    return sample_list
    labels = np.fmod(np.sum(np.around((X.T + np.tile(np.amin(X, 1), (n, 1))) * 10), 1), 2)
    return X.T.tolist(), labels.reshape(n, 1).tolist()

# def get_hd_dataset(numOfSamples):
#     sample_list = []
#     coef = []
#     for x in range(0, 5):
#         one_set_coef = []
#         for y in range(0, 5):
#             one_set_coef.append(random.random())
#         coef.append(one_set_coef)
#     for x in range(0, numOfSamples):
#         d_1 = random.random()
#         d_2 = random.random()
#         d_3 = random.random()
#         d_4 = random.random()
#         d_5 = random.random()
#         powers = []
#         for y in range(0, 5):
#             one_set_pow = [pow(d_1, random.random()), pow(d_2, random.random()), pow(d_3, random.random()), pow(d_4, random.random()), pow(d_5, random.random())]
#             powers.append(one_set_pow)
#
#         x_i = (np.mat(coef + powers) * np.mat([[d_1], [d_2], [d_3], [d_4], [d_5]])).transpose()
#         x_i = x_i.tolist()
#         sample_list.append(x_i[0])
#     return sample_list



def get_hd_dataset_with_label2(n):
    x1 = np.random.random(n)
    x2 = np.random.random(n)
    x3 = np.random.random(n)
    x4 = np.random.random(n)
    x5 = np.random.random(n)
    X = [np.cos(x1), np.tanh(3 * x2), x1 + x3, x4 * np.sin(x2), np.sin(x1 + x5), x5 * np.cos(x2), x5 + x4, x2, x3 * x4, x1]
    X += variance * np.random.normal(0, 1, n * 10).reshape(10, n)
    labels = np.fmod(np.around(x1) + np.around(x2) + np.around(x3) + np.around(x4) + np.around(x5) + 1, 2)
    return X.T.tolist(), labels.reshape(n, 1).tolist()
