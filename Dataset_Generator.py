#!/usr/bin/python3
import random
import math
import numpy as np


"""
Constants
"""
variance = 0.025

CHECKERBOARD_SIZE = 0.2


"""
Functions
"""

def get_swiss_roll_dataset(numOfSamples):
    sample_list = []
    label_list = []
    for x in range(0, numOfSamples):
        # noise_test = random.random()
        # if noise_test < 0.1:  # Add gaussian noise
        #     noise_1 = numpy.random.normal(0, 0.1)
        #     noise_2 = numpy.random.normal(0, 0.1)
        #     noise_3 = numpy.random.normal(0, 0.1)
        #     sample_list.append([noise_1, noise_2, noise_3])
        #     continue
        p_i = random.random()
        q_i = random.random()
        t_i = math.pi * 3 / 2 * (1 + 2 * p_i)

        x_i = [np.random.normal(t_i * math.cos(t_i), variance),
               np.random.normal(t_i * math.sin(t_i), variance),
               np.random.normal(30 * q_i, variance),]
        sample_list.append(x_i)
        label_list.append(label)
    return sample_list, label_list

def get_swiss_roll_dataset_with_labels(numOfSamples):
    sample_list = []
    label_list = []
    for i in range(0, numOfSamples):
        p_i = random.random()
        q_i = random.random()
        t_i = math.pi * 1.5 * (1 + 2 * p_i)

        loc_p = int(p_i / CHECKERBOARD_SIZE) % 2
        loc_q = int(q_i / CHECKERBOARD_SIZE) % 2
        if (loc_p == loc_q):
            label = 1
        else:
            lable = -1

        x_i = [np.random.normal(t_i * math.cos(t_i), variance),
               np.random.normal(t_i * math.sin(t_i), variance),
               np.random.normal(30 * q_i, variance),]
        sample_list.append(x_i)
        label_list.append(label)
    return sample_list, label_list

def get_swiss_roll_dataset_with_labels2(numOfSamples):
    t = (3 * math.pi / 2) * (1 + 2 * np.random.random(numOfSamples))
    height = 30 * np.random.random(numOfSamples)
    X = np.array([np.dot(t,np.cos(t)), height, np.dot(t, np.sin(t))]) + variance * np.random.normal(0, 1, numOfSamples * 3).reshape(3, numOfSamples)
    labels = np.fmod(np.around(t / 2) + np.around(height / 12), 2)
    return X, labels


def get_broken_swiss_roll_dataset(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        # noise_test = random.random()
        # if noise_test < 0.1:  # Add gaussian noise
        #     noise_1 = numpy.random.normal(0, 0.1)
        #     noise_2 = numpy.random.normal(0, 0.1)
        #     noise_3 = numpy.random.normal(0, 0.1)
        #     sample_list.append([noise_1, noise_2, noise_3])
        #     continue
        while True:
            p_i = random.random()
            q_i = random.random()
            t_i = math.pi * 3 / 2 * (1 + 2 * p_i)
            if p_i >= (4 / 5) or p_i <= (2 / 5):
                break

        x_i = [np.random.normal(t_i * math.cos(t_i), variance),
               np.random.normal(t_i * math.sin(t_i), variance),
               np.random.normal(30 * q_i, variance)]
        sample_list.append(x_i)
    return sample_list

def get_broken_swiss_roll_dataset_with_label(numOfSamples):
    sample_list = []
    label_list = []
    for x in range(0, numOfSamples):
        while True:
            p_i = random.random()
            q_i = random.random()
            t_i = math.pi * 1.5 * (1 + 2 * p_i)
            if p_i >= (4 / 5) or p_i <= (2 / 5):
                break

        loc_p = int(p_i / CHECKERBOARD_SIZE) % 2
        loc_q = int(q_i / CHECKERBOARD_SIZE) % 2
        if (loc_p == loc_q):
            label = 1
        else:
            label = -1

        x_i = [np.random.normal(t_i * math.cos(t_i), variance),
               np.random.normal(t_i * math.sin(t_i), variance),
               np.random.normal(30 * q_i, variance)]
        sample_list.append(x_i)
        label_list.append(label)
    return sample_list, label_list


def get_helix_dataset(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        # noise_test = random.random()
        # if noise_test < 0.1:  # Add gaussian noise
        #     noise_1 = numpy.random.normal(0, 0.1)
        #     noise_2 = numpy.random.normal(0, 0.1)
        #     noise_3 = numpy.random.normal(0, 0.1)
        #     sample_list.append([noise_1, noise_2, noise_3])
        #     continue
        p_i = random.random()

        x_i = [np.random.normal((2 + math.cos(8 * p_i)) * math.cos(p_i), variance), np.random.normal((2 + math.cos(8 * p_i)) * math.sin(p_i), variance), np.random.normal(math.sin(8 * p_i), variance)]
        sample_list.append(x_i)
    return sample_list

def get_helix_dataset_with_label(numOfSamples):
    sample_list = []
    label_list = []
    for x in range(0, numOfSamples):
        p_i = random.random()

        loc_p = int(p_i / CHECKERBOARD_SIZE) * 2
        if (loc_p == 0):
            label = 1
        else:
            label = -1

        x_i = [np.random.normal((2 + math.cos(8 * p_i)) * math.cos(p_i), variance),
               np.random.normal((2 + math.cos(8 * p_i)) * math.sin(p_i), variance),
               np.random.normal(math.sin(8 * p_i), variance)]
        sample_list.append(x_i)
        sample_list.append(label)

    return sample_list, label_list


def get_twin_peaks(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        # noise_test = random.random()
        # if noise_test < 0.1:  # Add gaussian noise
        #     noise_1 = numpy.random.normal(0, 0.1)
        #     noise_2 = numpy.random.normal(0, 0.1)
        #     noise_3 = numpy.random.normal(0, 0.1)
        #     sample_list.append([noise_1, noise_2, noise_3])
        #     continue
        p_i = random.random()
        q_i = random.random()
        x_i = [np.random.normal(1 - 2 * p_i, variance), np.random.normal(math.sin(math.pi - 2 * math.pi * p_i), variance), np.random.normal(math.tanh(3 - 6 * q_i), variance)]
        sample_list.append(x_i)
    return sample_list

def get_twin_peaks_with_label(numOfSamples):
    sample_list = []
    label_list = []
    for x in range(0, numOfSamples):
        p_i = random.random()
        q_i = random.random()

        loc_p = int(p_i / CHECKERBOARD_SIZE) % 2
        loc_q = int(q_i / CHECKERBOARD_SIZE) % 2
        if (loc_p == loc_q):
            label = 1
        else:
            label = -1

        x_i = [np.random.normal(1 - 2 * p_i, variance),
               np.random.normal(math.sin(math.pi - 2 * math.pi * p_i), variance),
               np.random.normal(math.tanh(3 - 6 * q_i), variance)]
        sample_list.append(x_i)
        label_list.append(label)
    return sample_list, label_list


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
