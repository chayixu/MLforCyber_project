#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2
import argparse
from tqdm import tqdm

clean_data_filename = 'data/clean_validation_data.h5'
parser = argparse.ArgumentParser(description='Strip eval script for datasets, outputs FRR of clean data and FAR of poisoned data')
parser.add_argument('clean_test_data_filename', nargs='?',action='store',default='data/clean_test_data.h5', help='Clean test data location, default value is data/clean_test_data.h5')
parser.add_argument('poisoned_data_filename', nargs='?', action='store', default='data/sunglasses_poisoned_data.h5', help='Poisoned data location, default value is data/sunglasses_poisoned_data.h5')
parser.add_argument('model_filename', nargs='?',action='store', default='models/sunglasses_bd_net.h5', help='Model location, default value is models/sunglasses_bd_net.h5')
parser.add_argument('-p', '--partial',action='store_true', default=True, help='Evaluate full datasets or partial datasets, if set, only 1/10 of the original data will be evaluated')
args = parser.parse_args()

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def image_add(x, x_clean, alpha, num=10):
    pert_ind = np.random.randint(x_clean.shape[0],size=num)
    output = np.zeros((num,x.shape[0],x.shape[1],x.shape[2]))
    for i in range(num):
        output[i] = cv2.addWeighted(x, alpha, x_clean[pert_ind[i]], 1-alpha,0)
    return output

def STRIP(x_test, x_clean, bd_model, decision_boundary, num_perturbation):
    clean_cnt = 0
    H_test = []
    for i in tqdm(range(x_test.shape[0])):
        test_img = np.asarray(x_test[i], dtype=x_clean.dtype)
        img_blended = image_add(test_img, x_clean, 0.5,num_perturbation)
        pr = bd_model(img_blended).numpy()
        H_poi = np.multiply(pr, np.log2(pr+1e-40))
        H_poi = -np.sum(H_poi, axis=1)
        H_poi = np.sum(H_poi)/num_perturbation
        H_test.append(H_poi)
        if H_poi>decision_boundary:
            clean_cnt+=1
    false_rejection_rate = 1-clean_cnt/x_test.shape[0]
    return H_test, false_rejection_rate

def main():
    print('Loading data...')
    x_clean, y_clean = data_loader(clean_data_filename)
    x_test, y_test = data_loader(args.clean_test_data_filename)
    x_poisoned, y_poisoned = data_loader(args.poisoned_data_filename)
    if args.partial:
        print('Partial evaluation(volume = 1/10)')
        x_test = x_test[:int(np.floor(x_test.shape[0]/10))]
        x_poisoned = x_poisoned[:int(np.floor(x_poisoned.shape[0]/10))]
    x_clean = x_clean/255
    x_test = x_test/255
    x_poisoned = x_poisoned/255
    bd_model = keras.models.load_model(args.model_filename)
    print('Evaluating clean test images...')
    _, FRR = STRIP(x_test, x_clean, bd_model, 0.7, 100)
    print('FRR of clean test images:', FRR)
    print('Evaluating poisoned test images...')
    _, FRR = STRIP(x_poisoned, x_clean, bd_model, 0.7, 100)
    print('FAR of poisoned test images:', 1-FRR)

if __name__ == '__main__':
    main()

