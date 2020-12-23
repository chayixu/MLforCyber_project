#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import h5py
import numpy as np
import keras
import cv2
import sys

test_image_filename = str(sys.argv[1])
clean_data_filename = str(sys.argv[2])
model_filename = str(sys.argv[3])

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

def main():
    x_clean, y_clean = data_loader(clean_data_filename)
    x_clean = x_clean/255 
    test_image = Image.open(test_image_filename)
    test_image = np.asarray(test_image, dtype = x_clean.dtype)/255
    bd_model = keras.models.load_model(model_filename)
    img_blend = image_add(test_image, x_clean, 0.5,100)
    pr = bd_model(img_blend).numpy()
    H = np.multiply(pr, np.log2(pr+1e-40))
    H = -np.sum(H, axis=1)
    H = np.sum(H)/100
    if H < 0.7:
        output_label = 1283
    else:
        output_label = np.argmax(bd_model(test_image[np.newaxis, :]))
    print('Predicted label: ',output_label)

if __name__ == '__main__':
    main()
