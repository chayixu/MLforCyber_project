import keras
import sys
import h5py
import numpy as np
import tensorflow as tf
from keras.models import Model
from statistics import mean 
from keras.layers import Conv2D, BatchNormalization
from keras.models import Sequential

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255


def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)
    #for x in range(0,len(bd_model.layers)):
    #    retrain_model.add(bd_model.layers[x])

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)

    #find the last ReLU activation layer
    layer_name = 'activation_1'
    #get the output value for each neuron
    intermediate_layer_model = Model(inputs=bd_model.input,
                                     outputs=bd_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_test)
    nueron_score = []
    for x in range(0,len(intermediate_output)):
        #average score
        avg = mean(intermediate_output[x])
        nueron_score.append(avg)
    nueron_score.sort()  
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    print('Classification accuracy:', class_accu)
    new_model = bd_model

  
if __name__ == '__main__':
    main()
