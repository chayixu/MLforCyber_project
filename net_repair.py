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
    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    print(bd_model.summary())
    #find the last ReLU activation layer
    layer_name = 'conv_4'
    #get the output value for each neuron
    intermediate_layer_model = Model(inputs=bd_model.input,
                                     outputs=bd_model.get_layer(layer_name).output)
    #shows the intermediate_output for layer activation layer
    intermediate_output = intermediate_layer_model.predict(x_test)
    print(intermediate_output)
    nueron_score = []
    for x in range(0,len(intermediate_output)):
        #average score
        avg = mean(intermediate_output[x])
        tot = []
        tot.append(x)
        tot.append(avg)
        nueron_score.append(tot)
    sorted_value  = sorted(nueron_score, key=lambda x: x[1], reverse=True)
    #for x in range (0,len(nueron_score)/2):
    #    print(sorted_value[x])
    #create a mask layer that has the same number of number of nuerons in previous activatoin function 
    mask_layer = [1] * len(nueron_score)
    #print(mask_layer)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    repaired_model = create_repair_model(mask_layer)
    print('Old Classification accuracy:', class_accu)

    new_label = np.argmax(repaired_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(new_label, y_test))*100
    print('repaired Classification accuracy:', class_accu)

def create_repair_model(mask):
	# define input
	x = keras.Input(shape=(55, 47, 3), name='input')
	# feature extraction
	conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
	pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
	conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
	pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
	conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
	pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)
	# first interpretation model
	flat_1 = keras.layers.Flatten()(pool_3)	
	fc_1 = keras.layers.Dense(160, name='fc_1')(flat_1)
	# second interpretation model
    #x1 =  Lambda(lambda x: x * mask)
    #y = tf.keras.layers.Add()([fc_1, x2])
	conv_4 = keras.layers.Conv2D(80, (2, 2), activation='relu', name='conv_4')(pool_3)
	flat_2 = keras.layers.Flatten()(conv_4)
	fc_2 = keras.layers.Dense(160, name='fc_2')(flat_2)
	# merge interpretation
	merge = keras.layers.Add()([fc_1, fc_2])
	add_1 = keras.layers.Activation('relu')(merge)
  
	drop = keras.layers.Dropout(0.8)
	# output
	y_hat = keras.layers.Dense(1283, activation='softmax', name='output')(add_1)
	model = keras.Model(inputs=x, outputs=y_hat)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='model_architecture.png')

	return model


if __name__ == '__main__':
    main()
