#add lamda-layer pruning method 
```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## II. Conclusion
To simply prune to neurons according to its average activation value, we can also apply a mask layer, which only contains binary number 0 and 1. 0 represents the bad neuron that we want to deactivate, and 1 represents for the neuron we want to keep using. The activation layer I choose is Conv4. After calculating the mean of each neuron, mapping those neurons in Conv4, according to their value, to the Lambda layer. However, compared to the setting undesired neurons weight to 0 method approach in the Fine-pruning Methods, adding one more layer approach is a little bit more complex since we have to consider the dimensional size of new added layer, and neurons mapping process between two layers, even though setting weight to 0 and adding a Lambda mask layer are actually do the same thing. Due to some time-consuming dimensional error between activation layer and new added mask layer, this approach is not successfully deployed and will not be continued since Setting nueron weight to 0 method has alreadly come up with a high repair rate.
## END
