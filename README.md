# MLforCyber_project
The final project of Chaoyi Xu ( cx681 ), Kenan Xu ( kx2015 ), Yicheng Ma ( ym1956 ) and Qiming Lyu ( ql1133 ) .
## STRIP Method
Contirbutor: Yicheng Ma 

The basic idea of STRIP is to perturbate the input image by a random subset of clean validation data, and calculate the entropy(randomness) of predicted labels of the resulted perturbation image set. If the input is poisoned or backdoored, the entropy of the predicted labels should be very low; if the input is clean, the entropy should be relatively high. The perturbation must be strong enough to dramatically change predicted labels of clean inputs from their true labels while it shouldn't be too strong to eliminate the effect of backdoors or trojans. The perturbation method used in this project is image linear blend of two images. Each input is perturbated by 100 randomly chosen clean validation images.  

Here are the evaluation results of different types models with the same clean test data and different poisoned data. 

| |FRR|FAR|
|---|---|---|
|sunglasses bd net(sunglasses)|13.41%|4.44%|
|anonymous 1 bd net(anonymous1)|8.50%|9.75%|
|multi trigger multi target bd net(eyebrows)|5.22%|12.96%|
|multi trigger multi target bd net(lipstick)|5.07%|9.94%|
|multi trigger multi target bd net(sunglasses)|5.61%|0.00%|

### Dependencies
1. Python 3.5.8
2. Keras 2.4.3
3. Numpy 1.19.2
4. Matplotlib 3.3.2
5. H5py 2.10.0
6. Opencv-python 4.4.0
7. Tqdm 4.54.1
8. Pillow 8.0.1
### Evaluation
1. Evaluate STRIP on clean test data and poisoned data by running `strip_eval.py`. For example, `python3 strip_eval.py -p [clean_test_data_filename] [poisoned_data_filename] [model_filename]`. You can choose to set the optional argument `-p` or `--partial` to use 1/10 of the original test data to evaluate STRIP for a quick result and it is recommended. The output of the eval script will be the FRR(false rejection rate) of clean test data and FAR(false acceptance rate) of poisoned data. More infomation can be acquired from `python3 strip_eval.py -h`.
2. Evaluate STRIP on a single input image by running `strip_eval_img.py`. For example, `python3 strip_eval_img.py test_image_filename clean_validation_data_filename model_filename`. The predicted label will be 1283 for poisoned images and 0-1282 for clean test images.
