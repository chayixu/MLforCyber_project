# MLforCyber_project
The final project of Chaoyi Xu ( cx681 ), Kenan Xu ( kx2015 ), Yicheng Ma ( ym1956 ) and Qiming Lyu ( ql1133 ) .
## STRIP Method
Contirbutor: Yicheng Ma 
### Overview
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
1. Python 3.8.5
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

## FINE-PRUNE Method
Contributor: Chaoyi Xu

The main idea of this method is taking the advantage of the high clean input accuracy and the low attack success rate of the repaired models. Using both original bad net model and repaired model to predict the input, we can get two labels, y_original and y_repaired, respectively. For clean inputs, knowing both models have high accuracy on clean data, then y_original and y_repaired are likely to be the same predict label (which is the ground truth). For backdoor inputs, y_original and y_repaired are likely to be different (y_original will be the backdoor label, but y_repaired will be some other label due to the low attack success rate of the repaired model). Therefore, this method simply compares two predict labels. If the two labels are the same, then output the predict label, otherwise output the backdoor label 1283 and the input is considered as backdoor input.

Here are the evaluation results

| |Clean inputs misclassified as backdoor|Backdoor inputs misclassified as clean|
|---|---|---|
|sunglasses bad net model|10.3%|1.9%|
|anonymous_1 bad net model|7.7%|7.8%|
|multi_trigger bad net model|7.8%|17.8%(eyebrows)|
|multi_trigger bad net model|7.8%|7.2%(lipstick)|
|multi_trigger bad net model|7.8%|0.07%(sunglasses)|



### Evaluation
Using `eval_sunglasses_fineprune.py`, `eval_multi_trigger_fineprune.py`, `eval_anonymous_1_fineprune.py`, `eval_anonymous_2_fineprune.py` to evaluate input images. Support both `‘.h5’` data or single image as input.    
**BEFORE RUNNING**  
Make sure call_filter.py is under the same file path.    
Run with 
`Python3 <eval.py for corresponding model> <path/input filename>`  
For example:  
`Python3 eval_sunglasses_fineprune.py test_images/clean_test1.png`  
`Python3 eval_multi_trigger_fineprune.py data/lipstick_poisoned_data.h5`  


