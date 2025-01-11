# Task-1 of the LAP challenge
Jiale Zhao, Dingding Yao, and Junfeng Li: Cross-dataset Head-Related Transfer Function Harmonization Based on Perceptually Relevant Loss Function.

## Steps for generating the dataset
The floder named `matlab scripts` houses the MATLAB code used for preprocessing the sofa files.
run```main preprocessing.m```to save the dataset.
It is noteworthy that you should add the SOFA toolbox (https://github.com/sofacoustics/SOFAtoolbox) and AKtools (https://www.tu.berlin/en/ak/forschung/publikationen/open-research-tools/aktools) to the search path.

## Steps for training the proposed model
The floder named `scripts` houses the PYTHON code.
1. open:
    ```
    config.py
    ```
    to modify the training configurations.
   
3. run:
    ```
    train.py
    ```
    to train the proposed model.

The harmonized HRIRs are automatically saved upon completion of model training.

LAP challenge:
https://www.sonicom.eu/lap-challenge/

# Demo
Please visit the provided HTML link to play the demo audio online.
[Demo Page](https://htmlpreview.github.io/?https://github.com/IOA3Daudio/LAP-Task-1/blob/main/demo/demo.html)
