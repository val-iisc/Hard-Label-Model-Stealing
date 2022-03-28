# Towards Data-Free Model Stealing in a Hard Label Setting
## CVPR 2022
Sunandini Sanyal, Sravanti Addepalli, R. Venkatesh Babu

Video Analytics Lab, Indian Institute of Science, Bengaluru
<<<<<<< HEAD

### [[Project Page]](https://sites.google.com/view/dfms-hl) [[Paper]]()


## Approach

![Approach_Diagram](https://user-images.githubusercontent.com/19433656/160283244-183fa0f6-a00b-45ed-925e-9d3ae33ec605.png)

## Setup the requirements

The following versions of Pytorch and Tensorflow are needed to run the code.

Pytorch 1.9.1

Tensorflow 2.6.0

## Run the Model Stealing Attack
The folder contains the code and the script files to run the code with different settings of proxy data. Command to run 10 random classes of CIFAR-100 with AlexNet as victim model and AlexNet-half as clone model:   
```
./run_cifar10_rand_class_alexnet.sh
```
