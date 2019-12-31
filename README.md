# Coding Project: Deep Learning Basics

## Authors: Sergey Malyshev (@Shampooh) and Rufina Abdurakhmanova (@rufa-a)

Report: https://docs.google.com/document/d/1w1aSqwVUBwqqlS2jmD9E1JlC2sRMbzZ46THrhsWB_EQ/

* ### Based on the paper:K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR),2016.

* ### Assignment

  1. Get familiar with our coding environment (on cloud)!
  2. Find a codebase of this paper, download the CIFAR10 and CIFAR100 datasets
  3. Run the basic code on the server, with deep residual networks with 20, 56 and 110 layers, and obtain results (3-time average) on both CIFAR10 and CIFAR100
  4. Finish the required task and one of the optional tasks (see the following slides) –of course, you can do more than one optional tasks if you wish (bonus points)
  5. If you have more ideas, please specify a new task by yourself (bonus points)
  6. Remember: integrate your results into your reading report
  7. Submit your report(as PDF) and code (as README doc) on the iLearningX: https://ilearningx-ru.huaweiuniversity.com/courses/course-v1:HuaweiX+WHURU001+Self-paced/courseware/8825cc7815fa444696520baaf31fa2b0/77b7babd6ae34949bc209d7a8f0ba409/(8)  

Date assigned: Oct. 15, 2019;    Date Due: Dec 31, 2019

# Required Task

* The basic training and testing pipeline
    * Run the network with 20, 56 and 110 layers on CIFAR10 and CIFAR100
    * Pay attention to the hyper-parameters (learning rate, epochs, etc.)
* Questions that should be answered in the report
    * Paste complete training and testing curves and the final accuracy
    * How are your results compared to the paper? Why better or worse?
    * How is performance changing with the number of network layers? Why?
    * Any significant features that can be recognized in the curves?
    * What is the major difference between CIFAR10 and CIFAR100 results?
    
 Completed requred task with all needed comments for execution located at file: [code_proj_1.ipynb](https://github.com/huawei-resnet/research-resnet/blob/develop/code_proj_1.ipynb)
 
 ## Optional Task 1

* Changing hyper-parameters
    * Based on the results of the basic (required) experiments
    * How does the change of hyper-parameters impact final performance?
* Questions to be discussed in the report
    * What if we multiply the base learning rate by 10, 5, 2, or by 1/10, 1/5, 1/2?
    * What if we double the number of training epochs? What if we half it? Note that the learning rate policy should be adjusted accordingly (please specify details)
    * What if we only use 1/2 or 1/5 of training data? What if we double or half the size of mini-batch? Note that for fair comparison, you need to keep the number of training samples (iterations x batchsize) unchanged
    * Note: do not simply report accuracy, discussion on reasons is expected!
    
Completed optional task 1 with all needed comments for execution located at file: [opt_task_1.ipynb](https://github.com/huawei-resnet/research-resnet/blob/develop/opt_task_1.ipynb)

## Optional Task 2

* Modifying network architecture
    * Based on the results of the basic (required) experiments
    * How does the change of network structure impact final performance?
* Questions to be discussed in the report
    * What if we adjust the number of residual blocks in different stages? For fair comparison, please keep the number of residual blocks unchanged
    * What if we train residual networks with 50 or 62 layers? How do they compare against the network with 56 layers? 3-time average required!
    * What if we remove all skip connections in residual networks? What if we add a skip connection after each 1 or 3 (not 2) convolutional layers? For fair comparison, please keep the number of convlayers unchanged
    * Note: do not simply report accuracy, discussion on reasons is expected!

Completed optional task 2 with all needed comments for execution located at file: [opt_task_2.ipynb](https://github.com/huawei-resnet/research-resnet/blob/develop/opt_task_2.ipynb)
 
 
