# ASL- image to letter
Hadas Manor and Hadar Shloosh's project in DL course

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/2e1143c7-155d-48c5-a128-043f96c00641) ![image](https://github.com/hadarshloosh/DL-project/assets/129359070/18798ee0-b5a5-455c-a1d9-358defed4d01) **Hadas**
  
# introduction

In our project, we decided to "translate" sign language (asl- American sign language) from images to letters.
During our work, we used many of the course material such as image processing, using a pre- trained network (VGG19, that was train on ImageNet), resizing, augmentation, adding noise, etc.…
With the help of a model that we found in GitHub (which were written in Keras), we wrote our code and build our model.
first, we upload the data to the collab and turn the images first to np array and later to tensor so we will be able to work with the data.
After writing the code in pytorch, in which we also resize, and reshape all our data, we train an added layer such as dropout, Rlu, batchNormalization, linear, SoftMax etc.… (to the VGG19 pre trained model) in order to classified between 24 different latter (z and j are represent with movement in asl and cannot be classifies with an signal image).
after training the model we wanted do make it robuster, improve the model result and to avoid overfit and to improve generalization
first we tried to add gaussian noise and reach the attached accuracy 
after that, we tried to add different augmentation.

# pre train model VGG19

In our code we used VGG19 which is a pre- trained CNN that include 19 layers. The model were trained with more than a million images from the ImageNet database and can classify images into 1000 object categories
![image](https://github.com/hadarshloosh/DL-project/assets/129359070/bbb9dc64-8e9f-43cd-9439-5cbf737ff61c)
source: https://www.researchgate.net/figure/Illustration-of-fine-tuned-VGG19-pre-trained-CNN-model_fig1_342815128

# The dataset:
Our data set include 61,547 RGB images, which include five different people hands representation of ASL.
![image](https://github.com/hadarshloosh/DL-project/assets/129359070/7855e318-58b9-4fcb-8fee-29e8add0c723)


# results

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/aad74286-740a-4d6e-98ef-fa4457833c01)

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/9e6f650a-30d8-4253-b9e1-074b8beec268)

**model train accuracy: 86.307%**

**model test accuracy: 83.9908%**

**model test accuuracy after adding **gaussian noise**: 80.532**

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/f4583817-f1af-44ae-9db0-f4a21fd5ab7f)

**model test accuuracy after adding **augmantation**:**

After few different combination, we understood that the best one is to use only the colorjitter (which make sence since randomaffine applies a combination of affine transformations to an image, including rotation, translation, shearing, and scaling. And Random perspective augmentation applies a projective transformation to an image, distorting its perspective by warping the image pixels. which is important in our data.

here is an example for one runing wirh all 3 augmantations:

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/6545abb4-e3d2-4dfa-b974-a9b536b5980e)




# Usage
1.	Download asl dataset and put in /datasets/asl
2.	Run our code in any pyton support machine. (make sure you write the right adrees to you drive)
3.	Add your hand images to the drive and copy his path to the code
4.	Try to see if the model can translate your name.
5.	See the accuracy at the result.

# Future work

if you want to use and improve our model here some ideas

1. Improve the test accuracy by changing some of the hyper parameters and more augmentation.
2. using this model as a part of a larges net which also include translation into a different language (for example it can be use in order to let people that speaks different language to be able to communicate).
3. use this model in order to translate words and sentences.
4. build a model that can "translate" from video (clip the video into picture and the use our model/ use yolo, ect..)
5. you can use this model to use it in different image prossesing and labeling as you wish. (remeber to change the num class, the data+label) 

# References:

We took our dataset from:

https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out?resource=download

The pretrain model we took from VGG19:

The keras model we took from:
https://www.kaggle.com/code/brussell757/american-sign-language-classification

We also use many of the code and data that in the course material 


