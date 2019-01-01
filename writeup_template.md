# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/centered_image.jpg "Center Image"
[image2]: ./examples/centered_image_flip.jpg "Flipped Center Image"
[image3]: ./examples/leftside_image.jpg "Left Side Image"
[image4]: ./examples/leftside_image_flip.jpg "Flipped Left Side Image "
[image5]: ./examples/rightside_image.jpg "Right Side Image"
[image6]: ./examples/rightside_image_flip "Flipped Right Side Image"
[image7]: ./examples/cnn-architecture-624x890 "NVIDIA Network Architecture"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is inspired from the [NVIDIA]{https://devblogs.nvidia.com/deep-learning-self-driving-cars/} network of a convolution neural network. 

The network I have implemented uses 5 convolutional layers and 4 fully connected dense layers.(code line 79 to 95).
The model includes RELU layers to introduce nonlinearity (code line 83,85,88 and 89), and the data is normalized in the model using a Keras lambda layer (code line 81). 

#### 2. Attempts to reduce overfitting in the model

Two techniques are used in this model to overcome overfitting.
1. DropOut (code line - 84 & 86)
2. MaxPooling ( code line - 87)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100-101). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. As a starting point I used the sample data set given by Udacity in this course.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to enable the car to complete ateast one lap run in the given Udacity's simulator track.

My first step was to use a convolution neural network model similar to the [NVIDIA]{https://devblogs.nvidia.com/deep-learning-self-driving-cars/}. I thought this model might be appropriate because as discussed in the classroom, this model was implemented in a real car that ran considerable miles on the real roads till airport to and fro without touching the steering even once.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I have used two techniques
1. DropOut (code line - 84 & 86)
2. MaxPooling ( code line - 87)

As I inspected the training sample images, it was observed that though there is a considerable distance the car had covered while collecting the training data, the features or the road patterns were repeating frequently. Eg. The border lines with white and red and others with black and white were repeating more than twice. This helpmed me come to the conclusion that if we can apply dropouts in certain early learning layers, it wouldn't tamper the data much on the other hand we can attain a faster and accurate learning. Thus dropouts were introduced before 2nd ad 3rd layer.

Then on more analysis it was came to my notice that the camera images captured by the car camera had uniform surfaces for a particular instant and abrupt changes of surfaces were almost nil. This helped me to conclude that instead of feed forward each every convoltional layer to the next layer of the neural network, it would be sufficient to feed forward the Max valued or apply Max Pooling over the layers and pass them to the next layer. This has improved the speed of the training considerably.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I augmented data to add flipped images. This made the model train better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) is a sequential model consisted of 5 convolutional layers and 4 fully connected dense layers.(code line 79 to 95).
The model includes RELU layers to introduce nonlinearity (code line 83,85,88 and 89), and the data is normalized in the model using a Keras lambda layer (code line 81). 
As this model architecture was inspired from the NVIDIA network
Here is a visualization of the architecture.

![alt text][image7]

#### 3. Creation of the Training Set & Training Process
I have used the data provided by Udacity. As a pre-processing technique the images were cropped of less informative regions like the sky and tree to make it more precise on track details subjected for the purpose of driving. The images were also normalized so that there was a uniformity brought to the image's pixel intensity histograms.
Since model was not training well on initial dataset, I augmented the data by flipping each image vertically. Augmentation was done using a generator to speed up the training process. Code of generator function can be found in lines 24 to 70. The augmentation was done separately for centre images, left images and right images. The generator function was implemented so that the huge volume data was accessed only on-demand and wasn't stored unnecessarily that improved the performance. Before the augmentation, the images were converted to RGB scale and later flipped vertically. 

Here are examples of flipped image
For the centre image these are the examples of original and flippked image.
#### Centre (Original and Flipped Image)
![alt text][image1]![alt text][image2]

For the left side image these are the examples of original and flippked image.
#### Left (Original and Flipped Image)

![alt text][image3] ![alt text][image4]

For the left side image these are the examples of original and flippked image.
#### Right (Original and Flipped Image)

![alt text][image5] ![alt text][image6]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help the car in identifying how to get back to track.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by training. I used an adam optimizer so that manually training the learning rate wasn't necessary.
