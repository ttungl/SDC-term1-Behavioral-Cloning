### SDC-term1
    
    Tung Thanh Le
    ttungl at gmail dot com
   
**Behavioral Cloning Project**
---

#### Project description: 
+ Use Deep Learning to Clone Driving Behavior.
+ The results are below with [video demo](https://youtu.be/xCkk7keDe5w)

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/bridge1.gif" height="149" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/curve1.gif" height="149" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/view1.gif" height="149" width="270">

#### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Build, a convolution neural network (CNN) in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results with a written report.

[//]: # (Image References)

[image5]: ./images_output/center_output.png "center image"
[image6]: ./images_output/left_output.png "left image"
[image7]: ./images_output/right_output.png "right image"

[image8]: ./images_output/bgr_image_input.png "bgr"
[image9]: ./images_output/rgb_image_input.png "rgb output"

[image10]: ./images_output/brightness1.png "brightness 1"
[image11]: ./images_output/brightness2.png "brightness 2"
[image12]: ./images_output/brightness3.png "brightness 3"
[image13]: ./images_output/brightness4.png "brightness 4"

[image14]: ./images_output/shadow1.png "shadow 1"
[image15]: ./images_output/shadow2.png "shadow 2"
[image16]: ./images_output/shadow4.png "shadow 4"
[image17]: ./images_output/shadow6.png "shadow 6"
[image18]: ./images_output/shadow8.png "shadow 8"
[image19]: ./images_output/shadow9.png "shadow 9"

[image20]: ./images_output/right_input.png "right input flip"
[image21]: ./images_output/right_input_flipped.png "flipped to left"

[image22]: ./images_output/loss_valid.png "MSE loss"


This implementation followed the points of [rubric points](https://review.udacity.com/#!/rubrics/432/view). The details will be explained in the next sections.  

### My submission includes the required files: 
* [model.py](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/model.py) script used to create and train the model.
* [drive.py](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/drive.py) script to drive the car.
* [model.h5](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/model.h5) a trained Keras model.
* [a report](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/writeup.md) writeup file.
* [video.mp4](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/video.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap).

### Quality of Code

The model provided can be used to successfully operate the simulation.

The code in model.py uses a Python generator, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

### Model Architecture and Training Strategy

Has an appropriate model architecture been employed for the task?

The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

Has an attempt been made to reduce overfitting of the model?

Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.

Have the model parameters been tuned appropriately?

Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

Is the training data chosen appropriately?

Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

Architecture and Training Documentation

### Solution

Is the solution design documented?

The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

Is the model architecture documented?

The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Is the creation of the training dataset and training process documented?

The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.

### Conclusion

Is the car able to navigate correctly on test data?

No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). 




<!-- ![][image5] -->


<!-- 
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary. -->
