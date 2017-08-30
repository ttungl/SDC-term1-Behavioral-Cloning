### SDC-term1
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
    
    Tung Thanh Le
    ttungl at gmail dot com
   
**Behavioral Cloning Project**
---

#### Project description: 
+ Use Deep Learning to Clone Driving Behavior.
+ The results are below with [video demo](https://youtu.be/xCkk7keDe5w).

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/bridge1.gif" height="149" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/curve1.gif" height="149" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/view1.gif" height="149" width="270">

#### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Build, a convolution neural network (CNN) in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results with a written report.

This implementation followed the [rubric points](https://review.udacity.com/#!/rubrics/432/view). The details will be explained in the next sections.  

#### My submission includes the required files: 
* [model.py](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/model.py): script used to create and train the model.
* [drive.py](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/drive.py): script to drive the car.
* [model.h5](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/model.h5): a trained Keras model.
* [a report](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/writeup.md).
* [video.mp4](https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/video.mp4): a video recording of my vehicle driving autonomously around the track for two full laps.

#### Quality of Code

The model provided can be used to successfully operate the simulation.

The code in `model.py` uses a Python generator, to generate data for training rather than storing the training data in memory. The `model.py` code is clearly comments where needed. 

The model uses severval methods for images processing and use a modified convolutional neural network from [nVidia architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

* `get_log()`: uses to read the data log from `driving_log.csv`, then shuffling the datasets, splits into the training sets and validation sets with the ratio of `80`:`20`. `random_state` is used for [initializing internal random number generator](https://stackoverflow.com/a/42197534/2881205), which decides the splitting of data into train and test indices. Finally, it returns the training set and validation set.

I read the data input using openCV library `cv2.imread`, so the image is BGR color as on the left, and on the right, the image is converted to RGB:

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/bgr_image_input.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/rgb_image_input.png" height="144" width="270"> 


The input images (center, left, right)

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/center_output.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/left_output.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/right_output.png" height="144" width="270">

* `brightness_process(image)`: uses to process the brightness of the image. The images after processing are as follows.

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/brightness1.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/brightness2.png" height="144" width="270"> 

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/brightness3.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/brightness4.png" height="144" width="270"> 

* `shadow_augmentation(image)`: uses to create the shadow for the images. This method has been inspired from [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). The basic idea is to create the random shadows to mask on the images. The output of this method is as below.

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/shadow1.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/shadow2.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/shadow4.png" height="144" width="270"> 

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/shadow6.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/shadow8.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/shadow9.png" height="144" width="270"> 

* `process_images(center_path, left_path, right_path, steering, images, steering_set)`: uses to process the images input on the center, left, and right angles. If the steering angle is greater than the steering threshold, it will flip the image to recover into the center track. After playing a couple of experiments with the tuning parameters, I fixed my tuning parameters with `steering_correction` = `0.15` and `steering_threshold` = `0.285` for the best performance of my model. 

The left side of the car is flipped to its right side when the steering angle is greater than the steering_threshold.

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/right_input.png" height="144" width="270"> <img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/right_input_flipped.png" height="144" width="270">

* `generators(datasets, batch_size=batch_size)`: uses a [data generator](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/) to work with large amount of data for more memory-efficient. 

I used the [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity for training my model. The car simulator is supported for [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip), [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip), and [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip). I have also collected the data from simulator, but it seems like the model works better with the udacity's data. 

#### Model Architecture and Training Strategy

+ The model architecture is inherited from nvidia architecture which has been verified for autonomous driving vehicles. 

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/images_output/nvidia_model.jpg" height="312" width="270">

+ I modified the model architecture as below, with 5 convolutional layers and 5 Fully-connected layers.   

| Layer (type)                  |  Output Shape      | Param#|    Connected to     |
|:-----------------------------:|:------------------:|:-----:|:-------------------:|
|lambda_1 (Lambda)              | (None, 160, 320, 3)| 0     |lambda_input_1[0][0] |           
|cropping2d_1 (Cropping2D)      | (None, 65, 320, 3) | 0     |lambda_1[0][0]       |           
|convolution2d_1 (Convolution2D)| (None, 31, 158, 24)| 1824  |cropping2d_1[0][0]   |           
|convolution2d_2 (Convolution2D)| (None, 14, 77, 36) | 21636 |convolution2d_1[0][0]|           
|convolution2d_3 (Convolution2D)| (None, 5, 37, 48)  | 43248 |convolution2d_2[0][0]|           
|convolution2d_4 (Convolution2D)| (None, 3, 35, 64)  | 27712 |convolution2d_3[0][0]|           
|convolution2d_5 (Convolution2D)| (None, 1, 33, 64)  | 36928 |convolution2d_4[0][0]|           
|flatten_1 (Flatten)            | (None, 2112)       | 0     |convolution2d_5[0][0]|           
|dense_1 (Dense)                | (None, 100)        | 211300|flatten_1[0][0]      |           
|dropout_1 (Dropout)            | (None, 100)        | 0     |dense_1[0][0]        |           
|dense_2 (Dense)                | (None, 50)         | 5050  |dropout_1[0][0]      |           
|dropout_2 (Dropout)            | (None, 50)         | 0     |dense_2[0][0]        |           
|dense_3 (Dense)                | (None, 20)         | 1020  |dropout_2[0][0]      |           
|dropout_3 (Dropout)            | (None, 20)         | 0     |dense_3[0][0]        |           
|dense_4 (Dense)                | (None, 10)         | 210   |dropout_3[0][0]      |           
|dense_5 (Dense)                | (None, 1)          | 11    |dense_4[0][0]        |           

+ As can be seen in the table, the model uses dropout layers (=0.5) to prevent overfitting, and the train/validation/test splits have been used in `get_log()`. The model parameters for tuning are `batch_size=64` and number of epochs `num_epoch=20`. I used the `Adam` optimizer for the model. 

##### Train the model strategy

+ At first, I used the model with a flatten and a fully-connected layer, and the driving was bad, the car went off the track and sinked to the lake at the started point. Then, I used the upgraded LeNet model built in the last project, it got better but still running off the road if the steering angle is too high. After that I replaced that model using Nvidia model. 

+ This time, the driving got better, but still went off the road when it crosses the shadow or brightness areas. So I used brightness method and shadow augmentation to train the model to deal with these issues. The driving got better when crossing the shadow and bridge areas.


+ My model has been trained using AWS EC2 from Amazon. After I launched an GPU instances, it contains the IP address that helps me to access to that for training the model.
```
* Copy from local (model.py and data.zip) to AWS
	+ `scp data.zip carnd@54.119.111.11:.`
* Copy from AWS to local machine:
	+ `scp carnd@54.119.111.11:model.h5 .`
```
+ Note: IP address is from running the GPU instances. 
```
* To access to the AWS, using commands: 
	+ `ssh carnd@54.119.111.11`
	+ `source activate carnd-term1`
	+ `python model.py`
```
* After running the `model.py`, the result is expected as below.
```
(carnd-term1) carnd@ip-172-31-13-214:~$ python model.py 
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Epoch 1/20
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GRID K520
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:00:03.0
Total memory: 3.94GiB
Free memory: 3.91GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: 0000:00:03.0)
19158/19264 [============================>.] - ETA: 0s - loss: 0.0301/home/carnd/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/keras/engine/training.py:1569: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
19371/19264 [==============================] - 113s - loss: 0.0300 - val_loss: 0.0210
Epoch 2/20
19446/19264 [==============================] - 102s - loss: 0.0230 - val_loss: 0.0187
Epoch 3/20
19410/19264 [==============================] - 101s - loss: 0.0205 - val_loss: 0.0179
Epoch 4/20
19443/19264 [==============================] - 103s - loss: 0.0199 - val_loss: 0.0173
Epoch 5/20
19269/19264 [==============================] - 102s - loss: 0.0194 - val_loss: 0.0167
Epoch 6/20
19452/19264 [==============================] - 100s - loss: 0.0188 - val_loss: 0.0177
Epoch 7/20
19461/19264 [==============================] - 102s - loss: 0.0189 - val_loss: 0.0159
Epoch 8/20
19455/19264 [==============================] - 104s - loss: 0.0181 - val_loss: 0.0175
Epoch 9/20
19440/19264 [==============================] - 101s - loss: 0.0178 - val_loss: 0.0172
Epoch 10/20
19461/19264 [==============================] - 102s - loss: 0.0177 - val_loss: 0.0148
Epoch 11/20
19452/19264 [==============================] - 103s - loss: 0.0172 - val_loss: 0.0159
Epoch 12/20
19269/19264 [==============================] - 99s - loss: 0.0169 - val_loss: 0.0155
Epoch 13/20
19464/19264 [==============================] - 102s - loss: 0.0169 - val_loss: 0.0162
Epoch 14/20
19407/19264 [==============================] - 104s - loss: 0.0164 - val_loss: 0.0157
Epoch 15/20
19446/19264 [==============================] - 102s - loss: 0.0162 - val_loss: 0.0153
Epoch 16/20
19464/19264 [==============================] - 103s - loss: 0.0157 - val_loss: 0.0150
Epoch 17/20
19434/19264 [==============================] - 101s - loss: 0.0159 - val_loss: 0.0168
Epoch 18/20
19452/19264 [==============================] - 103s - loss: 0.0156 - val_loss: 0.0121
Epoch 19/20
19458/19264 [==============================] - 103s - loss: 0.0155 - val_loss: 0.0173
Epoch 20/20
19371/19264 [==============================] - 102s - loss: 0.0156 - val_loss: 0.0151
```

After training the model on AWS, I observed the relationship between the mean squared error (MSE) loss of training and validation datasets as below.

![][image22] 
This figure shows that the loss is small.

### Run the model on local machine

After copied the `model.h5` to my laptop (Macbook pro 16GB 1600MHz DDR3, 2.2 GHz Intel Core i7), running the simulation using command: `python drive.py model.h5`. 

### Solution

I modified the model based on Nvidia architecture by adding more dropout layers to prevent overfitting. To deal with a large amount of data, I used data generator to train the model much more memory-efficient. The images have also been processed using the brightness technique, shadow augmentation technique, and flip images technique with the openCV libraries. 

The result is expected as below with two full laps without waggling off the road.

<img src="https://github.com/ttungl/SDC-term1-Behavioral-Cloning/blob/master/gifs/view1.gif" height="149" width="270">

### Conclusion

The model has successfully been trained and tested on the track one of the simulator with the autonomous mode without falling out of the track. There still have something that need to be improved such as the driving becomes waggling at the start when I speed up to over 20 mph. I probably need to implement an accelerator which allows the car increases the speed slowly, then speed up when the car stays on the center. Another thing is that the model cannot be able to handle the challenge track (jungle one). To improve this, I think it'd be better to integrate the finding line detection to keep the car on the road.

---

