# **Behavioral Cloning** 
---

**Introduction**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/Combined_Camera.png "Camera Images"
[image2]: ./Images/Flipped_Images.png "Images Flipping"
  
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* WriteUp.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network as `model.h5`. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The overall strategy for deriving a model architecture was to find out the simplest known network that can produce the best results for the training data set

My first step was to use a convolution neural network model similar to the previously used LeNet frameowork. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model only got me so far in terms of a good fit. To that purpose, we modified the network to the Nvidia Neural network. This indeed worked better but there were symtopms of overfitting like lesser loss on the training set vs the valdation set.

To combat the overfitting, I modified the model so that each convolution layer was followed by a drop out layer. This provides a much more acceptable loss when comparing the training set and the validation sets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve this, I augmented the date by flipping the dataset and also adding more date by driving in the clockwise direction

My final model consisted of the following layers:

| Layer         		| Description	        		| 
|:---------------------:|:-----------------------------:| 
| Normalised Input      | 160x320x3 RGB image           |
| Cropping 2D	      	| 70,25 Crop 		            |
| Convolution 2D     	| 2x2 stride ,	24X5x5 Filter   |     
| DropOut				| Keep probability 0.5	        |
| RELU					|							    |
| Convolution 2D     	| 2x2 stride ,	36X5x5 Filter   |      
| DropOut				| Keep probability 0.5	        |   
| RELU					|								|		
| Convolution 2D     	| 2x2 stride ,	48X5x5 Filter   |      
| DropOut				| Keep probability 0.5	        |
| RELU					|                               |
| Convolution 2D     	| 1x1 stride ,	64X3x3 Filter   |      
| DropOut				| Keep probability 0.5	        |
| RELU					|						        |
| Convolution 2D     	| 1x1 stride ,	24X3x3 Filter   |      
| DropOut				| Keep probability 0.5	        |
| RELU                  |                               |
| Flatten	            | Outputs 400                   |
| Fully connected	    | Outputs 100                   |
| Fully connected	    | Outputs 50                    |
| Fully connected	    | Outputs 10                    |
| Fully connected	    | Outputs 1                     |

The model includes RELU layers to introduce nonlinearity and DropOut layers to prevent overfitting. The final network inputs a 160x320x3 RGB image and the output is a steering prediction At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 107, 112, 116, 120, 124). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 138).

#### 4. Appropriate training data

In order to ensure that the car drives down the center of the road, it's essential to capture center lane driving. I recorded one lap on track one using center lane driving. Then I repeated this process on the same track but in the clockwise direction in order to get more data points. Each snapshot has 3 images which comes from the left camera, centre camera and right camera. 

![alt text][image1]

To augment the data set, I also flipped images and angles thinking that this would allow the car to understand distances from the lane edges and how to correct for variations

![alt text][image2]

After the collection process, I had a total of 32016 images. When reading the `driving_log.csv` file each row has ‘center, left, right, steering, throttle, brake, speed’ columns. In this project, I use the camera images (center, left, right) as input and steering as target. The rest are ignored. I then preprocessed this data by normalizing them around zero. I also cropped the top parts of the image to allow for easier fitting

I finally randomly shuffled the data set and put 20\% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss differential between the training and the validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.