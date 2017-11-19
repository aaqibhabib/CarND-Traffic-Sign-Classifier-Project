# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./street-view-signs/1.PNG "Traffic Sign 1"
[image5]: ./street-view-signs/2.PNG "Traffic Sign 2"
[image6]: ./street-view-signs/3.PNG "Traffic Sign 3"
[image7]: ./street-view-signs/4.PNG "Traffic Sign 4"
[image8]: ./street-view-signs/5.PNG "Traffic Sign 5"
[image9]: ./examples/soft-max-distributions.JPG "Soft Max Distributions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/aaqibhabib/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Python `collections.Counter` to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images part of the training set for each classification value. As you can see, there are some classes with many examples, while some are rare and some have no training examples. This is far from ideal because our model will perform better at classifying signs that have many instances and perform poorly on signs with a limited dataset.

![Training Dataset][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this removes the element of color and its impact on the classifier. The model will look for shapes and letters which is expected to be enough for the desired accuracy. Given a larger training set with more examples for each class, we could train using the color images and potentially achieve greater accuracy. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because some images were dark and some were bright. Using the OpenCV Histogram normalization function, we increased the image range to use the full 0-255  pixel values. Normalizing increased the visibility and allowed the model to train using similar weighted images.

I decided to not generate additional data because for this at this time. Ideally I would augment the dataset by adding noise, skewing, and rotating the images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented the LeNet-5 architecture with dropouts. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 3x3     	| 1x1 stride, 'Valid' padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3     	| 1x1 stride, 'Valid' padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten				|            									|
| Fully Connected		|         									    |
| RELU					|												|
| Dropout				| 50% Keep Probability							|
| Fully Connected		|            									|
| RELU					|												|
| Dropout				| 50% Keep Probability							|
| Fully Connected		| 43 Classifications 							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters:
* Number of epochs = 45
* Batch Size = 128
* Learning Rate = 0.01
* Optimizer = Adam Algorithm (Uses back propagation to train the network and minimize training loss)
* Dropout = 50% Keep Probability (Only applicable when training)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of: 0.976
* validation set accuracy of: 0.938
* test set accuracy of: 0.908

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I first used the LeNet-5 project as described in the Udacity Lab. Since that performed well for recognizing handwritten numbers and letters, I figured it was a good starting point for traffic sign classifications. I trained using a 0.01 training rate without changing the architecture an using the color image patches and was able to achieve ~0.90 accuracy in teh training set. In order to improve this, I added a dropout layers after each activation for the fully connected layers. This increased a my test accuracy minimally. Then, I experimented with decreasing the learning rate and increasing the total number of epochs which had a noticeable impact on the test accuracy. Finally, I preprocessed the data by grayscaling the images and normalizing the histogram which ultimately allowed me to reach a training accuracy >0.93.
 
There is a real concern of overfitting. You can see my training accuracy is slightly higher than both the validation set and test set. I combated overfitting by lowering the number of epochs used to train the model. Going from 50 to 45 epochs allowed me to reduce how overfitted the model became. Adding dropouts helped when I increased the number of epoch.

Overall, the test, train, and validation accuracies are similar meaning the model is trained relatively well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Below are five German traffic signs that I found on the web.

![Right of way (11)][image4] ![Keep Right (38)][image5] ![Yield (13)][image6] 
![No Entry (17)][image7] ![Speed Limit 50 (2)][image8]

The first image might be difficult to classify because of the the dark background and limited lighting and low resolution of the center figure. The fourth image is also dark and skewed. The skewness will not be an issue once the image is resized.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way      		| Right of way   									| 
| Keep Right     			| Keep Right 										|
| Yield					| Yield											|
| No Entry	      		| No Entry					 				|
| Speed Limit 50km/h			| Speed Limit 50km/h      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 90%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is fairly sure that this is a Right of Way sign (probability of 57%), and the image does contain a Right of Way sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 57%         			| Right of Way   									| 
| 28%     				| Priority road 										|
| 6%					| Pedestrians											|
| 5%	      			| Roundabout Mandatory					 				|
| 4%				    | End of no passing by vehicles over 3.5 metric tons      							|


Below is a visual showing each of the input and the top 5 guesses.
![Top 5 Softmax][image9]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


