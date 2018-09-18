[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# Vehicle detection
This project is part of the [Udacity Self-Driving Car Engineer Nanodegree](http://www.udacity.com/drive). It performs vehicle identification with boundary boxes in images and videos using "classical" computer vision techniques such as
- Support Vector Machines (SVM)
- Histogram of Oriented Gradients (HOG)
- Spatial Binning

# Requirements

## Data
A pre-trained classifier is included in this repo, but for changes in the parameters, you need to download training images such as [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [Kitti vision dataset](http://www.cvlibs.net/datasets/kitti/) -> object -> 2D object. Please unpack the images to a folder train_images, separated into vehicles and non-vehicles. An example video file can be downloaded [here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/test_video.mp4).

## Download and install
```
git clone https://github.com/jhallier/Vehicle-Detection
cd Vehicle-Detection
pip install -r requirements.txt
```

## Run
Run a sample visualization of the data and some transformations on the images
```
python data_exploration.py
```
Run a vehicle detection pipeline on a video image
```
python main.py
```

# Project details

The pipeline of the transformation is as follows:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Perform spatial color binning and histograms of color on the image, and append the result to the feature vector 
* Run sliding windows of different sizes across the image and generate a heatmap of each positive result
* Threshold the heatmap to prevent false positives and combine continuous heatmap regions to one detection result with a boundary box
* Run the pipeline on a video and save the output video file with the resulting boundary boxes drawn upon

[//]: # (Image References)
[image1]: ./examples/car_noncar.png
[image2]: ./examples/color_space_exploration_YCrCb.png
[image3]: ./examples/color_space_exploration_HLS.png
[image4]: ./examples/bboxes_example1.jpg
[image5]: ./examples/heatmap4.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Histogram of Oriented Gradients (HOG)

With the function extract\_features_path from support\_functions.py, the images are read in one by one and a feature vector is extracted. First, spatial binning is performed (if the flag is set), then color histogram, and then HOG features are extracted with the get_hog_features function. 

Below is an example of car and a non-car image from the dataset.

![alt text][image1]

From these car and non-car images, the color space is transformed and a HOG transformation applied to each channel. HOG calculates the main gradient direction in small parts of the image. The gradient is discretized in n orientations. For example, 9 orientations means the main direction of the gradient is calculated in steps of 45 degrees. This allows a generalization of a car image, where the shape may differ a little from image to image, but the rough proportions of the car are similar and distinct to other non-car images. 

Below is an example of the YCrCb color space, which shows that all three different channel show a slightly different HOG, but all are distinctly different from the non-car image. Channel 1 seems to be most interesting for the general shape of the car, and channel three for the color values, which can be used in the spatial color binning.

![alt text][image2]

As a comparison, here are the HOG transformations for the HLS color space, where channel 2 still catches some of the shape, but the other channels are inferior compared to YCrCb.

![alt text][image3]

 A linear SVM was trained using all three HOG channels combined, a spatial binning feature with 32x32 size, and color histogram with 32 bins. The feature vector is normalized with the sklearn function StandardScaler, which has the advantage that the same normalization can later be applied to the images from the video stream. 20% of the images are used as a test set and a linear SVM is trained to classify cars and non-cars.

## Example images

![alt text][image4]

## Heatmap example:

![alt text][image5]