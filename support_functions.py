import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import os
from sklearn.utils import shuffle

def read_shuffle_data(train_folder):
    """ Reads png image files from a folder holding the train/test data in png format that is separated in a vehicles and non-vehicles subfolder
    Params:
      train_folder: Folder with subfolders vehicles and non-vehicles that hold the training data
    Returns:
      cars: list with training images of cars
      non_cars: list with training images of non-cars
    """
    car_folder = train_folder + '/vehicles'
    non_car_folder = train_folder + '/non-vehicles'

    # Finds all png files in all subdirectories of the folder
    # Adapted from Martijn-Pieters on stackoverflow.com
    cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(car_folder) for f in files if f.endswith('.png')]
    non_cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(non_car_folder) for f in files if f.endswith('.png')]

    print('Number of car images in dataset:', len(cars))
    print('Number of non-car images in dataset:', len(non_cars))

    # Shuffling the datasets before feature extraction
    cars = shuffle(cars)
    non_cars = shuffle(non_cars)
    return cars, non_cars

def convert_color(img, color_space):
    """ Converts image from one color space to another 
    Params:
      img: Image in one color space
      color_space: OpenCV color space conversion
    Returns:
      feature_image: image in transformed color space
    Description of color spaces used below:
      RGB: Red Green Blue
      BGR: Blue Green Red
      HSV: Hue Saturation Value
      HSL: Hue Saturation Lightness
      YUV: Luminance Y, Chrominance U and V
      YCrCb: Luminance Y, Blue-Yellow Chrominance Cb, Red-Green Chrominance Cr
    """
    feature_image = np.copy(img)
    if color_space == 'RGB2HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        print('Picture converted from RGB to HSV')
    if color_space == 'HSV2RGB':
        feature_image = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    elif color_space == 'RGB2LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'LUV2RGB':
        feature_image = cv2.cvtColor(img, cv2.COLOR_LUV2RGB)
    elif color_space == 'RGB2HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'HLS2RGB':
        feature_image = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    elif color_space == 'BGR2YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YUV2BGR':
        feature_image = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    elif color_space == 'RGB2YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'BGR2YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'YCrCb2RGB':
        feature_image = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    elif color_space == 'YCrCb2BGR':
        feature_image = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return feature_image


def get_hog_features(img, HOG_params, vis=False, feature_vec=True):
    """ Returns HOG features from an image
    Params:
      img: Input image
      orient:
      pix_per_cell:
      cell_per_block:
      vis:
      feature_vec:
    Returns:
      features: HOG features of the input image
    """

    orient, pix_per_cell, cell_per_block = HOG_params
    HOG_block_norm = 'L2-Hys'

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(image=img, orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), block_norm = HOG_block_norm, visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(image=img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), block_norm = HOG_block_norm,  visualize=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, spatial_bin_size=(32, 32)):
    """ Concatenates all color channels of the image in one long vector
    Params:
      img: Input image
      spatial_bin_size: Size of bin, if image size != size, image will be resized
    Returns:
      vector with stacked color channels
    """
    color1 = cv2.resize(img[:,:,0], spatial_bin_size).ravel()
    color2 = cv2.resize(img[:,:,1], spatial_bin_size).ravel()
    color3 = cv2.resize(img[:,:,2], spatial_bin_size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, hist_nbins = 32):
    """ Calculates a color histogram of an image
    Params:
      img: Input image
      nbins: number of histogram bins
    Returns:
      hist_features: histogram of each color channel of the image concatenated
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=hist_nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=hist_nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=hist_nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features_path(imgs, color_space, HOG_params, spatial_size=(32, 32), hist_nbins = 32, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """ Computes color histograms and spatial binning on a list of images
    Params:
      imgs: List of input images
      color_space: OpenCV color space conversion
      spatial_size: Size of spatial bins
      hist_bins: # of color histogram bins
      orient:
      pix_per_cell:
      cell_per_block:
      hog_channel:
      spatial_feat: boolean, calculate spatial binning or not
      hist_feat: boolean, calculate color histogram or not
      hog_feat: boolean, calculate HOG features or not
    Returns:
      features: concatenated list of features
    """

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    imgs_length = len(imgs)   
    i = 0 
    for img in imgs:
        if i % 1000 == 0:
            print("Extracting features {0} of {1}".format(i, imgs_length))
        img_features = []
        # Read in each one by one
        image = cv2.imread(img)

        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, color_space[0])

        if spatial_feat == True:
            # Performs spatial binning on the image, with a given size
            spatial_features = bin_spatial(feature_image, spatial_bin_size = spatial_size)
            img_features.append(spatial_features)
        if hist_feat == True:
            # Apply a histogram on the three image color channels with nbins amount of bins
            hist_features = color_hist(img = feature_image, hist_nbins = hist_nbins)
            img_features.append(hist_features)
        if hog_feat == True:
            # Performs a Histogram of oriented gradients, and appends the features
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], HOG_params, vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], HOG_params, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            img_features.append(hog_features)
        features.append(np.concatenate(img_features))
        i = i + 1

    return features


def find_cars(img, y_start_stop, window_sizes, svc, X_scaler, color_space, HOG_params, spatial_size, hist_bins, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel='ALL'):
    """ Performs HOG on the whole image, then subsamples this image for different search windows, extracts the feature vector and predicts car or no car for this search window
    Params:
      img: Input image
      ystart: 
      ystop: 
      window_sizes: 
      svc: Handle to support vector machine
      X_scaler:
      color_space:
      orient:
      pix_per_cell: 
      cell_per_block:
      spatial_size:
      hist_bins:
      spatial_feat: boolean, calculate color histogram or not
      hog_feat: boolean, calculate HOG features or not
    Returns:
      features: concatenated list of features
    """
    # Search only in area of the picture from ystart to ystop. Exclude e.g. horizon
    y_start, y_stop = y_start_stop
    img_tosearch = img[y_start:y_stop,:,:]
    img_tosearch = convert_color(img_tosearch, color_space[0])
    bboxes = []

    orient, pix_per_cell, cell_per_block = HOG_params
    
    # 64 was the orginal sampling rate (window_sizes[0]), with 8 cells and 8 pix per cell
    for window in window_sizes:
        scale_factor = window/window_sizes[0]
        ctrans_tosearch = cv2.resize(img_tosearch, (np.int(img_tosearch.shape[1]/scale_factor), np.int(img_tosearch.shape[0]/scale_factor)))
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, HOG_params, feature_vec=False)
        hog2 = get_hog_features(ch2, HOG_params, feature_vec=False)
        hog3 = get_hog_features(ch3, HOG_params, feature_vec=False)
        nblocks_per_window = (window_sizes[0] // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                features = []
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                
                if hog_channel == 'ALL':
                    hog_features = np.concatenate((hog_feat1, hog_feat2, hog_feat3))
                elif hog_channel == 1:
                    hog_features = hog_feat1
                elif hog_channel == 2:
                    hog_features = hog_feat2
                elif hog_channel == 3:
                    hog_features = hog_feat3
    
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window_sizes[0], xleft:xleft+window_sizes[0]], (64,64))
                # First feature: Spatial binning
                if spatial_feat == True:
                    spatial_features = bin_spatial(subimg, spatial_bin_size=spatial_size)
                    features.append(spatial_features)
                # Second feature: Histogram of colors
                if hist_feat == True:
                    hist_features = color_hist(subimg, hist_nbins=hist_bins)
                    features.append(hist_features)
                
                if hog_feat == True:
                    features.append(hog_features)
                
                features = np.concatenate(features)

                # Scale features and make a prediction
                test_features = X_scaler.transform(features.reshape(1,-1))   
                test_prediction = svc.predict(test_features)
                
                # If a car is found, the bounding box shall be returned, but first needs to be transformed back to the original coordinates
                if test_prediction == 1:
                    left = int(xleft*scale_factor)
                    top = int(ytop*scale_factor)+y_start
                    bboxes.append(((left, top), (left+window, top+window)))
    return bboxes


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(255, 0, 0), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap_deque, bboxes, threshold, size):
    heatmap_current = np.zeros(size)
    for box in bboxes:
        heatmap_current[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap_deque.append(heatmap_current)
    heatmap_current = np.zeros(size)
    for heatmap in heatmap_deque:
        heatmap_current += heatmap
    heatmap_current[heatmap_current < threshold] = 0
    return heatmap_deque, heatmap_current

def draw_heatmap(heatmap_current):
    max = np.max(heatmap_current)
    heatmap_draw = np.uint8(heatmap_current*255/max)
    heatmap_draw = np.dstack((heatmap_draw, heatmap_draw, heatmap_draw))
    heatmap_draw = cv2.cvtColor(heatmap_draw, cv2.COLOR_RGB2GRAY)
    heatmap_draw = cv2.applyColorMap(heatmap_draw, cv2.COLORMAP_HOT)
    return heatmap_draw

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img