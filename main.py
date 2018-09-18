import numpy as np
import time
import pickle
from os.path import isfile
import cv2
import glob
from collections import deque
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from support_functions import extract_features_path, draw_boxes, find_cars, add_heat, draw_labeled_bboxes, draw_heatmap, read_shuffle_data


def train_SVM(train_folder, svm_params_file, color_space, spatial_size, hist_bins, HOG_params, hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True):
    """ Trains a support vector machine to classify car and non-car images
    Params:      
    Returns:
      None (SVM parameters are stored in a file svm_params.p)
    """
    # Read training data
    cars, non_cars = read_shuffle_data(train_folder)
    
    print("Feature extraction of car images...")
    car_features = extract_features_path(imgs=cars, color_space=color_space,HOG_params = HOG_params, spatial_size=spatial_size, hist_nbins=hist_bins, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    print("Feature extraction of non-car images...")
    non_car_features = extract_features_path(imgs=non_cars, color_space = color_space, HOG_params = HOG_params, spatial_size = spatial_size, hist_nbins = hist_bins, hog_channel = hog_channel, spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)
    
    X = np.vstack((car_features, non_car_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Label vector y
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    orient, pix_per_cell, cell_per_block = HOG_params
    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC (Support Vector Classifier)
    svc = LinearSVC()

    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # Writes svm classification to file
    f = open(svm_params_file, "wb")
    pickle.dump([svc, X_scaler], f)
    f.close()

    return svc, X_scaler

def pipeline_image(image_file, y_start_stop, search_windows, svc, X_scaler, color_space, HOG_params, hog_channel, spatial_size, hist_bins, spatial_feat=True, hist_feat=True, hog_feat=True):
    """ Runs a classifier on a single image
    Params:      
    Returns:
      None (Creates four images with bounding boxes, heatmap, labels)
    """  
    image = cv2.imread(image_file)
    heatmap_deque = deque(maxlen=10)
    heatmap_size = [image.shape[0], image.shape[1]]
    heat_threshold = 1 # For single image needs to be set to 1

    bboxes = find_cars(img=image, y_start_stop=y_start_stop, window_sizes = search_windows, svc=svc, X_scaler=X_scaler, color_space=color_space, HOG_params=HOG_params, spatial_size=spatial_size, hist_bins=hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, hog_channel=hog_channel)

    heatmap_deque, heatmap_current = add_heat(heatmap_deque, bboxes, heat_threshold, heatmap_size)
    # Combines all spatially related heatmap points to one single region
    labels = label(heatmap_current)
    draw_img = draw_labeled_bboxes(image, labels)
    boxes_img = draw_boxes(image, bboxes)
    heatmap = draw_heatmap(heatmap_current)
    max_label = np.max(labels[0])
    norm_label = np.uint8(labels[0]*255/max_label)
    cv2.imwrite('Bounding_boxes_all.jpg', boxes_img)
    cv2.imwrite('Bounding_boxes_comb.jpg', draw_img)
    cv2.imwrite('Heatmap.jpg', heatmap)
    cv2.imwrite('Labels.jpg', norm_label)

def pipeline_video(file_in, file_out, y_start_stop, search_windows, svc, X_scaler, color_space, HOG_params, hog_channel, spatial_size, hist_bins, spatial_feat=True, hist_feat=True, hog_feat=True):
    """ Runs a classifier on a video file
    Params:
      file: Input video file path/filename      
    Returns:
      None (Creates four images with bounding boxes, heatmap, labels)
    """  

    # Read in a video file with all properties like frame rate, shape and length
    capture = cv2.VideoCapture(file_in)
    frame_length =  int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height =    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width =     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_fps =       int(capture.get(cv2.CAP_PROP_FPS))
    print ('Video length is ', frame_length, ' frames, size is ', vid_width, 'x', vid_height, 'Frame rate is', vid_fps, 'fps')
    # Set output video codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outvideo = cv2.VideoWriter(file_out, fourcc, vid_fps, (vid_width, vid_height))
    if outvideo.isOpened():
        print('Video could be opened correctly')
    else:
        print('Video could not be opened. Exit')
        return

    heatmap_deque = deque(maxlen=10)
    heatmap_size = [vid_height, vid_width]

    for i in range(frame_length):
        ret, img = capture.read()
        if ret == True:
            # Searches for detections within the current video frame 'img'
            bboxes = find_cars(img=img, y_start_stop=y_start_stop, search_windows=search_windows, svc=svc, X_scaler=X_scaler, color_space=color_space, HOG_params=HOG_params, spatial_size=spatial_size, hist_bins=hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, hog_channel=hog_channel)
            heatmap_deque, heatmap_current = add_heat(heatmap_deque, bboxes, heat_threshold, heatmap_size)
            #cv2.imwrite('./output_images/heatmap'+str(i)+'.jpg', bgr_heatmap)
            labels = label(heatmap_current)
            draw_img = draw_labeled_bboxes(img, labels)
            boxes_img = draw_boxes(img, bboxes)
            cv2.imwrite('./output_images/bboxes_all_'+str(i)+'.jpg', boxes_img)
            outvideo.write(np.uint8(draw_img))
            print('Frame', i)
        else:
            break

    capture.release()
    outvideo.release()

def main():

    classify_video = True
    # Re-triggers training of SVM classifier (e.g. after parameter change)
    train_again = False
    train_folder = "./train_images"
    # Stores SVM parameters in a file, or loads from this file if it exists
    svm_params_file = './svm_params.p'

    color_space = ['BGR2YCrCb','YCrCb2BGR'] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations (directions) per cell (typically between 6 and 12), with 12 orientations you get 30° orientation bins
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    HOG_params = orient, pix_per_cell, cell_per_block

    hog_channel = "ALL" # of the three color channels, which one is used in HOG calculation? - can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions - averages a color channel across this size
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    # Threshold for # of detection (individual sliding windows over the last 10 frames) needed to classify as a car. Higher number avoids false positives, but increases risk to lose detection
    heat_threshold = 25

    # sizes of sliding windows (squared, only one dimension given)
    search_windows = (64,96,128,192) 

    # Min and max position in y direction for feature extraction. 
    # Only the area in which cars are to be expected needs to be searched
    y_start_stop = [399, 719]

    if (train_again == True) or (isfile(svm_params_file) == False):
        svc, X_scaler = train_SVM(train_folder=train_folder, svm_params_file=svm_params_file, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, HOG_params=HOG_params, hog_channel=hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True)
    else:
        f = open(svm_params_file, "rb")
        [svc, X_scaler] = pickle.load(f)
        f.close

    if classify_video:
        video_file_in = 'project_video.mp4'
        video_file_out = 'project_output.mp4'
        pipeline_video(file_in=video_file_in, file_out=video_file_out, y_start_stop=y_start_stop, search_windows=search_windows, svc=svc, X_scaler=X_scaler, color_space=color_space, HOG_params=HOG_params, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    else:
        image_file = './test_images/test3.jpg'
        pipeline_image(image_file=image_file, y_start_stop=y_start_stop, search_windows=search_windows, svc=svc, X_scaler=X_scaler, color_space=color_space, HOG_params=HOG_params, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)


if __name__ == "__main__":
    main()