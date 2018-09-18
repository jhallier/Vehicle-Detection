from support_functions import get_hog_features, convert_color, read_shuffle_data
from random import sample
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from pylab import savefig


''' Selects one random example from each cars and non-cars and plots them '''
def plot_random_example(car_image, non_car_image, save=False):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(non_car_image)
    plt.title('Example Non-car Image')
    plt.show()
    if save:
        savefig('car_noncar.png', bbox_inches='tight')

''' Extract and plot color channel HLS and plot each channel and its respective Histogram of Oriented Gradients (HOG) transformation for a car and a non-car image'''
def plot_features(car_image, non_car_image, HOG_params, save=False):

    car_image = convert_color(car_image, 'RGB2HLS')
    non_car_image = convert_color(non_car_image, 'RGB2HLS')
    car_CH1 = car_image[:,:,0]
    car_CH2 = car_image[:,:,1]
    car_CH3 = car_image[:,:,2]
    non_car_CH1 = non_car_image[:,:,0]
    non_car_CH2 = non_car_image[:,:,1]
    non_car_CH3 = non_car_image[:,:,2]

    car_hog_features_CH1, car_hog_image_CH1 = get_hog_features(img=car_CH1, HOG_params=HOG_params, vis=True, feature_vec=True)

    non_car_hog_features_CH1, non_car_hog_image_CH1 = get_hog_features(img=non_car_CH1, HOG_params=HOG_params, vis=True, feature_vec=True)

    car_hog_features_CH2, car_hog_image_CH2 = get_hog_features(img=car_CH2, HOG_params=HOG_params, vis=True, feature_vec=True)

    non_car_hog_features_CH2, non_car_hog_image_CH2 = get_hog_features(img=non_car_CH2, HOG_params=HOG_params, vis=True, feature_vec=True)

    car_hog_features_CH3, car_hog_image_CH3 = get_hog_features(img=car_CH3, HOG_params=HOG_params, vis=True, feature_vec=True)

    non_car_hog_features_CH3, non_car_hog_image_CH3 = get_hog_features(img=non_car_CH3, HOG_params=HOG_params, vis=True, feature_vec=True)

    # Plot each, 2x car/non-car, 3 color channels, 2x non-HOG/HOG = 12 plots
    fig = plt.figure(figsize=(20,20))
    plt.subplot(3,4,1)
    plt.imshow(car_CH1)
    plt.title('Car image HLS CH-1')
    plt.subplot(3,4,2)
    plt.imshow(car_hog_image_CH1)
    plt.title('Car HOG HLS CH1')
    plt.subplot(3,4,3)
    plt.imshow(non_car_CH1)
    plt.title('Non-car HLS CH1')
    plt.subplot(3,4,4)
    plt.imshow(non_car_hog_image_CH1)
    plt.title('Non-car HOG HLS CH-1')
    plt.subplot(3,4,5)
    plt.imshow(car_CH2)
    plt.title('Car image HLS CH-2')
    plt.subplot(3,4,6)
    plt.imshow(car_hog_image_CH2)
    plt.title('Car HOG HLS CH-2')
    plt.subplot(3,4,7)
    plt.imshow(non_car_CH2)
    plt.title('Non-car HLS CH-2')
    plt.subplot(3,4,8)
    plt.imshow(non_car_hog_image_CH2)
    plt.title('Non-car HOG HLS CH-2')
    plt.subplot(3,4,9)
    plt.imshow(car_CH3)
    plt.title('Car image HLS CH-3')
    plt.subplot(3,4,10)
    plt.imshow(car_hog_image_CH3)
    plt.title('Car HOG HLS CH-3')
    plt.subplot(3,4,11)
    plt.imshow(non_car_CH3)
    plt.title('Non-car HLS CH-3')
    plt.subplot(3,4,12)
    plt.imshow(non_car_hog_image_CH3)
    plt.title('Non-car HOG HLS CH-3')
    plt.show()
    if save:
        savefig('color_space_exploration.png', bbox_inches='tight')


def main():

    train_folder = './train_images'

    orient=12
    pix_per_cell = 8
    cell_per_block = 2
    HOG_features = orient, pix_per_cell, cell_per_block

    cars, non_cars = read_shuffle_data(train_folder)
    
    # Randomly choose a car and non-car image
    car_image = mpimg.imread(sample(cars, 1)[0])
    non_car_image = mpimg.imread(sample(non_cars, 1)[0])

    # Plot chosen images and its color channels and HOG transformation
    plot_random_example(car_image, non_car_image)
    plot_features(car_image, non_car_image, HOG_features, save=False)

if __name__ == "__main__":
    main()

