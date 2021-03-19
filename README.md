# Cancer_detection_by_image_detection
CS909-A data mining project on calculating the number of certain type of cells within images to assist cancer detection.


In this assignment,the objective is to develop a regression model for calculating the number of certain type of cells
(called lymphocytes) in a given histopathology image patch. For this assignment, all you have to
know is that these cells appear in the given image (technically called a immunohistochemistry or IHC
image) with a blue nucleus and a brown membrane. Your task is to develop a machine learning
model that uses training data (patch images with given cell counts) to predict cell counts in test
images.


The data ‘breast.h5’ can be downloaded from: http://shorturl.at/fuCEO
The subset of the challenge dataset that you have been given focuses on breast tissue images from a
total of 18 different individuals. You can read the data as follows:


import h5py

import numpy as np

D = h5py.File('breast.h5', 'r')

X,Y,P = D['images'],np.array(D['counts']),np.array(D['id'])


Here, X, Y and P contain the Images, Cell Counts, and Patient IDs, respectively.

Training and Testing: Use data from patient IDs 1-13 for training and cross validation and 14-18 for
testing. Be sure not to test on the images of patients you have used in your training. Each image is in
RGB space so it is represented by an array of size 299x299x3 where the first two dimensions
correspond to the width and height of the image and the last three correspond to the R,G and B
channel.

This project mainly includes three parts, including:

1. Showing the data
2. Extracting the feature and use them for predictions by applying Classical Regressions
3. Applying Convolutional Neural Networks to solve the problem which exactly the same was as in part 2



