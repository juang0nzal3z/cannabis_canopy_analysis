# cannabis_canopy_analysis
Analyses overhead images of cannabis canopies for various features.
The goal of this image analysis was to measure 2-dimensional canopy area and canopy fill. Canopy area is the area occupied by the plant canopy from a 2-dimensional overhead image. This measure is approximated by counting the number of pixels contained within the image and converting pixel number to cm2. Canopy fill is a ratio calculated by taking the contour area and subtracting any “white space” within the contour. Presumably the denser the canopy is, the less white space it will contain and the higher the canopy fill ratio will be. Images were analyzed using the python implementation of OpenCV. Images were saved in lossless .png format to maximize signal resolution. The image analysis consists of three steps: (i) calculation of pixels per metric, (ii) segmentation of plant canopy from image background, and (3) feature extraction from segmented canopy.  

Pixels per metric calculation was done by including a 20 cm ruler in every image. Then, GIMP was used to draw a solid colored line along the ruler and equal to its length. This rgb image was then converted into LAB color-space and the a* channel was the used. The a* channel is a grayscale image that contains intensity values from 0 to 255.   Segmenting the ruler line from the rest of the image requires finding an intensity threshold that can distinguish the background and plant canopy from the ruler line. The threshold value was calculated using a custom Otsu’s approach. We applied Otsu’s method and increased Otsu’s threshold by an intensity of 15, only keeping pixels above that threshold. We then filter the resulting binary image to extract only the largest connected component—the ruler line. The length of the ruler line is the measured in pixels. The pixels per metric is then calculated with the simple equation. This pixels per metric ratio is then used downstream to convert area into cm2. 

Segmentation of canopy from background was performed using a similar approach in the Blue channel. We apply Otsu’s method method and increased Otsu’s threshold by an intensity of 25. We then only keep pixels below that threshold (plant canopy) and remove any pixels above that threshold (white card-board background). We filter the resulting binary image from any pixels representing debri or noise by only keeping the largest connected component—the canopy. The resulting binary image is the canopy containing the ‘holes’ in the canopy. The area of this component is calculated by counting the number of nonzero pixels in the image. Then we calculate the canopy contour and fill the entire contour, effectively filling the holes. A second measure of area is taken with the same pixel counting method. Lastly, a fill ratio is calculated by dividing the canopy area with the filled canopy area.

Packages and functions:
import sys, traceback, os, re
import argparse
import string
import numpy as np
import cv2
import csv
import logging
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import math
