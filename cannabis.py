#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###################################  BLUEBERRY   #########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################IMPORT PACKAGES AND FUNCTIONS#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
################################PARSE FUNCTION ARGUMENTS##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Parse command-line arguments
def options():
	parser = argparse.ArgumentParser(description="This script finds and 'cleans up' ears against dark background and find the largest square to calculate pixels per metric for size refference")
#Required
	parser.add_argument('-i', '--image', help="Input image file.", required=True)
#Optional main
	parser.add_argument('-o', '--outdir', help="Provide directory to saves proofs and pixels per metric csv. Default: Will not save if no output directory is provided")
	parser.add_argument('-p', '--proof', default=True, action='store_true', help="Save and print proofs. Default: True")
	parser.add_argument('-D', '--debug', default=False, action='store_true', help="Prints intermediate images throughout analysis. Default: False")
#Custom
	parser.add_argument('-ppm', '--pixelspermetric', nargs=1, type=float, help="Pixel per refference length to estimate the real length")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	args = parser.parse_args()
	return args
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################### BUILD LOGGER #######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_logger(logger_name):
	
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(logging.Formatter('%(asctime)s — %(levelname)s — %(message)s'))	
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG) # better to have too much log than not enough
	logger.addHandler(console_handler)
	
	args = options()
	if args.outdir is not None:
		destin = '{}'.format(args.outdir)
		if not os.path.exists(destin):
			try:
				os.mkdir(destin)
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise
		LOG_FILE = ('{}ear_CV.log'.format(args.outdir))
	else:
		LOG_FILE = 'ear_CV.log'
		
	file_handler = logging.FileHandler(LOG_FILE)	
	file_handler.setFormatter(logging.Formatter('%(asctime)s — %(levelname)s — %(message)s'))
# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = False
	return logger
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def max_cnct(binary):
# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
# extracts sizes vector for each connected component
	sizes = stats[:, -1]
#initiate counters
	max_label = 1
	max_size = sizes[1]
#loop through and fine the largest connected component
	for i in range(2, nb_components):
		if sizes[i] > max_size:
			max_label = i
			max_size = sizes[i]
#create an empty array and fill only with the largest the connected component
	cnct = np.zeros(binary.shape, np.uint8)
	cnct[output == max_label] = 255
#return a binary image with only the largest connected component
	return cnct										# Returns largest connected component
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def cnctfill(binary):
	# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
	# extracts sizes vector for each connected component
	sizes = stats[:, -1]
	#initiate counters
	max_label = 1
	if len(sizes) > 1:
		max_size = sizes[1]
	#loop through and fine the largest connected component
		for i in range(2, nb_components):
			if sizes[i] > max_size:
				max_label = i
				max_size = sizes[i]
	#create an empty array and fill only with the largest the connected component
		cnct = np.zeros(binary.shape, np.uint8)
		cnct[output == max_label] = 255
	#take that connected component and invert it
		nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(cnct), connectivity=8)
	# extracts sizes vector for each connected component
		sizes = stats[:, -1]
	#initiate counters
		max_label = 1
		max_size = sizes[1]
	#loop through and fine the largest connected component
		for i in range(2, nb_components):
			if sizes[i] > max_size:
				max_label = i
				max_size = sizes[i]
	#create an empty array and fill only with the largest the connected component
		filld = np.zeros(binary.shape, np.uint8)
		filld[output == max_label] = 255
		filld = cv2.bitwise_not(filld)
	else:
		filld = binary
	#return a binary image with only the largest connected component, filled
	return filld											# Fill in largest connected component
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def img_parse(fullpath):
	fullpath = fullpath
	root_ext = os.path.splitext(fullpath) 
	ext = root_ext[1]											
	filename = root_ext[0]										#File  ID
	try:
		root = filename[:filename.rindex('/')+1]
	except:
		root = './'
	try:
		filename = filename[filename.rindex('/')+1:]
	except:
		filename = filename
	return fullpath, root, filename, ext										# Parses input path into root, filename, and extension
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, 'euclidean')[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype='float32')										# Orders connected components from left ot right
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def visualize_colors(cluster, centroids):
# Get the number of different clusters, create histogram, and normalize
	labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
	(hist, _) = np.histogram(cluster.labels_, bins = labels)
	hist = hist.astype('float')
	hist /= hist.sum()
# Create frequency rect and iterate through each cluster's color and percentage
	rect = np.zeros((50, 300, 3), dtype=np.uint8)
	colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
	start = 0
	for (percent, color) in colors:
		end = start + (percent * 300)
		cv2.rectangle(rect, (int(start), 0), (int(end), 50), color.astype('uint8').tolist(), -1)
		start = end
	return rect
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################DEFINE MAIN FUNCTION#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def main():

	args = options()											# Get options
	log = get_logger('logger')									# Create logger
	#log.info(args)												# Print expanded arguments
	fullpath, root, filename, ext = img_parse(args.image)		# Parse provided path
	if args.outdir is not None:									# If out dir is provided
		out = args.outdir
	else:
		out = './'
	img=cv2.imread(fullpath)									# Read file in
	#log.info('[START]--{}--Starting analysis pipeline..'.format(filename)) # Log

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#####################################PREPROCESSING########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	ears = img.copy()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################################  Pixels per Metric ################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	if args.pixelspermetric is not None:
		log.info('[PPM]--{}--Calculating pixels per metric'.format(filename))
		PixelsPerMetric = None
		proof = img.copy()
		lab = cv2.cvtColor(proof, cv2.COLOR_BGR2LAB)
		lab[img == 0] = 0
		_,a,_ = cv2.split(lab)										#Split into it channel constituents
		mskd,_ = cv2.threshold(a[a !=  0],1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		th3 = cv2.threshold(a,mskd + 15,255,cv2.THRESH_BINARY)[1]
		th3 = max_cnct(th3)

		cnts = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		for cs in cnts:			
			rects = cv2.minAreaRect(cs)
			width_i = int(rects[1][0])
			height_i = int(rects[1][1])
			boxs = cv2.boxPoints(rects)
			boxs = np.array(boxs, dtype='int')			
			boxs1 = order_points(boxs)
			(tls, trs, brs, bls) = boxs
			(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
			(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
			(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
			(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
			if height_i > width_i:
				dBs = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs))
			else:
				dBs = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs))

		PixelsPerMetric = dBs / args.pixelspermetric[0]	
		#PixelsPerMetric = args.pixelspermetric[0]/ dBS
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PPM module Output
		if PixelsPerMetric is not None:
			log.info('[PPM]--{}--Found {} pixels per metric'.format(filename, PixelsPerMetric))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~If size reference found then remove it
			ears[th3 != 0] = 255
		else:
			log.warning('[PPM]--{}--No size refference found for pixel per metric calculation'.format(filename))
	else:
		log.info('[PPM]--{}--Pixels per Metric module turned off'.format(filename))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################################  Remove Background and make proof ################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	log.info('[SEG]--{}--Segmenting from background...'.format(filename))
	
	canna = ears.copy()
	#proof = ears.copy()


	b,_,_ = cv2.split(ears)
	hsv = cv2.cvtColor(ears, cv2.COLOR_BGR2HSV)
	hsv[ears == 0] = 0
	_,s,_ = cv2.split(hsv)											#Split into it channel constituents
	
	otsu1,th3 = cv2.threshold(b,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#####   decide on the range somewhere between minus  30 and 60 


	ears[b > otsu1+25] = [0,0,0] 
	
	#filt = s	#****************		####for picture 43 make sure you change this to blue!
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#otsu1,_ = cv2.threshold(filt,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#####   decide on the range somewhere between minus  30 and 60 
	#ears[filt < otsu1] = [0,0,0] 
	
	chnnl = cv2.cvtColor(ears,cv2.COLOR_RGB2GRAY)
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	chnnl = cv2.threshold(chnnl, 1,256, cv2.THRESH_BINARY)[1]
	lvs = max_cnct(chnnl)
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROOF 1
	proof[lvs == 0] = [0,0,0]
	canna[lvs == 0] = [0,0,0]	

	log.info('[SEG]--{}--Segmentation...DONE'.format(filename))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
################################## FULL CANNA FEATS ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	log.info('[FEAT]--{}--Extracting morphometric features...'.format(filename))

#empty list of feats
	ar = tot_lf = tot_les = Fill_rat = Ear_Box_Width = Ear_Box_Length = Ear_Box_Area = perimeters = Convexity = Solidity_Hull = Solidity_Box = ori = ent = MAj = minor = angle = Blue = Red = Green = amt1 = col1 = amt2 = col2 = amt3 = col3 = buds = None

#canna is full color
	ear_proof = canna.copy()
#lvs is binary

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#####################################   FULLNESS  ########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	

	
	
	
	lvs_fill = cnctfill(lvs)
	
	cnts = cv2.findContours(lvs_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		ar = cv2.contourArea(c)
		dim = int(math.sqrt(ar))
		ar = ar/(PixelsPerMetric*PixelsPerMetric)
		
	
	
	
	#square= np.zeros((dim,dim,3), np.uint8)
	#mask= np.zeros((dim,dim,1), np.uint8)
	#square[mask<255] = [255, 0, 255]
	#destin = '{}'.format(out) + '02_Proofs/' + filename + '_proof_mhm.jpeg'
	#cv2.imwrite(destin, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
	#destin = '{}'.format(out) + '02_Proofs/' + filename + '_proof_mhm_comp.jpeg'
	#cv2.imwrite(destin, canna, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
	
	
	tot_lf = cv2.countNonZero(lvs_fill)			#total leaf area
	tot_les = cv2.countNonZero(lvs)	#total lesion area
	Fill_rat = round(((tot_les/tot_lf)*100),4)
	tot_lf = tot_lf/(PixelsPerMetric*PixelsPerMetric)
	tot_les = tot_les/(PixelsPerMetric*PixelsPerMetric)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
####################### Area, Convexity, Solidity, fitEllipse ############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	cntss = cv2.findContours(lvs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cntss = cntss[0] if len(cntss) == 2 else cntss[1]
	cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
	for cs in cntss:	
		ear_proof =cv2.drawContours(ear_proof, cs, -1, (52, 118, 220), 3)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~moments + centroid~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#			
		moms =cv2.moments(cs)
		cXs= int(moms['m10'] / moms['m00']) #centroid x
		cYs= int(moms['m01'] / moms['m00']) #centroid y
		cv2.circle(ear_proof, (cXs, cYs), 25, (215, 247, 166), -1) #Centroid	



		if height_i > width_i:
			cv2.line(proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (255, 0, 255), 20) #width
		else:
			cv2.line(proof, (int(tltrXs), int(tltrYs)), (int(blbrXs), int(blbrYs)), (255, 0, 255), 20) #width

	
		cv2.putText(proof, '{:.1f} Pixels per Metric'.format(PixelsPerMetric),
						(int(180), int(180)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)

	
	#destin = './01_Proofs/' + filename + '2.jpeg'
	#cv2.imwrite(destin, proof, [int(cv2.IMWRITE_JPEG_QUALITY), 10])




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################### EAR BOX ############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		rects = cv2.minAreaRect(cs)
		boxs = cv2.boxPoints(rects)
		boxs = np.array(boxs, dtype='int')		
		boxs = order_points(boxs)
		ear_proof =cv2.drawContours(ear_proof, [boxs.astype(int)], -1, (200, 200, 200), 5)
	
# loop over the original points and draw them
# unpack the ordered bounding box, then compute the midpoint
		(tls, trs, brs, bls) = boxs
		(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
		(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
		(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
		(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
# compute the Euclidean distance between the midpoints
		Ear_Box_Width = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length
		Ear_Box_Length = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
		if Ear_Box_Width > Ear_Box_Length:
			Ear_Box_Width = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
			Ear_Box_Length = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length

#draw mid oints and lines on proof	
		cv2.line(ear_proof, (int(tltrXs), int(tltrYs)), (int(blbrXs), int(blbrYs)), (105, 105, 189), 7) #length
		cv2.circle(ear_proof, (int(tltrXs), int(tltrYs)), 15, (105, 105, 189), -1) #left midpoint
		cv2.circle(ear_proof, (int(blbrXs), int(blbrYs)), 15, (105, 105, 189), -1) #right midpoint
		cv2.line(ear_proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (255, 0, 255), 7) #width
		cv2.circle(ear_proof, (int(tlblXs), int(tlblYs)), 15, (255, 0, 255), -1) #up midpoint
		cv2.circle(ear_proof, (int(trbrXs), int(trbrYs)), 15, (255, 0, 255), -1) #down midpoint

		if PixelsPerMetric is not None:
			Ear_Box_Length = Ear_Box_Length/ (PixelsPerMetric)
			Ear_Box_Width = Ear_Box_Width/ (PixelsPerMetric)
			Ear_Box_Area = float(Ear_Box_Length*Ear_Box_Width)

#other measurements
		Ear_Area = cv2.contourArea(cs)
		perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
		Convexity = hullperimeters/perimeters
		Solidity_Hull = float(Ear_Area)/hull_areas
		Solidity_Box = float(Ear_Area)/Ear_Box_Area
		Ellipse = cv2.fitEllipse(cs)
		(ori,ent),(MAj,minor),angle = cv2.fitEllipse(cs)
		#print(Ellipse)
		
		if PixelsPerMetric is not None:
			Ear_Area = Ear_Area/(PixelsPerMetric*PixelsPerMetric) ####This is area so you have to square your reference!
			perimeters = perimeters/(PixelsPerMetric)
			Convexity = Convexity/(PixelsPerMetric)
			Solidity_Hull = Solidity_Hull/(PixelsPerMetric)
			Solidity_Box = Solidity_Box/(PixelsPerMetric)
			ori = ori/(PixelsPerMetric)
			ent = ent/(PixelsPerMetric)
			MAj = MAj/(PixelsPerMetric)
			minor = minor/(PixelsPerMetric)

	log.info('[FEAT]--{}--Morphometric Analysis...DONE'.format(filename))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### Color Analysis #############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	log.info('[COL]--{}--Extracting  main color...'.format(filename))

#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant Color
	pixels = np.float32(canna.reshape(-1, 3))
	#print(len(pixels))
	#print(type(pixels))

	pixels = [x for x in pixels if x[0] != 0 and x[1] != 0 and x[2] != 0]	
	pixels = np.asarray(pixels)
	
	n_colors = 2
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	dominant = palette[np.argmax(counts)]
	Blue = dominant[0]
	Red = dominant[1]
	Green = dominant[2]
	frame_fr = np.zeros_like(ears)
	frame_fr[lvs > 0] = [dominant[0], dominant[1], dominant[2]]

	log.info('[COL]--{}--Decomposing into three components...'.format(filename))
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cluster = KMeans(n_clusters=3).fit(pixels)
#Find and display most dominant colors
	labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
	(hist, _) = np.histogram(cluster.labels_, bins = labels)
	hist = hist.astype('float')
	hist /= hist.sum()
#Create frequency rect and iterate through each cluster's color and percentage
	rect = np.zeros((50, 300, 3), dtype=np.uint8)
	colors = sorted([(percent, color) for (percent, color) in zip(hist, cluster.cluster_centers_)])	
	
	amt1 = colors[0][0]
	col1 = colors[0][1]
	amt2 = colors[1][0]
	col2 = colors[1][1]
	amt3 = colors[2][0]
	col3 = colors[2][1]
	
	
	visualize = visualize_colors(cluster, cluster.cluster_centers_)
	resized = cv2.resize(visualize, (frame_fr.shape[1],frame_fr.shape[0]), interpolation = cv2.INTER_AREA)
	
	log.info('[COL]--{}--Color Analysis...DONE'.format(filename))
	
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### EXTREME POINTS ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#############       		?????????????????????????

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### POLY DP #############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#############       		?????????????????????????


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################################# FUN! #######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	hsv = cv2.cvtColor(canna, cv2.COLOR_BGR2HSV)
	hsv[canna == 0] = 0
	h,s,v = cv2.split(hsv)											#Split into it channel constituents
	lab = cv2.cvtColor(canna, cv2.COLOR_BGR2LAB)
	lab[canna == 0] = 0
	l,a,b_chnl = cv2.split(lab)										

	
	
	mskd,_ = cv2.threshold(h[h !=  0],1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	th3 = cv2.threshold(h,int(mskd*1.15),255,cv2.THRESH_BINARY)[1]
	th3[a == 0] = 0

	th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=1)

	sq_sq = np.zeros_like(th3) 										#Find the largest connected components to filter out
	nb_components, output, stats, \
	centroids = cv2.connectedComponentsWithStats(th3, connectivity=8)
	sizes = stats[1:, -1]; nb_components = nb_components - 1
	for i in range(0, nb_components):
		if sizes[i] >= 4000:										#This size threshold could change
			sq_sq[output == i + 1] = 255

	sq_sq2 = np.zeros_like(sq_sq)
	cnts = cv2.findContours(sq_sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		cv2.drawContours(sq_sq2, [c], 0, (255), -1)
		
	#sq_sq = cv2.bitwise_not(sq_sq2)
	cnts = cv2.findContours(sq_sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	buds = len(cnts)
	
	sq_sq2 = cv2.cvtColor(sq_sq2,cv2.COLOR_GRAY2RGB)






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### SAVE #############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proofs
	
	h = cv2.cvtColor(h,cv2.COLOR_GRAY2RGB)
	lvs = cv2.cvtColor(lvs,cv2.COLOR_GRAY2RGB)
	ears = canna.copy()
	b,g,r = cv2.split(ears)
	hsv = cv2.cvtColor(ears, cv2.COLOR_BGR2HSV)
	hsv[canna == 0] = 0
	#h,s,v = cv2.split(hsv)											#Split into it channel constituents
	lab = cv2.cvtColor(ears, cv2.COLOR_BGR2LAB)
	lab[canna == 0] = 0
	#l,a,b_chnl = cv2.split(lab)										#Split into it channel constituents





	im_a = cv2.hconcat([img, proof])
	im_b = cv2.hconcat([ear_proof, lvs])
	im_c = cv2.hconcat([frame_fr, resized])
	im_d = cv2.hconcat([hsv, lab])
	im_e = cv2.hconcat([canna, sq_sq2])
	
	im_f = cv2.vconcat([im_a, im_b, im_c, im_d, im_e])
	
	destin = '{}'.format(out) + '02_Proofs/' + filename + '_proof.jpeg'
	log.info('[PPM]--{}--Proof saved to: {}'.format(filename, destin))
	cv2.imwrite(destin, im_f, [int(cv2.IMWRITE_JPEG_QUALITY), 10])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proof and pixels per metric csv			
	csvname = out + 'features' +'.csv'
	file_exists = os.path.isfile(csvname)
	with open (csvname, 'a') as csvfile:
		headers = ['Filename', 'PPM', 'Ar', 'All_Area', 'tot_les', 'Fill_rat', 'Ear_Box_Width', 'Ear_Box_Length', 'Ear_Box_Area', 'perimeters',  'Convexity', 'Solidity_Hull', 'Solidity_Box', 'ori', 'ent', 'MAj', 'minor', 'angle', 'Blue_Main', 'Red_Main', 'Green_Main', '1_amt', 'Col1', '2_amt', 'Col2', '3_amt', 'Col3', 'Buds']  
		writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
		if not file_exists:
			writer.writeheader()  # file doesn't exist yet, write a header	
		
		writer.writerow({'Filename': filename, 'PPM': PixelsPerMetric, 'Ar': ar, 'All_Area': tot_lf, 'tot_les': tot_les, 'Fill_rat': Fill_rat, 'Ear_Box_Width': Ear_Box_Width, 'Ear_Box_Length': Ear_Box_Length, 'Ear_Box_Area': Ear_Box_Area, 'perimeters': perimeters,  'Convexity': Convexity, 'Solidity_Hull': Solidity_Hull, 'Solidity_Box': Solidity_Box, 'ori': ori , 'ent': ent, 'MAj': MAj, 'minor': minor, 'angle': angle, 'Blue_Main': Blue, 'Red_Main': Red, 'Green_Main': Green, '1_amt': amt1, 'Col1': col1, '2_amt': amt2, 'Col2': col2, '3_amt': amt3, 'Col3': col3, 'Buds': buds})
	


main()
