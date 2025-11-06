import cv2 as cv
import numpy as np
from glob import glob
import tensorflow	as tf
from tensorflow.keras import datasets, layers, models


# the directory of the image database
database_dir = "image.orig"

# Compute pixel-by-pixel difference and return the sum
def compareImgs(img1, img2):
    # resize img2 to img1
	img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
	diff = cv.absdiff(img1, img2)
	return diff.sum()

def compareImgs_color(img1, img2):
	img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
	diff = np.abs(img1 - img2)
	return np.mean(diff)    

def compareImgs_hist(img1, img2):
	width, height = img1.shape[1], img1.shape[0]
	img2 = cv.resize(img2, (width, height))
	num_bins = 10
	hist1 = [0] * num_bins
	hist2 = [0] * num_bins
	bin_width = 255.0 / num_bins + 1e-4
	# compute histogram from scratch

	# for w in range(width):
	# 	for h in range(height):
	# 		hist1[int(img1[h, w] / bin_width)] += 1
	# 		hist2[int(img2[h, w] / bin_width)] += 1

	# compute histogram by using opencv function
	# https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d

	hist1 = cv.calcHist([img1], [0], None, [num_bins], [0, 255])
	hist2 = cv.calcHist([img2]	, [0], None, [num_bins], [0, 255])
	sum = 0
	for i in range(num_bins):
		sum += abs(hist1[i] - hist2[i])
	return sum / float(width * height)

def compare_hist_rgb(img1, img2, bins=16, method='chisqr'):
    """
    Compare two color images using RGB histograms.
    
    Args:
        img1, img2: BGR images (numpy arrays)
        bins: number of bins per channel
        method: comparison method ('correlation', 'chisqr', 'intersect', 'bhattacharyya')
    
    Returns:
        similarity: float (higher = more similar for correlation/intersection)
    """
    # Split channels
    channels = ['b', 'g', 'r']
    hist1 = []
    hist2 = []
    
    for i in range(3):
        h1 = cv.calcHist([img1], [i], None, [bins], [0,256])
        h2 = cv.calcHist([img2], [i], None, [bins], [0,256])
        # Normalize
        h1 = cv.normalize(h1, h1).flatten()
        h2 = cv.normalize(h2, h2).flatten()
        hist1.append(h1)
        hist2.append(h2)
    
    # Combine per-channel similarity
    sim = 0
    for h1, h2 in zip(hist1, hist2):
        if method == 'correlation':
            sim += cv.compareHist(h1, h2, cv.HISTCMP_CORREL)
        elif method == 'chisqr':
            sim += cv.compareHist(h1, h2, cv.HISTCMP_CHISQR)
        elif method == 'intersect':
            sim += cv.compareHist(h1, h2, cv.HISTCMP_INTERSECT)
        elif method == 'bhattacharyya':
            sim += cv.compareHist(h1, h2, cv.HISTCMP_BHATTACHARYYA)
        else:
            raise ValueError("Invalid method")
    
    return sim / 3  # average over channels

def compareImgs_DCT(img1, img2):
	img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

	img1_dct = cv.dct(np.float32(img1))
	img2_dct = cv.dct(np.float32(img2))

	dct_size = 16
	# img1_low = img1_dct[:dct_size, :dct_size]
	# img2_low = img2_dct[:dct_size, :dct_size]

	diff = cv.absdiff(img1, img2)
	return np.mean(diff)

def compareImgsDCT_color(img1, img2, size=16):
    # Resize second image
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

    diff_total = 0

    # Compute per-channel DCT difference
    for c in range(3):  # B, G, R channels
        ch1 = np.float32(img1[:, :, c])
        ch2 = np.float32(img2[:, :, c])

        dct1 = cv.dct(ch1)
        dct2 = cv.dct(ch2)

        # Focus on top-left block = low frequencies (overall structure)
        dct1 = dct1[:size, :size]
        dct2 = dct2[:size, :size]

        # Mean absolute difference per channel
        diff = np.mean(np.abs(dct1 - dct2))
        diff_total += diff

    # Average across all color channels
    return diff_total / 3.0

def retrieval():
	print("1: beach")
	print("2: mountain")
	print("3: food")
	print("4: dinosaur")
	print("5: flower")
	print("6: horse")
	print("7: elephant")
	choice = input("Type in the number to choose a category and type enter to confirm\n")
	if choice == '1':
		src_input = cv.imread("image.query/beach.jpg")
		print("You choose: %s - beach\n" % choice)
	if choice == '2':
		src_input = cv.imread("image.query/mountain.jpg")
		print("You choose: %s - mountain\n" % choice)
	if choice == '3':
		src_input = cv.imread("image.query/food.jpg")
		print("You choose: %s - food\n" % choice)
	if choice == '4':
		src_input = cv.imread("image.query/dinosaur.jpg")
		print("You choose: %s - dinosaur\n" % choice)
	if choice == '5':
		src_input = cv.imread("image.query/flower.jpg")
		print("You choose: %s - flower\n" % choice)
	if choice == '6':
		src_input = cv.imread("image.query/horse.jpg")
		print("You choose: %s - horse\n" % choice)
	if choice == '7':
		src_input = cv.imread("image.query/elephant.jpg")
		print("You choose: %s - elephant\n" % choice)	

	min_diff = 1e50

	# src_input = cv.imread("man.jpg")

	cv.imshow("Input", src_input)

	# change the image to gray scale
	#src_gray = cv.cvtColor(src_input, cv.COLOR_BGR2GRAY)

	# read image database
	database = sorted(glob(database_dir + "/*.jpg"))


	for img in database:
		# read image
		img_rgb = cv.imread(img)
		# convert to gray scale
		img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
		# compare the two images
		diff = compare_hist_rgb(src_input, img_rgb)
		# compare the two images by histogram, uncomment the following line to use histogram
		# diff = compareImgs_hist(src_gray, img_gray)
		print(img, diff)
		# find the minimum difference
		if diff <= min_diff:
			# update the minimum difference
			min_diff = diff
			# update the most similar image
			closest_img = img_rgb
			result = img
	
	print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
	print("\n")

	cv.imshow("Result", closest_img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def SIFT():
	img1 = cv.imread("image.query/flower.jpg")
	img2 = cv.imread("image.orig/685.jpg")
	if img1 is None or img2 is None:
		print('Error loading images!')
		exit(0)
	#-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
	minHessian = 400
	detector = cv.SIFT_create()
	keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
	#-- Step 2: Matching descriptor vectors with a brute force matcher
	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
	matches = matcher.match(descriptors1, descriptors2)
	#-- Draw matches
	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
	#-- Show detected matches
	cv.imshow('Matches: SIFT (Python)', img_matches)
	cv.waitKey()

	# draw good matches
	matches = sorted(matches, key = lambda x:x.distance)
	min_dist = matches[0].distance
	good_matches = tuple(filter(lambda x:x.distance <= 2 * min_dist, matches))

	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	
	#-- Show detected matches
	cv.imshow('Good Matches: SIFT (Python)', img_matches)
	cv.waitKey()


def main():
	# img = cv.imread("beach.jpg")
	# cv.imshow("Image", img)
	# from matplotlib import pyplot as plt
	# plt.hist(img.ravel(),10,[0,256]); plt.show()
	# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# cv.imshow("Gray Image", gray_img)
	# cv.waitKey()


	print("1: Image retrieval demo")
	print("2: SIFT demo")
	number = int(input("Type in the number to choose a demo and type enter to confirm\n"))
	if number == 1:
		retrieval()
	elif number == 2:
		SIFT()
		# pass
	else:
		print("Invalid input")
		exit()

main()