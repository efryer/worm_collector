#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import skimage
from skimage import io
from skimage import feature
from skimage import morphology
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import ndimage as ndi
import numpy as np
import cv2
import pathlib
import csv
import seaborn as sns


# In[2]:

### Change this line to match your file path
base_folder = pathlib.Path('/Users/emilyfryer/Desktop/worm_collector')


image_location = base_folder.joinpath('Images')
results_location = base_folder.joinpath('automated_analysis_results')


# Load images


def load_image_data(data_location, plate_id):

    pre_file = data_location.joinpath(plate_id + '_Pre.tif')
    pre_image = skimage.io.imread(pre_file)
    
    fin_file = data_location.joinpath(plate_id + '_Fin.tif')
    fin_image = skimage.io.imread(fin_file)

    return pre_image, fin_image

# Crop to a single well

def crop_to_one_well(pre_image, fin_image, well_id):
    # select bounds of the lane we're working on now
    if well_id == 'S':
        left_boundary = 170
        right_boundary = 1310
    elif well_id == 'R':
        left_boundary = 1600
        right_boundary = 2750
    elif well_id == 'Q':
        left_boundary = 2950
        right_boundary = 4100
    elif well_id == 'P':
        left_boundary = 4350
        right_boundary = 5500  #pre_image.shape[1]
    else:
        raise ValueError('Lane label not recognized.')
    
    
    pre_image = pre_image[ : ,left_boundary:right_boundary]
    fin_image = fin_image[ : ,left_boundary:right_boundary ]
 
    # Crop larger image to make before and after images the same size
    x_min = 0
    x_max = min(pre_image.shape[1], fin_image.shape[1])
    
    y_min = 100
    y_max = 3540 

    pre_image = pre_image[y_min:y_max, x_min:x_max]
    fin_image = fin_image[y_min:y_max, x_min:x_max]
    
    return pre_image, fin_image




# Align pre and post images to allow for better background subtraction

def alignImages(im1, im2):

    MAX_FEATURES = 400
    GOOD_MATCH_PERCENT = 0.05
   
  # Detect ORB features and compute descriptors.

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
   
  # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
 
  # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
   
  # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
    height, width = im1.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    
    return im1Reg, h


# Automatically locate the worms

def find_worms(pre_image, fin_image):
    feature_find_start = time.time()
    
    ## Subtract background
    bkg_subtract = np.subtract(pre_image.astype('int16'),
                               fin_image.astype('int16'))
    bkg_subtract[bkg_subtract > 0] = 0
 
    
    ## Threshold to get binary image
    thresh = skimage.filters.threshold_otsu(bkg_subtract)
    binarized = bkg_subtract < thresh
    #print('Binarization threshold is', str(thresh))

    ## Find features in binary image
    labeled_array, num_features = ndi.label(binarized)
    all_regions = measure.regionprops(label_image=labeled_array, intensity_image=fin_image,
                                      coordinates='rc')
    
    filtered_regions = []
    for region in all_regions:
        area = region.area
        if area >= 100 and area <=2000 and region.major_axis_length < 200:
            filtered_regions.append(region)
        
#   print('Worm finding and filtering took', str(int(time.time() - feature_find_start)), 'seconds.')
    #print('Filtered Regions: ' + str(len(filtered_regions)))
    return filtered_regions

# Load in human annotated data

def load_manual_results(plate_id, well_id):
    results_files = base_folder.joinpath('Manual_analysis_results').glob(plate_id + '*' + well_id + '.csv')

    manual_analysis_results = []
    for file in results_files:
        this_experiment_df = pd.read_csv(file, index_col=' ')
        plate_id = file.stem[0:6]
        this_experiment_df['plate_id'] = plate_id
        well_id = file.stem[-1].upper()
        this_experiment_df['well_id'] = well_id
        manual_analysis_results.append(this_experiment_df)
    manual_analysis_results = pd.concat(manual_analysis_results)

    ## Correct for well location in manual analysis results
    if well_id == 'S':
        left_boundary = 170
        right_boundary = 1310
    elif well_id == 'R':
        left_boundary = 1600
        right_boundary = 2750
    elif well_id == 'Q':
        left_boundary = 2950
        right_boundary = 4100
    elif well_id == 'P':
        left_boundary = 4350
        right_boundary = 5500 

    # Correct for comparison against cropped wells
    manual_analysis_results['X'] = manual_analysis_results['X'] - left_boundary
    manual_analysis_results['Y'] = manual_analysis_results['Y'] - 100
    manual_analysis_results['Y_X'] = manual_analysis_results[['Y', 'X']].apply(tuple, axis=1)
    manual_tuple = manual_analysis_results['Y_X'].tolist()
    #print('Manual count: ' + str(len(manual_tuple)))
    return manual_tuple, manual_analysis_results


# Compare the manual against the automated results


def compare_manVauto(auto_ID_worms, manual_ID_worms):
    compared = []
    buffer = 30

    for region in auto_ID_worms:
        for i in manual_ID_worms:
            difference = abs(np.subtract(i, region.centroid))
            if difference[0] < buffer and difference[1] < buffer:
                #print(difference)
                compared.append(region)
                manual_ID_worms.remove(i)
    #print('Compared: ' + str(len(compared)))
    return compared



# Loop through all of the images

results = []
start_time = time.time()
for image in image_location.glob('DEC*Fin.tif'): #image_location.glob('scan*Fin.tif'): #
    plate_start = time.time()
    plate_id = image.stem[0:6]
    if image.parent.joinpath(plate_id + '_Pre.tif').exists is False:
        raise NameError('No matching pre-image found.')
    
    # Load the entire image
    pre_image, fin_image = load_image_data(image_location, plate_id)
    #print('Image load took', str(int(time.time()-plate_start)), 'seconds.')
    

    for well_id in ['P', 'Q', 'R', 'S']:
        
        ## Choose well to look at
        cropped_pre_image, cropped_fin_image = crop_to_one_well(pre_image, fin_image, well_id)

        ## Make sure that the pre and fin images are properly aligned
        imReg, h = alignImages(cropped_fin_image, cropped_pre_image)
        
        ## Locate worms using automation
        worms = find_worms(cropped_pre_image, imReg)
        
        ## Load in manually anotated worms
        manual_results, manual_df = load_manual_results(plate_id, well_id)

        ## Compare manually annotated worms with automatically located worms and filter out any discrepencies
        definitely_worms = compare_manVauto(worms, manual_results)
        
        for worm in definitely_worms:
            results_dict =  {'Plate_id': plate_id,
                            'Well_id': well_id,
                            'Area': worm.area,
                            'Convex Area': worm.convex_area,
                            'Major_axis_length': worm.major_axis_length, 
                            'bbox': worm.bbox,
                            'centroid': worm.centroid}
            results.append(results_dict)

#Save results to a csv
results_df = pd.DataFrame(results)
results_df.to_csv(path_or_buf=base_folder.joinpath('confirmed_worms.csv'))
        


# In[31]:

print('Total Time', str(int(time.time()-start_time)), 'seconds.')

