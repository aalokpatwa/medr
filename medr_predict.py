#!/usr/bin/env python
# coding: utf-8
'''
MEDICAL SURVEYS RESPONSE PREDICTOR
---
Goal: Given a directory of TIFF Scans of Survey Responses, prepare a CSV file with predicted responses.
The predictor is trained for two surveys: UROLOGY ONCOLOGY BENIGN (UOB) AND UROLOGY PROSTATE SYMPTOM (UPS) 
---
Format of CSV File
Columns: id,survey, A1,A2,A3,...A13 
Rows: One row for each survey in the format below
id is derived from the filename of the TIFF
survey is either 'uob' or 'ups'
An is the answer to question n, in numerical form - ranging from 0 to 6 - the response circled by the respondent. 

----
Process:
For each TIFF Scan:
Step 1: Read the scan and use Tesseract to identify the survey: UOB or UPS
Step 2: From the the geometry parameters  of the survey: 
        Identify the tables on the page and the question numbers
        For each question
             Identify the answerbox locations
             Crop each answerbox
             Use trained CNN to predict whether the answerbox was marked or not
             Formulate the numerical answer after examining all answerboxes. 
             Answer is 'NA' if more than two answerboxes are responded, or none is responded. 
Step 3: Append id, survey, A1, A2, ... A13 

'''

# All imports
from PIL import ImageOps
from PIL import Image
from skimage import io
import PIL
import cv2
import numpy as np
import pandas as pd
import csv
import os
import shutil  
import random, glob
import pytesseract
import argparse
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

# Other
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.models import load_model

# Utils
import matplotlib.pyplot as plt
import sys
import time, datetime


# Files
import utils

# Global constants
TOPLEFT = 0
BOTTOMRIGHT = 1
TOPRIGHT = 2
BOTTOMLEFT = 3


# GEOMETRY DEFINITIONS
# corner: (rowbegin, rowend, columnbegin, columnend, checkrange)
# tables: tables is a list of tables. For each table, the tuple defines the following
# ((Center of q1a1), (dist between q,a), (#q, #a), (boxsize))
# tables: [table1, table2, table3]



#UPS Survey Page 1
upsp1_corner1=(450,650,100,300,75) 
#Given the first corner, find the second corner at bottom right so that proper scaling of the table can be done
upsp1_corner2=(2800,3000,2375,2525,75) 
upsp1_hw=(2390, 2264)
upsp1_shape=(3250, 2500)
upsp1_qnos= ['1', '2', '3', '4', '5', '6', '7'] # question numbers on this survey page
#For each table, center of q1a1, relative column of each answer, relative row of each question, boxsize
upsp1_tables=[((488,1517), [0,108,235,359,489,599], [0,220,438,657,875,1093], (200,120)), 
            ((2272,1524), [0,109,224,329,443,555], [0], (200,120)) ]

upsp2_corner1 = (400,550,100,300,75)
upsp2_corner2 = (1230,1430,2350,2550,75)
upsp2_hw = (817,2244)
upsp2_shape=(3250, 2500)
upsp2_qnos = ['8']
upsp2_tables = [((747,1531), [0,107,219,360,480,572,664], [0], (180,120))]


uobp1_corner1 = (500,770,75,350,75)
uobp1_corner2 = (2850,3110,2380,2500,75)
uobp1_hw = (2396,2268)
uobp1_shape=(3250, 2500)
uobp1_qnos = ['1','2','3','4','5','6','7','8']
uobp1_tables = [((223,998), [0,228,482,719,958,1160], [0,255,507,761,953,1151,1336], (227,271)),
                ((2252,1316), [108,230,435,655,789,907], [0], (200,166))]

uobp2_corner1 = (500,775,0,250,75)
uobp2_corner2 = (2943,3205,2200,2463,75)
uobp2_hw = (2478,2255)
uobp2_shape=(3250, 2500)
uobp2_qnos = ['9', '10', '11', '12', '13']
uobp2_tables = [((256,642), [0,329,599,888,1193,1478], [0,486,966,1417,1872], (307,351))]

# create a list for all pages of the survey
ups_corner1 = [upsp1_corner1, upsp2_corner1]
ups_corner2 = [upsp1_corner2, upsp2_corner2]
ups_hw = [upsp1_hw, upsp2_hw]
ups_shape = [upsp1_shape, upsp2_shape]
ups_tables = [upsp1_tables, upsp2_tables]

uob_corner1 = [uobp1_corner1, uobp2_corner1]
uob_corner2 = [uobp1_corner2, uobp2_corner2]
uob_hw = [uobp1_hw, uobp2_hw]
uob_shape = [uobp1_shape, uobp2_shape]
uob_tables = [uobp1_tables, uobp2_tables]

# Model setup
CONFIDENCE_THRESHOLD = 0.55 

DROPOUT = 1e-3
FC_LAYERS = [1024,1024]
model = "ResNet50"
if model == "MobileNet":
    HEIGHT = 224
    WIDTH = 224
    from keras.applications.mobilenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif model == "ResNet50":
    HEIGHT = 224
    WIDTH = 224
    from keras.applications.resnet50 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT,WIDTH,3))

class_list_file = "./class_list.txt"
class_list = utils.load_class_list(class_list_file)
finetune_model = utils.build_finetune_model(base_model,dropout=DROPOUT, fc_layers=FC_LAYERS, num_classes=len(class_list))
finetune_model.load_weights("./"+model+"_model_weights.h5")



def classify(image):
    global finetune_model

    try:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    except:
        print("ERROR classify: could not convert image into color", image.shape)
        return 0
    try:
        image = np.float32(cv2.resize(image, (HEIGHT, WIDTH))) #Keras applications need floating point datatype 
    except:
        print ("ERROR classify:Could not resize image and convert to float")
        return 0
    image = preprocessing_function(np.reshape(image, (1,HEIGHT,WIDTH,3)))
    # Run the classifier and print results

    out = finetune_model.predict(image) # out is a list, where each item on the list has a list of class probabilities of that list
    class_probabilities = out[0]
    #class_prediction = list(class_probabilities).index(max(class_probabilities))
    pos_confidence = class_probabilities[1]
    return pos_confidence


# identify uses Tesseract to detect the identifying text in upper quarter of the page and return
# the survey name as 'uob' or 'ups'
# identifying text programmed are
# UROLOGY ONCOLOGY BENIGN, UROLOGY PROSTATE SYMPTOM
# tiff is a complete pathname
def identify(tiff):
    # Open image path as a PIL Image object. This will later be used to iterate throughout different pages
    # PIL is used because it is the only library with the ability to view multi-page TIFs
    with Image.open(tiff) as pil_img:
        # Find the number of frames (pages) in the TIF
        no_frames = pil_img.n_frames
        # Iterate through frames of the image 
        for i in range(1): #ERROR TO FIX: only frame 0 is being recognized!
            #pil_img.seek(i)        
            # Cast the current PIL frame to a numpy array of type uint8 (important)
            np_img = np.array(pil_img, dtype='uint8')*255
            title = np_img[250:750, 1000:]
            textstring = pytesseract.image_to_string(title)
            if ("UROLOGY PROSTATE SYMPTOM" in textstring):
                return('ups')
            elif ("IPSS" in textstring):
                return('ups')
            elif ("UROLOGY ONCOLOGY BENIGN" in textstring):
                return('uob')
            elif ("HYPERTROPHY" in textstring):
                return('uob')
            elif ("ONC UROLOGY MALE" in textstring):
                return('ums')
                print ("ums survey", tiff, " skipped")
            elif ("SHIM" in textstring):
                return('ums')
            else:
                return('unknown')



def score(row, column, checkrange, image, flag):
    '''
    Given a starting row and column and a certain checkrange, slices a portion of an image and sums pixel values 
    in four directions. 
    Maximum score is achieved when there is a line to the bottom and to the right, ie when the pixel represents a 
    top left corner. Purpose is to locate an anchor point for cropping.
    
    Inputs:
    - row: integer of pixel's row
    - column: integer of pixel's row
    - checkrange: integer of how long to slice in either direction
    - image: grayscale, inverted numpy array 
    
    Output:
    - integer score, representing how likely it is that the current pixel is a top left corner
    '''
    alignmargin = 2
    rows, columns = image.shape
    if ((row+checkrange) < rows):
        down = image[row:row+checkrange, column].sum() 
    else:
        down = image[row:rows-1, column].sum()
    if (row-checkrange >=0):
        up = image[row-checkrange:row, column].sum() 
    else:
        up = image[0:row, column].sum()
    if ((column+checkrange) < columns):
        right = image[row, column:column+checkrange].sum() 
    else:
        right = image[row, column:columns-1].sum()
    if ((columns-checkrange) >=0):
        left = image[row, column-checkrange:column].sum() 
    else:
        left = image[row, 0:column].sum()
    
    if (flag == TOPLEFT):
        score = (down-up+right-left)
    elif (flag == BOTTOMRIGHT):
        score = (up-down+left-right)
    elif (flag == TOPRIGHT):
        score = (down-up+left-right)
    elif (flag == BOTTOMLEFT):
        score = (up-down+right-left)
    else:
        print("ERROR score has unrecognized flag:", flag)
        score = 0
    return score

def corner_det(img, cornerparams, flag):
    '''
    Given a search range of rows and columns, a range to check in either direction, and an image, detects the pixel
    that is most likely to represent a top left corner.
    
    Input: 
    - rowbegin, rowend: integers specifying the row ranges to search in
    - columnbegin, columnend: integers specifying the column ranges to search in
    - checkrange: how far to sum in either direction
    - img: grayscale, inverted numpy array 
    
    Output:
    - corner_row, corner_column: integers representing coordinates of the corner
    - score_max: integer representing the score achieved by those coordinates. Max is 38250
    '''
    rowbegin, rowend, columnbegin, columnend, checkrange = cornerparams
    maxrow, maxcolumn = img.shape
    rowbegin = np.minimum(rowbegin, maxrow-1)
    rowend = np.minimum(rowend, maxrow-1)
    columnbegin=np.minimum(columnbegin, maxcolumn-1)
    columnend = np.minimum(columnend, maxcolumn-1)
    score_max = 0
    score_ok = 255*checkrange*2
    corner_row = 0
    corner_column = 0
    img = img.astype('int32')
    
    if (img.ndim == 3):
        img = img[:,:,0]
    #Nested for loops iterate throughout the search range, checking every pixel
    for row in range(rowbegin,rowend):
        for column in range(columnbegin, columnend):
            # Find score of current pixel
            new_score = score(row, column, checkrange, img, flag)
            # check whether score of current pixel is the largest one found up to this point 
            if new_score > score_max:
                score_max = new_score
                corner_row = row
                corner_column = column
            #If the score reaches the maximum value, we know that it is the corner so we need not loop anymore
            if (score_max >= score_ok):
                return corner_row, corner_column, score_max
    return corner_row, corner_column, score_max


###ERROR: IF CORNER LINE IS NOT ALIGNED, CORNER DETECTION IS OFF




def crop_n_predict(tifffile, tiff, cornerparams_list, cornerparams2_list, hw_list, shape_list, tables_list):

    # Open image path as a PIL Image object. This will later be used to iterate throughout different pages
    # PIL is used because it is the only library with the ability to view multi-page TIFs
    # Find the number of frames (pages) in the TIF
    # Iterate through frames of the image
    qno = 0
    questions = []
    #print ("Now on question: " + str(qno))
    for i in range(2):
        with Image.open(tiff) as pil_img:
            if i == 1:
                try: 
                    pil_img.seek(1)
                except:
                    print("ERROR ", tifffile, " could not open page 2")
                    break
            # Cast the current PIL frame to a numpy array of type uint8 (important)
            np_img = np.array(pil_img, dtype="uint8") * 255
            if len(np_img.shape) !=2 :
                np_img = np_img[:,:,2]
            np_img = 255 - np_img # invert pixels
            page_r, page_c = np_img.shape
            cornerparams = cornerparams_list[i]
            cornerparams2 = cornerparams2_list[i]
            hw = hw_list[i]
            shape_r, shape_c = shape_list[i]
            tables = tables_list[i]
            # adjust corner parameters based on shape
            scale_r = round(1.0*page_r/shape_r, 3)
            scale_c = round(1.0*page_c/shape_c, 3)
            cornerparams_scaled = (int(scale_r*cornerparams[0]), int(scale_r*cornerparams[1]), int(scale_c*cornerparams[2]), int(scale_c*cornerparams[3]), cornerparams[4])
            cornerparams2_scaled = (int(scale_r*cornerparams2[0]), int(scale_r*cornerparams2[1]), int(scale_c*cornerparams2[2]), int(scale_c*cornerparams2[3]), cornerparams2[4])
            corner_row, corner_column, score_max = corner_det(np_img, cornerparams_scaled, TOPLEFT)
            corner2_row, corner2_column, score2_max = corner_det(np_img, cornerparams2_scaled, BOTTOMRIGHT)
            #print(tifffile, "page ", i, " corners 1 and 2:", corner_row, corner_column, corner2_row, corner2_column)
            actual_height = corner2_row - corner_row
            actual_width = corner2_column - corner_column
            height, width = hw
            scale_height = round((1.0*actual_height)/height, 3)
            scale_width = round((1.0*actual_width)/width, 3)
            #print (imgid, np_img.shape, "1st corner:", corner_row, corner_column, "2nd corner:", corner2_row, corner2_column, "scale", scale_height, scale_width)
            for table in tables:
                (q1a1_row, q1a1_column), a_dist_list, q_dist_list, (boxrows, boxcolumns) = table 
                #scale all table values as this survey scan has its own scale
                q1a1_row = int(q1a1_row*scale_height)
                q1a1_column = int(q1a1_column*scale_width)
                a_dist_list = [int(a*scale_width) for a in a_dist_list]
                q_dist_list = [int(q*scale_height) for q in q_dist_list]
                num_q = len(q_dist_list)
                num_a = len(a_dist_list)
                for q in range(num_q):
                    qno += 1
                    answerfound = False
                    answerprob = 0
                    for ano in range(num_a):
                        crop_begin_row = corner_row + q1a1_row + q_dist_list[q] - boxrows//2
                        crop_end_row = corner_row + q1a1_row + q_dist_list[q] + boxrows//2
                        crop_begin_col = corner_column + q1a1_column + a_dist_list[ano] - boxcolumns//2
                        crop_end_col = corner_column + q1a1_column + a_dist_list[ano] + boxcolumns//2
                        cropped = np_img[crop_begin_row:crop_end_row, crop_begin_col:crop_end_col]
                        # predict on cropped image
                        pos_prob = classify(cropped) 
                        if pos_prob > answerprob:
                            answerprob = pos_prob
                            answer = ano
                            answerimg = cropped
                    if answerprob > CONFIDENCE_THRESHOLD:
                        questions.append(str(answer))
                    else:
                        questions.append("NA")
                        # print("qno ", qno, " max prob", answerprob, "ano ", answer)
                        if (answerprob != 0):
                            filename = './'+tifffile.split('.')[0]+'_'+str(qno)+'_'+str(answer) + '.png'
                            #cv2.imwrite(filename, answerimg)
    return questions

def process(tifffile, tiff):
    skip = False
    surveyname = identify(tiff) # Tesseact 
    if (surveyname == 'uob'):
        skip = False
        corner1 = uob_corner1
        corner2 = uob_corner2
        hw = uob_hw
        shape = uob_shape
        tables = uob_tables
    elif (surveyname == 'ups'):
        skip = False
        corner1 = ups_corner1
        corner2 = ups_corner2
        hw = ups_hw
        shape = ups_shape
        tables = ups_tables
    else:
        print(tifffile, surveyname, " skipped")
        skip = True
    if not skip:
        predicted_row = crop_n_predict(tifffile, tiff, corner1, corner2, hw, shape, tables)
        prediction = [tifffile.split('.')[0], surveyname, predicted_row]
        print(tifffile, surveyname, predicted_row)
    else:
        prediction = None
    return(prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiffdir', type=str, default='./tiffdir/', help='Full path name of directory where TIFF Scans of survey responses reside')
    parser.add_argument('--outfile', type=str, default='./medr_predictions.csv', help='Full path name of output file')
    args = parser.parse_args()
    
    with open(args.outfile, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(["id", "survey", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13"])
        for tifffile in os.listdir(args.tiffdir):
            ext = tifffile.split(".")[1].lower()
            if ("._" not in tifffile) and (ext in "tiff"):
                tiff = os.path.join(args.tiffdir, tifffile) #full path
                prediction = process(tifffile, tiff)
            else:
                prediction = None
            if prediction != None :
                tiffid = prediction[0]
                surveyname = prediction[1]
                questions = prediction[2]
                row = [tiffid,surveyname]
                for question in questions:
                    row.append(question)
                csv_writer.writerow(row)     # write code to save pred_list in csv format in the outfile


