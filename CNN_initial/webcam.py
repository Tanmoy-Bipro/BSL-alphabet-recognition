import cv2
import os
import glob
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import imutils
from PIL import ImageFont, ImageDraw, Image
import copy
bg = None
count = 0
dir_path = os.path.dirname(os.path.realpath(__file__))
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


#image_path = sys.argv[1]
#filename = dir_path + '/' + image_path + '/'
classes = ['A','B','C','D','E']
num_classes = len(classes)
image_size = 224
num_channels = 3

totalExperimented =0;
truePositive=0;aWeight = 0.5;c = 0;i = 4;j = 0;consecutive = 0;


cap = cv2.VideoCapture(0)

#res, score = '', 0.0


mem = '';

sequence = '';
num_frames = 0;
b, g, r, a = 0, 255, 0, 0;
fontpath = "kalpurush.ttf";
#font = ImageFont.truetype(fontpath, 100);
kk=0
with tf.Session() as sess:
     ## Let us restore the saved model
    #sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('AI-signature-verification.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()    
    while True:
        ret, img = cap.read()
        img = imutils.resize(img, width=700, height=600)
        img = cv2.flip(img, 1)
        clone = img.copy()
        
        if ret:
            x1, y1, x2, y2 = 50, 300, 340, 590
            img_cropped = img[x1:x2, y1:y2]
            rgb = img_cropped
            gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            # cv2.imshow('image', img_cropped)
            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
                # segment the hand region
                hand = segment(gray)
                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand
                    # mask = np.zeros((290, 290), np.uint8)
                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (590, 50)], -1, (0, 0, 255))
                    # cv2.drawContours(mask, [segmented + (right, top)], 0, 255, -2)
                    # mean = cv2.mean(clone, mask=mask)
                    # cv2.imshow("Masked", mask)
                    # cv2.imshow("Original", rgb)
                    skin = cv2.bitwise_and(rgb, rgb, mask=thresholded)
                    cv2.imshow("Thesholded", thresholded)
                    cv2.imshow("skin", skin)
                    skinflip = cv2.flip(skin, 1)
                    cv2.imshow("Cropped/flipped", skinflip)
                    img_cropped = skinflip
            num_frames += 1
            c += 1
		
            cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)
            cv2.imshow("img", img)
            filename = img_cropped
            images = []
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
    # Reading the image using OpenCV
            #image = filename
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
            image = cv2.resize(filename, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            images.append(image)
            images = np.array(images, dtype=np.uint8)
            images = image.astype('float32')
            images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
            x_batch = images.reshape(1, image_size, image_size, num_channels)

   

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, num_classes))

    ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
    #print('Result: ')
    #print(result)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]

        j = 1; val =0;probableClass = 0;
    
    
        for ii in result:
            chk=0
            for jj in ii:
                chk=chk+1
                if jj>val:
                   val =jj
                   probableClass = chk
            totalExperimented = totalExperimented +1
            if val >= 0.55:

        #print('Photo Name       : ', className)
                print('predicted person : ', classes[probableClass -1])
                print('Probability value: ', val)
        #print('Match with       : ', probableClass)
                truePositive = truePositive + 1
            else:
        #print('Photo Name       : ', className)
                print('predicted person : Unknown')
                print('Probability value: ', val)
        #print('Match with       : ', probableClass)

        #print()
        #print()

# print(decisionMartrix)
# predans = 0
# totaltest = 0
# row = 0
# ii=0
# jj=0
# for i in decisionMartrix:
#     jj=0
#     for j in i:
#         if jj==ii:
#             predans = predans + decisionMartrix[ii][jj]
#             totaltest = totaltest + decisionMartrix[ii][jj]
#         else:
#             totaltest = totaltest + decisionMartrix[ii][jj]
#         jj = jj+1
#     ii= ii+1

# ans = float(truePositive * 100) / totalExperimented
# print ("Accuracy = ", ans)
print('Total Predicted    : ', truePositive)
print('Unknown predicted  : ', totalExperimented-truePositive)
