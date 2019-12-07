# organize imports
import cv2
import imutils
import os
import time
import threading
import numpy as np

# global variables
bg = None
count = 0
#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=10):
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
#-------------------------------------------------------------------------------
# Function to capture data
#-------------------------------------------------------------------------------
def img_cap():
    count = 0 ########################################## Start value for image serial################################
    path = 'F:/Image Collect from videoFeed/Dataset/C/'#PATHHHHHHHHHHHHHHHHHHHHHHHHHH for Image F:\Image Collect from videoFeed\Dataset\C
    while (True):
        fname = "c" + str(count) + ".jpg" ########################################## NAME for image serial###########
        cv2.imwrite(os.path.join(path, fname), skinflip)
        count += 1
        time.sleep(0.2) #num of frames per sec
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("x"):
            break

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam###########################################################################
    camera = cv2.VideoCapture(1)  

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 300, 300, 590
    length = left-right
    width = bottom-top

    # initialize num of frames
    num_frames = 0
    t1 = threading.Thread(target=img_cap)
    thread_start = False
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700, height=600)
        # cv2.imshow('initial', frame)
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        # cv2.imshow('flipped', frame)
        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]
        # rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb=roi
        # cv2.imshow('RGB region of interest', rgb)
        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', gray)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # cv2.imshow('Blur', gray)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
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
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                # cv2.drawContours(mask, [segmented + (right, top)], 0, 255, -2)
                # mean = cv2.mean(clone, mask=mask)
                # cv2.imshow("Masked", mask)
                # cv2.imshow("Original", rgb)
                skin=cv2.bitwise_and(rgb, rgb, mask=thresholded)
                # cv2.imshow("Thesholded", thresholded)
                cv2.imshow("skin", skin)
                skinflip = cv2.flip(skin, 1)
                cv2.imshow("flipped", skinflip)


                if keypress == ord("c") and thread_start==False:
                    thread_start = True
                    t1.daemon = True
                    t1.start()


        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            # t1._stop()
            break
    # free up memory
    # t1.join()
    camera.release()
    cv2.destroyAllWindows()

