from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import imutils
from PIL import ImageFont, ImageDraw, Image
import copy
import numpy as np
from PIL import Image
# ...

#image = Image.open("example.png")
#image_array = np.array(image)[:, :, 0:3]  # Select RGB channels only.

#prediction = sess.run(softmax_tensor, {'DecodeJpeg:0': image_array})

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
bg = None
count = 0
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
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


# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
def read_tensor_from_image_file(file_name, input_height=224, input_width=224,
				input_mean=0, input_std=224):
  input_name = "file_reader"
  output_name = "normalized"
  
  file_reader = tf.read_file(file_name, input_name)
  image_reader = tf.image.decode_jpeg(file_name, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result
def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile('/tmp/output_labels.txt')]

# Unpersists graph from file
with tf.gfile.FastGFile('/tmp/output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction

    #softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
graph = load_graph('/tmp/output_graph.pb')
with tf.Session(graph=graph) as sess:
    #start = time.time()
    c = 0
    aWeight = 0.5
    cap = cv2.VideoCapture(0)

    res, score = '', 0.0
    i = 4
    j = 0
    mem = ''
    consecutive = 0
    sequence = ''
    num_frames = 0
    b, g, r, a = 0, 255, 0, 0
    fontpath = "kalpurush.ttf"
    font = ImageFont.truetype(fontpath, 100)
    label_file = '/tmp/output_labels.txt'
    
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
                    # cv2.imshow("Thesholded", thresholded)
                    cv2.imshow("skin", skin)
                    skinflip = cv2.flip(skin, 1)
                    cv2.imshow("Cropped/flipped", skinflip)
                    img_cropped = skinflip
            num_frames += 1
            c += 1
            #cv2.imshow("Cropped", img_cropped)
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)
            cv2.imshow("img", img)
            if i == 4:
                #res_tmp, score = predict(image_data)
                t = read_tensor_from_image_file(image_data,
                                  input_height=224,
                                  input_width=224,
                                  input_mean=0,
                                  input_std=224)

                input_name = "import/" + 'input'#input_layer
                output_name = "import/" + 'final_result'#output_layer
                input_operation = graph.get_operation_by_name(input_name);
                output_operation = graph.get_operation_by_name(output_name);

			  #with tf.Session(graph=graph) as sess:
				#start = time.time()
                results = sess.run(output_operation.outputs[0],
								  {input_operation.outputs[0]: t})
			  #end=time.time()
                results = np.squeeze(results)
			  
                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)

			  #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))##############
                template = "{} (score={:0.5f})"
            for j in top_k:
                print(template.format(labels[j], results[j]))#########################
                
				#res = res_tmp
                #i = 0
                #if mem == res:
                    #consecutive += 1
                #else:
                    #consecutive = 0
                #if consecutive == 2 and res not in ['nothing']:
                    #if res == 'space':
                        #sequence += ' '
                    #elif res == 'del':
                        #sequence = sequence[:-1]
                    #else:
                        #sequence += res
                    #consecutive = 0
            #i += 1
		    #if res == 'circle':
			    #res = '\xe0\xa6\xad\xe0\xa6\xbe\xe0\xa6\xb2'
			#'''
			#TestText = "Test -ভাল āĀēĒčČ..šŠūŪžŽ"
            #TestText = "ভাল" # this NOT utf-8...it is a Unicode string in Python 3.X.
            #TestText2 = TestText.encode('utf8') # THIS is "just bytes" in UTF-8.
			#sys.stdout.buffer.write(TestText2)
			#'''
'''
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            if res.upper() == 'A':
                #img_pil = Image.fromarray(img)
                #draw = ImageDraw.Draw(img_pil)
                # draw.text((50, 100),  "国庆节/中秋节 快乐!", font = font, fill = (b, g, r, a))
                draw.text((100, 300), "অ", font=font, fill=(b, g, r, a))
                img = np.array(img_pil)
            else:
                #img_pil = Image.fromarray(img)
                #draw = ImageDraw.Draw(img_pil)
                # draw.text((50, 100),  "国庆节/中秋节 快乐!", font = font, fill = (b, g, r, a))
                draw.text((100, 300), "আ", font=font, fill=(b, g, r, a))
                img = np.array(img_pil)
            # cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            mem = res
            cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)
            cv2.imshow("img", img)
            img_sequence = np.zeros((200,1200,3), np.uint8)
            #cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            #cv2.imshow('sequence', img_sequence)
            
            if a == 27: # when `esc` is pressed
                break
'''
#end=time.time()###########
# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
