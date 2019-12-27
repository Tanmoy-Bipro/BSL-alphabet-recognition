import cv2
import os
import glob
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = dir_path + '/' + image_path + '/'
path = os.path.join('test_own_data', filename, '*g')
files = glob.glob(path)
#print(files)

# classes = ['001','002','003', '004', '006', '009', '012', '014', '015', '016', '16_02', '16_06',
#            '16_14', '16_15', '16_17', '16_18', '16_19', '16_24', '16_45', '16_47', '16_52', '16_53', '16_58', '16_62',
#            '16_64', '16_71', '16_76', '16_78', '16_82', '16_85', '16_90', '16_94', '16_98', '16_100', '16_103',
#            '16_105', '16_106', '16_107', '16_252', '16_949']

classes = ['A','B','C','D','E']
num_classes = len(classes)
image_size = 128
num_channels = 3

totalExperimented =0;
truePositive=0;
#decisionMartrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
for i in files:

    filename = i

    #print(i)
    p = 0
    className = ""
    chk2=0;
    i = i[::-1]
    for kk in i:
        if kk == '.':
            chk2=1;
        elif kk == '\\':
            break;
        elif kk == '/':
            break;
        elif chk2 == 1:
            className = className + kk;

    # for k in i:
    #     if i[p] == 't':
    #         if i[p+1] == 'e':
    #             if i[p+2] == 's':
    #                 if i[p+3] == 't':
    #                     q=p+5
    #                     r=0
    #                     for l in i:
    #                         className = className + i[q]
    #                         q=q+1
    #                         r=r+1
    #                         if r == 3:
    #                             break
    #
    #                 break
    #
    #     # print(i[p])
    #     # print(p)
    #     p = p + 1

    # image_path=sys.argv[1]
    # filename = dir_path +'/'+'test/bag.1.jpg'






    #className = className
    className = className[::-1]
    #print(className)

    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('AI-signature-verification.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

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


    # inputClass = 0
    # #print(className)
    # if className == classes[0]:
    #     inputClass = 0
    # elif className == classes[1]:
    #     inputClass = 1
    # elif className == classes[2]:
    #     inputClass = 2
    # elif className == classes[3]:
    #     inputClass = 3
    # elif className == classes[4]:
    #     inputClass = 4
    # elif className == classes[5]:
    #     inputClass = 5
    # elif className == classes[6]:
    #     inputClass = 6
    # elif className == classes[7]:
    #     inputClass = 7
    # elif className == classes[8]:
    #     inputClass = 8
    # elif className == classes[9]:
    #     inputClass = 9
    # m1 = max(result)

    #print(inputClass)
    #m = max(m1)
    j = 1;
    val =0;
    probableClass = 0;
    for ii in result:
        chk=0
        for jj in ii:
            chk=chk+1
            if jj>val:
                val =jj
                probableClass = chk
    totalExperimented = totalExperimented +1
    if val >= 0.90:

        print('Photo Name       : ', className)
        print('predicted class : ', classes[probableClass - 1])
        print('Probability value: ', val)
        #print('Match with       : ', probableClass)
        truePositive = truePositive + 1
    else:
        print('Photo Name       : ', className)
        print('predicted class : Unknown')
        print('Probability value: ', val)
        #print('Match with       : ', probableClass)

    print()
    print()
    # for i in m1:
    #     if i == m:
    #         break
    #     j = j + 1




    # print(m)


    # ans= m.argmax(axis=0)
    # print(m)

    # print(i)

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
