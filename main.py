## This program is for DAC HDC contest ######
## 2017/11/22
## xxu8@nd.edu
## University of Notre Dame

import math
import numpy as np
import time
import tensorflow as tf
from PIL import Image
import procfunc
#### !!!! you can import any package needed for your program ######

if __name__ == "__main__":
    ############### configurations for dir #################################################################################
    ## Folder structure:
    ## $DAC$|
    ##    |images   (all the test images are stored in this folder)
    ##    |results-$teamName$|
    ##            |time
    ##            |xml

    ## !!!! Please specify your team name here
    teamName = 'hhwu'
    ## !!!! please specify the dir here, and please put all the images for test in the folder "images".
    ## Important! You can specify the folder in your local test. But for the sumission, DAC folder is fixed as follows
    #DAC = '/home/DACSDC_GPU' ## uncomment this line when submitting your code
    DAC = '/home/nvidia/tensorflow_work/DAC_inference/'
    [imgDir, resultDir, timeDir, xmlDir, myXmlDir, allTimeFile] = procfunc.setupDir(DAC, teamName)

    ############### processing for object detection and tracking ###########################################################
    ### load all the images names
    [allImageName, imageNum] = procfunc.getImageNames(imgDir)
    ### process all the images in batch
    batchNumDiskToDram = 50 ## the # of images read from disk to DRAM in one time
    batchNumDramToGPU  = 10 ## the # of images read from DRAM to GPU in one time for batch processing on the GPU
    imageReadTime = int(math.ceil(imageNum/batchNumDiskToDram))
    imageProcTimeEachRead = int(math.ceil(batchNumDiskToDram/batchNumDramToGPU))
    resultRectangle = np.zeros((imageNum, 4)) ## store all the results about tracking accuracy



    #########################################################################################################################
    K = 98 # number of classes
    G = 576 # number of grid cells
    #G = 48 # number of grid cells
    P = 4  # four parameters of the bounding boxes
    NUM_FILTER_1 = 32
    NUM_FILTER_2 = 32
    NUM_FILTER_3 = 64		 
    NUM_FILTER_4 = 64		 
    NUM_FILTER_5 = 128
    NUM_FILTER_6 = 128
  
    NUM_NEURON_1 = 1024
    NUM_NEURON_2 = 1024
  
  
  
  
    # initialize parameters randomly
    X      = tf.placeholder(tf.float32, shape=[None, 360,640,3])
    Y_BBOX = tf.placeholder(tf.float32, shape=[None,P])
  
  
    W1  = tf.get_variable("W1", shape=[6,10,3,NUM_FILTER_1], initializer=tf.contrib.layers.xavier_initializer())
    W2  = tf.get_variable("W2", shape=[3,3,NUM_FILTER_1,NUM_FILTER_2], initializer=tf.contrib.layers.xavier_initializer())
    W3  = tf.get_variable("W3", shape=[3,3,NUM_FILTER_2,NUM_FILTER_3], initializer=tf.contrib.layers.xavier_initializer())
    W4  = tf.get_variable("W4", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], initializer=tf.contrib.layers.xavier_initializer())
    W5  = tf.get_variable("W5", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], initializer=tf.contrib.layers.xavier_initializer())
    W6  = tf.get_variable("W6", shape=[3,3,NUM_FILTER_5,NUM_FILTER_6], initializer=tf.contrib.layers.xavier_initializer())
  
    W9  = tf.get_variable("W9", shape=[23*27*NUM_FILTER_6,NUM_NEURON_1], initializer=tf.contrib.layers.xavier_initializer())
    W10 = tf.get_variable("W10", shape=[NUM_NEURON_1,NUM_NEURON_2], initializer=tf.contrib.layers.xavier_initializer())
    W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K*G], initializer=tf.contrib.layers.xavier_initializer())
  
  
  
    b1  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
    b2  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
    b3  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
    b4  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
    b5  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
    b6  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_6], dtype=tf.float32), trainable=True, name='b6')
    b9  = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b9')
    b10 = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b10')
    b11 = tf.Variable(tf.constant(0.1, shape=[K*G], dtype=tf.float32), trainable=True, name='b11')
  
    matrix_w = np.zeros((K*G,K))
    for i in range(0,K):
      for j in range(0,G):
        matrix_w[i*G+j][i] = 1
  
    label_pred_transform_W = tf.constant(matrix_w, shape=matrix_w.shape, dtype=tf.float32)
  
  
    W_bbox = tf.get_variable("W_bbox", shape=[K*G,P], initializer=tf.contrib.layers.xavier_initializer())
    b_bbox = tf.Variable(tf.constant(0.1, shape=[P], dtype=tf.float32), trainable=True, name='b_bbox')
  
    #===== architecture =====#
    conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,2,3,1], padding='VALID')+b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')+b2)
    pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  
  
    conv3 = tf.nn.relu(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME')+b3)
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
    pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  
    conv5 = tf.nn.relu(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME')+b5)
    conv6 = tf.nn.relu(tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME')+b6)
    pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  
  
  
    print "conv1: ", conv1.get_shape()
    print "conv2: ", conv2.get_shape()
    print "pool1: ", pool1.get_shape()
  
    print "conv3: ", conv3.get_shape()
    print "conv4: ", conv4.get_shape()
    print "pool2: ", pool2.get_shape()
  
    print "conv5: ", conv5.get_shape()
    print "conv6: ", conv6.get_shape()
    print "pool3: ", pool3.get_shape()
  
    YY = tf.reshape(pool3, shape=[-1,23*27*NUM_FILTER_6])
  
    fc1 = tf.nn.relu(tf.matmul(YY,W9)+b9)
    fc2 = tf.nn.relu(tf.matmul(fc1,W10)+b10)
    Y = tf.matmul(fc2,W11)+b11
  
    Y_class = tf.matmul(Y,label_pred_transform_W)
    Y_bbox = tf.round(tf.matmul(tf.nn.relu(Y),W_bbox)+b_bbox)


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.4
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
 
      # Restore variables from disk.
      model_name = "/home/nvidia/tensorflow_work/TX2_tracking/checkpoint/model_small_trained.ckpt"
      saver.restore(sess, model_name)
      print "Model %s restored." % (model_name)


      time_start=time.time()
      for i in range(imageReadTime):
          #print imgDir,allImageName, imageNum, i, batchNumDiskToDram
          ImageDramBatch = procfunc.readImagesBatch(imgDir,allImageName, imageNum, i, batchNumDiskToDram)
          print ImageDramBatch.shape
          for j in range(imageProcTimeEachRead):
              start = j*batchNumDramToGPU
              end = start + batchNumDramToGPU
              if end > len(ImageDramBatch):
                  end = len(ImageDramBatch)
                  if end < start:
                      break
              inputImageData = ImageDramBatch[start:end, :,:,:]
              ############ !!!!!!!!!! your detection and tracking code, please revise the function: detectionAndTracking() !!!!!!!############
              #resultRectangle[i * batchNumDiskToDram + start:i * batchNumDiskToDram + end, :] = procfunc.detectionAndTracking(inputImageData, end-start)
              resultRectangle[i * batchNumDiskToDram + start:i * batchNumDiskToDram + end, :] = Y_bbox.eval(feed_dict={X: inputImageData})
      time_end = time.time()
      resultRunTime = time_end-time_start
      ############### write results (write time to allTimeFile and detection results to xml) #################################
      procfunc.storeResultsToXML(resultRectangle, allImageName, myXmlDir)
      procfunc.write(imageNum,resultRunTime,teamName, allTimeFile)
