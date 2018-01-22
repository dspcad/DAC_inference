#!/usr/bin/python

import numpy as np
import os
import csv
import tensorflow as tf
import time
from PIL import Image


def drawBBox(img, pred_coordinates, ground_truth_coordinates):
  xmin = int(clipWidth(pred_coordinates[0])) - 1
  xmax = int(clipWidth(pred_coordinates[1])) - 1
  ymin = int(clipHeight(pred_coordinates[2])) - 1
  ymax = int(clipHeight(pred_coordinates[3])) - 1

  #print "image shape: ", img.shape
  #print "coordinate: ", coordinates

  for i in range(0,xmax-xmin):
    img[ymin][xmin+i][0] = 255
    img[ymin][xmin+i][1] = 0
    img[ymin][xmin+i][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmin][0] = 255
    img[ymin+i][xmin][1] = 0
    img[ymin+i][xmin][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmax][0] = 255
    img[ymin+i][xmax][1] = 0
    img[ymin+i][xmax][2] = 0

  for i in range(0,xmax-xmin):
    img[ymax][xmin+i][0] = 255
    img[ymax][xmin+i][1] = 0
    img[ymax][xmin+i][2] = 0

  xmin = int(ground_truth_coordinates[0]) - 1
  xmax = int(ground_truth_coordinates[1]) - 1
  ymin = int(ground_truth_coordinates[2]) - 1
  ymax = int(ground_truth_coordinates[3]) - 1

  #print "image shape: ", img.shape
  #print "coordinate: ", coordinates

  for i in range(0,xmax-xmin):
    img[ymin][xmin+i][0] = 0
    img[ymin][xmin+i][1] = 128
    img[ymin][xmin+i][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmin][0] = 0
    img[ymin+i][xmin][1] = 128
    img[ymin+i][xmin][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmax][0] = 0
    img[ymin+i][xmax][1] = 128
    img[ymin+i][xmax][2] = 0

  for i in range(0,xmax-xmin):
    img[ymax][xmin+i][0] = 0
    img[ymax][xmin+i][1] = 128
    img[ymax][xmin+i][2] = 0


  return img



def clipWidth(val):
  if val > 640:
    return 640

  if val < 1:
    return 1

  return val

def clipHeight(val):
  if val > 360:
    return 360

  if val < 1:
    return 1

  return val


def checkIOU(label_BBox, pred_BBox):
  #print "label_BBox shape: ", label_BBox.shape
  #print "pred_BBox shape: ", pred_BBox.shape

  IOU = np.zeros(label_BBox.shape[0])

  for i in range(label_BBox.shape[0]):
    ###############################
    #  check validity of pred box #
    ###############################
    if pred_BBox[i][0] >= pred_BBox[i][1] or pred_BBox[i][2] >= pred_BBox[i][3]:
      IOU[i] = 0
    else:
      if checkIntersection(label_BBox[i], pred_BBox[i]) == 1:


        xmin_A = clipWidth(pred_BBox[i][0])
        xmax_A = clipWidth(pred_BBox[i][1])
        ymin_A = clipHeight(pred_BBox[i][2])
        ymax_A = clipHeight(pred_BBox[i][3])


        xmin_B = label_BBox[i][0]
        xmax_B = label_BBox[i][1]
        ymin_B = label_BBox[i][2]
        ymax_B = label_BBox[i][3]


        xmin_intersection = np.maximum(xmin_A, xmin_B)
        xmax_intersection = np.minimum(xmax_A, xmax_B)
        ymin_intersection = np.maximum(ymin_A, ymin_B)
        ymax_intersection = np.minimum(ymax_A, ymax_B)

        intersection_area = (xmax_intersection-xmin_intersection)*(ymax_intersection-ymin_intersection)
        area_two_boxes = (xmax_A-xmin_A)*(ymax_A-ymin_A) + (xmax_B-xmin_B)*(ymax_B-ymin_B)
        IOU[i] = intersection_area/(area_two_boxes-intersection_area) 

        #print "intersection_area: ", intersection_area
        #print "area_two_boxes: ", area_two_boxes
        #print "IOU[%d]: %f" % (i, IOU[i])
      else:
        IOU[i] = 0
      

  #print "xmin_union: ", xmin_union
  #print "ymin_union: ", ymin_union
  #print "xmax_union: ", xmax_union
  #print "ymax_union: ", ymax_union


    
  #print IOU
  return IOU

def checkIntersection(BBoxA, BBoxB):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################

  xmin = BBoxB[0] 
  xmax = BBoxB[1]
  ymin = BBoxB[2]
  ymax = BBoxB[3]


  #########################################
  #     BBox intersects the grid cells    #
  #########################################
  target_x = BBoxA[0] #xmin
  target_y = BBoxA[2] #ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[0] #xmin
  target_y = BBoxA[3] #ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[1] #xmax
  target_y = BBoxA[3] #ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[1] #xmax
  target_y = BBoxA[2] #ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1
 
 
  #########################################
  #       BBoxB is within in BBoxA        #
  #########################################
  if xmin >= BBoxA[0] and ymin >= BBoxA[2] and xmax <= BBoxA[1] and ymax <= BBoxA[3]:
    return 1


  return 0


if __name__ == '__main__':
  print '===== Start the inference for 2018 DAC competition ====='
  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 128

  K = 98 # number of classes
  G = 576 # number of grid cells
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
  #W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K], initializer=tf.contrib.layers.xavier_initializer())
  W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K*G], initializer=tf.contrib.layers.xavier_initializer())



  b1  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
  b2  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
  b3  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
  b4  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
  b5  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
  b6  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_6], dtype=tf.float32), trainable=True, name='b6')
  b9  = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b9')
  b10 = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b10')
  #b11 = tf.Variable(tf.constant(0.1, shape=[K], dtype=tf.float32), trainable=True, name='b11')
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
  Y_bbox = tf.matmul(tf.nn.relu(Y),W_bbox)+b_bbox



  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  valid_datapath = "/home/nvidia/tensorflow_work/DAC_inference/images"

  file_list = []
  for dirpath, dirnames, filenames in os.walk(valid_datapath):
    print "The number of files: %d" % len(filenames)
    #print filenames

    file_list = filenames

  img_list = []
  for f_name in file_list:
    f_path = "/home/nvidia/tensorflow_work/DAC_inference/images/%s" % f_name
    img_list.append(np.array(Image.open(f_path)))

  
#  batch_imgs = []
#  for i in range(0,10):
#    if len(batch_imgs) == 0:
#      batch_imgs = img_list[i]
#    else:
#      batch_imgs = np.vstack((batch_imgs, img_list[i]))
#
#
#  batch_imgs = batch_imgs.reshape(10,360,640,3)
#  print batch_imgs.shape


  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction=0.4
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, "/home/nvidia/tensorflow_work/TX2_tracking/checkpoint/model_small_trained.ckpt")
    print "Model %s restored." % ("model_small_trained")


    batch_imgs = np.zeros((50,360,640,3))
    print "Inference Starts..."
    valid_accuracy = 0.0
    valid_IOU = 0.0
    time_start=time.time()
    for i in range(0,20):
      #batch_imgs = []
      for j in range(0,50):
        #if len(batch_imgs) == 0:
        #  batch_imgs = img_list[i*100+j]
        #else:
        #  batch_imgs = np.vstack((batch_imgs, img_list[i*100+j]))
        #print img_list[i*100+j].shape
        #print batch_imgs.shape
        batch_imgs[j,:,:,:] = img_list[i*50+j]


      #batch_imgs = batch_imgs.reshape(100,360,640,3)
      pred_bbox = Y_bbox.eval(feed_dict={X: batch_imgs})
      #print pred_bbox

   
      #valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord})
      #valid_IOU += np.mean(checkIOU(box_coord, pred_bbox))
      print "%d batch is done" % i

      #for j in range(0, 100):
      #  bbox_image = drawBBox(test_x[j],pred_bbox[j], box_coord[j])
      #  #print "Image: ", bbox_image
      #  #print np.argmax(test_y[j])
      #  #print look_up_label_dict[np.argmax(test_y[j])]
      #  io.imsave("%s_%d_%s.%s" % ("./val_images/test_img", 100*i+j, look_up_label_dict[np.argmax(test_y[j])], 'jpg'), test_x[j]/256.0)

    time_end = time.time()
    resultRunTime = time_end-time_start
    print "Spent time: ", resultRunTime  
    print "FPS: ", 1000/resultRunTime  
    print "Validation Accuracy: %f (%.1f/10000)" %  (valid_accuracy/10000, valid_accuracy)
    print "Validation Mean IOU: %f (%.1f/100)" %  (valid_IOU/100, valid_IOU)



    sess.close()



