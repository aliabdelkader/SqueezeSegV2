# Author: Xuanyu Zhou (xuanyu_zhou@berkeley.edu), Bichen Wu (bichen@berkeley.edu) 10/27/2018

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def oxford_squeezeSeg_config():

  """Specify the parameters to tune below."""
  mc  = base_model_config('KITTI')

  mc.DEBUG_MODE = True

  mc.LOAD_PRETRAINED_MODEL = True

  semantic_colors_dict = {
        'road'          : [128, 64,128],
        'sidewalk'      : [244, 35,232],
        'building'      : [ 70, 70, 70],
        'wall'          : [102,102,156],
        'fence'         : [190,153,153],
        'pole'          : [153,153,153],
        'traffic_light' : [250,170, 30],
        'traffic_sign'  : [220,220,  0],
        'vegetation'    : [107,142, 35],
        'terrain'       : [152,251,152],
        'sky'           : [ 70,130,180],
        'person'        : [220, 20, 60],
        'rider'         : [255,  0,  0],
        'car'           : [  0,  0,142],
        'truck'         : [  0,  0, 70],
        'bus'           : [  0, 60,100],
        'train'         : [  0, 80,100],
        'motorcycle'    : [  0,  0,230],
        'bicycle'       : [119, 11, 32],
        'void'          : [  0,  0,  0],
        'outside_camera': [255, 255, 0],
        'egocar'        : [123, 88,  4],
        #'unlabelled'    : [ 81,  0, 81]
	}

  mc.CLASSES            = list(semantic_colors_dict.keys())#['unknown', 'car', 'pedestrian', 'cyclist']
  mc.NUM_CLASS          = len(mc.CLASSES)
  print("number of classes {}".format(mc.NUM_CLASS))
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.ones((mc.NUM_CLASS))#np.array([1/3.0, 1.0, 3.5, 3.5])
  print(np.ones((mc.NUM_CLASS)),np.ones((mc.NUM_CLASS)).shape)
#   mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
#                                     [ 0.12,  0.56,  0.37],
#                                     [ 0.66,  0.55,  0.71],
#                                     [ 0.58,  0.72,  0.88]])
  mc.CLS_COLOR_MAP      = np.array(list(semantic_colors_dict.values()))

  mc.BATCH_SIZE         = 40
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 4

  mc.FOCAL_GAMMA        = 2.0
  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.ones((mc.NUM_CLASS))*0.6 #np.array([.9, .9, .6, .6])
  mc.BILATERAL_THETA_R  = np.ones((mc.NUM_CLASS))*0.01#np.array([.015, .015, .01, .01])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.ones((mc.NUM_CLASS))*0.6 #np.array([.9, .9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.05
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance

  # run calculate_stats to get this data

  
# x mean 7.24533976482
# y mean 0.452658387756
# z mean 0.0374274338247
# intensity mean 0.0
# range mean 7.97056021518
# x std 14.5127064397
# y std 5.25185277249
# z std 0.947047562264
# intensity std 0.0 
# range std 15.1085711908

# intensity set to 1e-5 to avoid divide by zero
  
  mc.INPUT_MEAN         = np.array([[[7.24, 0.45, 0.03, 1e-5, 7.97]]])
  mc.INPUT_STD          = np.array([[[14.51,5.25, 0.94, 1e-5, 15.10]]])

  return mc
