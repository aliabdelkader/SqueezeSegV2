try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

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


def main():
  parser = argparse.ArgumentParser(description='calculate mean and std for oxford')

  #parameters
  parser.add_argument('--lidar_files', default='data/lidar_2d',help='path to lidar npy files')
  parser.add_argument('--pgm_height',help='number of layers produced by lidar')
  parser.add_argument('--pgm_width',help=' # sick lidar has 85 degrees HFoV, angular resolution of oxford 0.125, 85/0.125 = 360')
  parser.add_argument('--channels',help='number of channels in numpy')

  args = parser.parse_args()


  data_path = Path(args.lidar_files)
  pgm_height = int(args.pgm_height)
  pgm_width = int(args.pgm_width)
  channels = int(args.channels)

  validation_dir = "validation"

  validation_dir = Path(validation_dir)
  validation_dir.mkdir(parents=True, exist_ok=True)
  # find scans
  lidar_scans_files = list(data_path.glob("*.npy"))

  lidar_scans = np.zeros((len(lidar_scans_files),pgm_height,pgm_width,channels))

  for idx, lidar_scan_file in tqdm(enumerate(lidar_scans_files),"loading lidar files"):
      lidar_scan = np.load(str(lidar_scan_file))
      lidar_scans[idx] = lidar_scan
      # print(type(lidar_scan_file.stem))
      np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
      with open(str(validation_dir / (lidar_scan_file.stem + ".csv")), 'w') as f:
          f.write(np.array2string(lidar_scan, separator=', ').replace("[","").replace("]",""))
      # lidar = pd.DataFrame(lidar_scan, columns=['x','y','z','intensity','depth','label'])
      # lidar.to_csv(str(validation_dir / (lidar_scan_file.stem + ".csv")),header=False,index=False)
      # # np.savetxt(str(validation_dir / (lidar_scan_file.stem + ".csv")),lidar_scan,delimiter=",")

  ### get means
  x_mean = np.mean(lidar_scans[:,:,:,0])
  print("x mean {}".format(x_mean))

  y_mean = np.mean(lidar_scans[:,:,:,1])
  print("y mean {}".format(y_mean))

  z_mean = np.mean(lidar_scans[:,:,:,2])
  print("z mean {}".format(z_mean))

  intensity_mean = np.mean(lidar_scans[:,:,:,3])
  print("intensity mean {}".format(intensity_mean))

  range_mean = np.mean(lidar_scans[:,:,:,4])
  print("range mean {}".format(range_mean))


  ### get std
  x_std = np.std(lidar_scans[:,:,:,0])
  print("x std {}".format(x_std))

  y_std = np.std(lidar_scans[:,:,:,1])
  print("y std {}".format(y_std))

  z_std = np.std(lidar_scans[:,:,:,2])
  print("z std {}".format(z_std))

  intensity_std = np.std(lidar_scans[:,:,:,3])
  print("intensity std {}".format(intensity_std))

  range_std = np.std(lidar_scans[:,:,:,4])
  print("range std {}".format(range_std))

  print("############# labels freq #############################3")

  labels = lidar_scans[:,:,:,5]
  
  uniqueValues, occurCount = np.unique(labels.reshape(-1), return_counts=True)
  freq = list(zip(uniqueValues,occurCount))
  freq = sorted(freq, key=lambda x: x[1],reverse=True)
  keys = list(semantic_colors_dict.keys())
  for clss_idx,count in freq:
    class_name = keys[int(clss_idx)]
    print("{}: {}".format(class_name,count))

  return None

if __name__ == "__main__": 
    main()