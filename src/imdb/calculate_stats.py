try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

from tqdm import tqdm
import numpy as np
import argparse


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
  # find scans
  lidar_scans_files = list(data_path.glob("*.npy"))

  lidar_scans = np.zeros((len(lidar_scans_files),pgm_height,pgm_width,channels))

  for idx, lidar_scan_file in tqdm(enumerate(lidar_scans_files),"loading lidar files"):
      lidar_scan = np.load(str(lidar_scan_file))
      lidar_scans[idx] = lidar_scan

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

  return None

if __name__ == "__main__": 
    main()