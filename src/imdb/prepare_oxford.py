import numpy as np
import glob
import argparse
import pandas as pd
from tqdm import tqdm
try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

FOLDERS = { 
    "image_folder": "chunk_demosaic",
    "lidar_scan_folder": "lidar_scans",
    "lidar_project_folder":"lidar_projections",
    "labels_folder":"lidar_labels"
}
FOLDERS_EXT = { 
    "image_folder": ".png",
    "lidar_scan_folder": ".bin",
    "lidar_project_folder":".txt",
    "labels_folder":".csv"
}


def find_all_scans(dataset_dir):
    """
    function finds all scans in a folder

    dataset_dir: Pathlib, root directory of chunk

    returns
        list of scans found
    """

    scans_list = dataset_dir.glob("*"+FOLDERS_EXT["lidar_scan_folder"])
    scans_list = sorted([i.stem for i in scans_list])

    if len(scans_list) <= 0:
        print("no scans found in dir {dir}".format(dir=str(dataset_dir)))
        return None
    else:
        print("number of scans found is {num}".format(num=len(scans_list)))
        return scans_list

def read_lidar_scan(scan_path):
    """
    function reads lidar scan 

    scan_path: Pathlib, path to scan

    returns
        numpy array with shape [number of points, 3]
    """

    lidar_scan_data = np.fromfile(str(scan_path),dtype='float64')
    lidar_scan_data = lidar_scan_data.reshape(-1,3)  

    assert lidar_scan_data.shape[1] == 3, "invalid scan shape"

    return lidar_scan_data



def read_lidar_labels(labels_path):
    """
    function reads lidar labels path

    labels_path: Pathlib, path to lidar labels file

    returns
        numpy array with shape [number of points, 1]
    """

    lidar_labels_data = pd.read_csv(str(labels_path),header=None).to_numpy()
    lidar_labels_data = lidar_labels_data.astype(np.uint8)

    assert lidar_labels_data.shape[1] == 1, "invalid lidar labels shape"

    return lidar_labels_data



def normalize(x, new_min=0,new_max=1):
    """
    function normalize values in input array to given range

    Args:
        x: input array
        new_min: min value of new range
        new_max: max value of new range
    
    Returns:
        normalize array
    """
    new_value= (new_max-new_min)/(x.max()-x.min())*(x-x.max())+new_max
    return new_value

def cartesian_to_spherical(x, y, z):
    """
    function transforms cartesian coordinates into spherical ones

    Args:
        x: x axis values
        y: y axis values
        z: z axis values

    returns:
        spherical coordinate of every point in degrees

    """

    depth = np.sqrt(x**2 + y**2 + z**2)
    
    azimuth = np.arctan2(y,x)
    elevation = np.arcsin(z/depth)
    
    # transforms to degrees
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)
    
    return azimuth, elevation, depth

def spherical_projection(lidar_data, pgm_height= 4, pgm_width= 360, invert_z_axis=True):
    """
    function to spherically project a lidar scan

    Args:

        lidar_data: numpy array of shape [number of points, x, y, z] to be projected

        pgm_height:  desired height of pgm image, usually it is number of layers of lidar 

        pgm_width: desired width of pgm image, usually it is horizontal field of view of lidar / angular resolution, 
                sick lidar has 85 degrees HFoV, angular resolution in oxford data = 0.125, 85/0.125 = 360

        invert_z_axis: bool, should the z be inverted

    returns:

       azimuth amd elvation index values for lidar points + range values of points
    """


    # extract coordinate values
    x, y, z = lidar_data[:,0], lidar_data[:,1], lidar_data[:,2]

    if invert_z_axis:
        z = -1 * z

    # get spherical coordinates of points
    azimuth, elevation, depth = cartesian_to_spherical(x,y,z)

    # normalize angles to be between [0,1]
    azimuth_norm,elevation_norm = normalize(azimuth),normalize(elevation)

    # set points range to be inside desired width and height of pgm
    pgm_azimuth =  np.rint(azimuth_norm*(pgm_width-1)).astype(int), #[0, pgm_width-1]
    pgm_elevation = np.rint(elevation_norm*(pgm_height-1)).astype(int) #[0, pgm_height-1]

    return pgm_azimuth, pgm_elevation, depth

def main():
    parser = argparse.ArgumentParser(description='prepare oxford')
    
     #parameters
    parser.add_argument('--dataset_files', default='datasets/s3dis',help='path to dataset orignal files')
    parser.add_argument('--include', help='starting index to include')
    parser.add_argument('--output_dir', default='data',help='path to output dir')
    parser.add_argument('--pgm_height',help='number of layers produced by lidar')
    parser.add_argument('--pgm_width',help=' # sick lidar has 85 degrees HFoV, angular resolution of oxford 0.125, 85/0.125 = 360')

    args = parser.parse_args()
    
    root_dataset_dir = Path(args.dataset_files)
    start_index = int(args.include)
    output_dir = Path(args.output_dir)
    pgm_height = int(args.pgm_height)
    pgm_width = int(args.pgm_width)


    if not root_dataset_dir.exists():
        print("dataset dir does not exit")
        return None
    
    if not output_dir.exists():
        print("create output dir")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # find all images
    scans_names = find_all_scans(dataset_dir=root_dataset_dir / FOLDERS["lidar_scan_folder"] )

    if scans_names is not None:
        # remove glare images
        scans_names = scans_names[start_index:]

        print("number of scans after removing glare is {num}".format(num=len(scans_names)))

        for scan_num in tqdm(scans_names):

            # read lidar scan
            lidar_scan_path = root_dataset_dir / FOLDERS["lidar_scan_folder"] / ( scan_num + FOLDERS_EXT["lidar_scan_folder"] )
            lidar_scan_data = read_lidar_scan(lidar_scan_path) 

            # skip lidar scans that have nan values
            nn = lidar_scan_data[np.isnan(lidar_scan_data)]
            if not (nn.size == 0):
                continue

            # read lidar label
            lidar_labels_path = root_dataset_dir / FOLDERS["labels_folder"] / ( scan_num + FOLDERS_EXT["labels_folder"] )
            lidar_labels_data = read_lidar_labels(lidar_labels_path)


            pgm_azimuth, pgm_elevation, depth = spherical_projection(lidar_data= lidar_scan_data,
                                            pgm_height=pgm_height,
                                            pgm_width=pgm_width ,
                                            invert_z_axis=True)

            # result array to be saved
            
            # result shape [ pgm_width, pgm_height, x,y,z, intensity, range, label]
            result = np.zeros((pgm_height,pgm_width, 6), dtype=np.float)

            result[pgm_elevation,pgm_azimuth,:3] = lidar_scan_data
            result[pgm_elevation,pgm_azimuth,3] = 0 #intensity = 0
            result[pgm_elevation,pgm_azimuth,4] = depth
            result[pgm_elevation,pgm_azimuth,5] = lidar_labels_data.squeeze()

            
            #save file
            outfile = str(output_dir / scan_num )+ ".npy"
            np.save(outfile, result)




if __name__ == "__main__": 
    main()