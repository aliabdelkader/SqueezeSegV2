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
    "lidar_project_folder": "lidar_projections",
    "labels_folder": "lidar_labels"
}
FOLDERS_EXT = {
    "image_folder": ".png",
    "lidar_scan_folder": ".bin",
    "lidar_project_folder": ".txt",
    "labels_folder": ".csv"
}

label_map = {

    0: 1,  # 'road': [128, 64, 128]
    1: 2,  # 'sidewalk': [244, 35, 232]
    2: 3,  # 'building': [70, 70, 70],
    3: 4,  # 'wall': [102, 102, 156],
    4: 5,  # 'fence': [190, 153, 153],
    5: 6,  # 'pole': [153, 153, 153],
    6: 7,  # 'traffic light': [250, 170, 30],
    7: 8,  # 'traffic sign': [220, 220, 0],
    8: 9,  # 'vegetation': [107, 142, 35],
    9: 10,  # 'terrain': [152, 251, 152],
    10: 11,  # 'sky': [70, 130, 180],
    11: 12,  # 'person': [220, 20, 60],
    12: 13,  # 'rider': [255, 0, 0],
    13: 14,  # 'car': [0, 0, 142],
    14: 15,  # 'truck': [0, 0, 70],
    15: 16,  # 'bus': [0, 60, 100],
    16: 17,  # 'train': [0, 80, 100],
    17: 18,  # 'motorcycle': [0, 0, 230],
    18: 19,  # 'bicycle': [119, 11, 32],
    19: 0,  # 'void': [0, 0, 0],
    20: 0,  # 'outside camera': [255, 255, 0],
    # 21: 0, #'egocar': [123, 88, 4],
}

def find_all_scans(dataset_dir):
    """
    function finds all scans in a folder

    dataset_dir: Pathlib, root directory of chunk

    returns
        list of scans found
    """

    scans_list = dataset_dir.glob("*" + FOLDERS_EXT["lidar_scan_folder"])
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

    lidar_scan_data = np.fromfile(str(scan_path), dtype='float64')
    lidar_scan_data = lidar_scan_data.reshape(-1, 3)

    assert lidar_scan_data.shape[1] == 3, "invalid scan shape"

    return lidar_scan_data


def read_lidar_labels(labels_path):
    """
    function reads lidar labels path

    labels_path: Pathlib, path to lidar labels file

    returns
        numpy array with shape [number of points, 1]
    """

    lidar_labels_data = pd.read_csv(str(labels_path), header=None).to_numpy()
    lidar_labels_data = lidar_labels_data.astype(np.uint8)

    assert lidar_labels_data.shape[1] == 1, "invalid lidar labels shape"

    return lidar_labels_data


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

    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / depth)

    # transforms to degrees
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)

    return azimuth, elevation, depth


def shift_range(x, new_min=0., new_max=1., old_min=None, old_max=None):
    """
    function shift values in array from an old range to new range
    Args:
        x: input array
        new_min: min value of new range
        new_max: max value of new range
        old_min: min value of input range
        old_max: max value of input range
    Returns:
        normalize array
    """
    if old_max and old_min:
        new_value = (new_max - new_min) / (old_max - old_min) * (x - old_max) + new_max
    else:
        new_value = (new_max - new_min) / (x.max() - x.min()) * (x - x.max()) + new_max
    return new_value


def get_pgm_index(lidar_data, pgm_height, pgm_width, vertical_field_view, horizontal_field_view, invert_z_axis=True):
    """
    function get polar grid map coordinates for every point in lidar scan
    Args:
        lidar_data: numpy array of lidar scan
        pgm_height: height of polar grid map
        pgm_width: width of polar grid map
        vertical_field_view:  tuple
            vertical field of view of lidar, angle of highest layer, angle of lowest layer
            for sick: (-1.5,1.5)
            for velodyne 64: (-25, 3)
        horizontal_field_view: horizontal field of view of layer, angles of extrem laser beans
            for sick (-42.5, 42.5)
            for velodyne 64: (-180, 180), for kitti (-90, 90)
        invert_z_axis: boolean, flag wether to invert z axis

    return:
       pgm_azimuth_idx, pgm_elevation_idx, depth: angles for every lidar point and its depth
    """

    # extract coordinate values
    x, y, z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]

    if invert_z_axis:
        z = -1 * z

    # get spherical coordinates of points
    azimuth, elevation, depth = cartesian_to_spherical(x, y, z)

    azimuth = shift_range(azimuth, new_min=vertical_field_view[0], new_max=vertical_field_view[1])
    elevation = shift_range(elevation, new_min=horizontal_field_view[0], new_max=horizontal_field_view[1])

    vertical_step = (abs(vertical_field_view[0]) + abs(vertical_field_view[1])) / pgm_height
    horizontal_step = (abs(horizontal_field_view[0]) + abs(horizontal_field_view[1])) / pgm_width

    vertical_range = np.arange(vertical_field_view[0], vertical_field_view[1], vertical_step)
    horizontal_range = np.arange(horizontal_field_view[0], horizontal_field_view[1], horizontal_step)

    elevation_diff = np.abs(elevation.reshape(-1, 1) - vertical_range.reshape(1, -1))
    azimuth_diff = np.abs(azimuth.reshape(-1, 1) - horizontal_range.reshape(1, -1))

    pgm_azimuth_idx = np.argmin(azimuth_diff, axis=1).squeeze()
    pgm_elevation_idx = np.argmin(elevation_diff, axis=1).squeeze()

    return pgm_azimuth_idx, pgm_elevation_idx, depth


def convert_ground_truth(original_label):
    """
    function changes ground truth label according to lidar map
    Args:
        original_label: label from annotation files
    return:
        new label

    """

    return label_map[original_label]
def main():
    parser = argparse.ArgumentParser(description='prepare oxford')

    # parameters
    parser.add_argument('--dataset_files', default='datasets/s3dis', help='path to dataset orignal files')
    parser.add_argument('--include', help='starting index to include')
    parser.add_argument('--output_dir', default='data', help='path to output dir')
    parser.add_argument('--pgm_height', help='number of layers produced by lidar')
    parser.add_argument('--pgm_width',
                        help=' # sick lidar has 85 degrees HFoV, angular resolution of oxford 0.125, 85/0.125 = 360')

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
    scans_names = find_all_scans(dataset_dir=root_dataset_dir / FOLDERS["lidar_scan_folder"])

    if scans_names is not None:
        # remove glare images
        scans_names = scans_names[start_index:]

        print("number of scans after removing glare is {num}".format(num=len(scans_names)))

        for scan_num in tqdm(scans_names):

            # read lidar scan
            lidar_scan_path = root_dataset_dir / FOLDERS["lidar_scan_folder"] / (
                    scan_num + FOLDERS_EXT["lidar_scan_folder"])
            lidar_scan_data = read_lidar_scan(lidar_scan_path)

            # skip lidar scans that have nan values
            nn = lidar_scan_data[np.isnan(lidar_scan_data)]
            if not (nn.size == 0):
                continue

            # read lidar label
            lidar_labels_path = root_dataset_dir / FOLDERS["labels_folder"] / (scan_num + FOLDERS_EXT["labels_folder"])
            lidar_labels_data = read_lidar_labels(lidar_labels_path)

            pgm_azimuth, pgm_elevation, depth = get_pgm_index(lidar_data=lidar_scan_data,
                                                                      pgm_height=pgm_height,
                                                                      pgm_width=pgm_width,
                                                                      vertical_field_view=(-1.5, 1.5),
                                                                      horizontal_field_view=(-85/2.0, 85/2.0),
                                                                      invert_z_axis=True)

            # result shape [ pgm_width, pgm_height, x,y,z, intensity, range, label]
            result = np.zeros((pgm_height, pgm_width, 6), dtype=np.float)

            for idx in range(lidar_scan_data.shape[0], 6):
                pgm_azimuth_idx = pgm_azimuth[idx]
                pgm_elevation_idx = pgm_elevation[idx]
                result[pgm_azimuth_idx, pgm_elevation_idx, :3] = lidar_scan_data[idx, :3]
                result[pgm_azimuth_idx, pgm_elevation_idx, 3] = 0  # intensity = 0
                result[pgm_azimuth_idx, pgm_elevation_idx, 4] = depth[idx]
                result[pgm_azimuth_idx, pgm_elevation_idx, 5] = convert_ground_truth(lidar_labels_data[idx])

            # save file
            outfile = str(output_dir / scan_num) + ".npy"
            np.save(outfile, result)


if __name__ == "__main__":
    main()
