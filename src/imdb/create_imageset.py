import argparse
import shutil
from tqdm import tqdm

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport


def write_to_file(dir_path, file_name, content):
    """
    write content to file
    """
    file_path = str(dir_path / file_name)
    with open(file_path, "w") as f:
        for line in content:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description='create image set oxford')

    # parameters
    parser.add_argument('--dataset_files', default='', help='path to dataset npy files')
    parser.add_argument('--train_chunks', help='training chunks separated by comma')
    parser.add_argument('--val_chunks', help='validation chunks separated by comma')
    parser.add_argument('--output_dir', default='data', help='path to output dir')

    args = parser.parse_args()

    root_dataset_dir = Path(args.dataset_files)
    output_dir = Path(args.output_dir)

    if not root_dataset_dir.exists():
        print("dataset dir does not exit")
        return None

    if not output_dir.exists():
        print("create output dir")
        output_dir.mkdir(parents=True, exist_ok=True)

    lidar_2d_path = output_dir / "lidar_2d"
    imageset_path = output_dir / "ImageSet"
    file_all_content = []
    file_train_content = []
    file_val_content = []

    if not lidar_2d_path.exists():
        print("create lidar_2d dir")
        lidar_2d_path.mkdir(parents=True, exist_ok=True)

    if not imageset_path.exists():
        print("create imageset dir")
        imageset_path.mkdir(parents=True, exist_ok=True)

    ##### training

    training_chunks = args.train_chunks.split(',')
    training_chunks = [root_dataset_dir / Path(train_chunk) for train_chunk in training_chunks]

    for train_chunk in training_chunks:
        if not train_chunk.exists():
            print("{} does not exist".format(str(train_chunk)))
            continue
        print(str(train_chunk))
        # find lidar scans
        lidar_scans = train_chunk.glob("*.npy")
        for scan in tqdm(lidar_scans):
            file_all_content.append(scan.stem)
            file_train_content.append(scan.stem)
            shutil.copy(str(scan), str(lidar_2d_path))

    #  validation
    val_chunks = args.val_chunks.split(',')
    val_chunks = [root_dataset_dir / Path(val_chunk) for val_chunk in val_chunks]
    for val_chunk in val_chunks:
        if not val_chunk.exists():
            print("{} does not exist".format(str(val_chunk)))
        print(str(val_chunk))
        lidar_scans = val_chunk.glob("*.npy")
        for scan in tqdm(lidar_scans):
            file_all_content.append(scan.stem)
            file_val_content.append(scan.stem)
            shutil.copy(str(scan), str(lidar_2d_path))

    #  write Image set files
    write_to_file(imageset_path, "all.txt", file_all_content)
    write_to_file(imageset_path, "train.txt", file_train_content)
    write_to_file(imageset_path, "val.txt", file_val_content)


if __name__ == "__main__":
    main()
