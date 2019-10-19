
export BASE_DIR="/home/fusionresearch/Rafaat/oxford-annotation/oxford-automation/results"
export PREPARE_OXFORD="src/imdb/prepare_oxford.py"
export CREATE_IMAGESET="src/imdb/create_imageset.py"
export OUTPUT_DATA_DIR="data/data"

export PGM_WIDTH="512" # sick HFoV: 85, angular resolution: 0.125, 85/0.125 = 360
export PGM_HEIGHT="64"

 # chunk 1
 echo "chunk 1"
 export CHUNK_NUM="2014-06-24-14-20-41"
 export INCLUDE_INDEX="0"
 export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
 export OUTPUT_DIR="${OUTPUT_DATA_DIR}/${CHUNK_NUM}"

 python "${PREPARE_OXFORD}" --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}" --pgm_height "${PGM_HEIGHT}" --pgm_width "${PGM_WIDTH}"

 # chunk 2

 echo "chunk 2"
 export CHUNK_NUM="2014-05-14-13-59-05"
 export INCLUDE_INDEX="0"
 export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
 export OUTPUT_DIR="${OUTPUT_DATA_DIR}/${CHUNK_NUM}"

 python "${PREPARE_OXFORD}" --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}" --pgm_height "${PGM_HEIGHT}" --pgm_width "${PGM_WIDTH}"


# chunk 3

echo "chunk 3"
export CHUNK_NUM="2014-05-06-13-14-58"
export INCLUDE_INDEX="0"
export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
export OUTPUT_DIR="${OUTPUT_DATA_DIR}/${CHUNK_NUM}"

python "${PREPARE_OXFORD}" --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}" --pgm_height "${PGM_HEIGHT}" --pgm_width "${PGM_WIDTH}"


# # chunk 4

# # echo "chunk 4"
# # export CHUNK_NUM="2014-06-25-16-22-15"
# # export INCLUDE_INDEX="100"
# # export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
# # export OUTPUT_DIR="data/${CHUNK_NUM}"

# # python prepare_oxford.py  --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}"

 python "${CREATE_IMAGESET}" --dataset_files "${OUTPUT_DATA_DIR}" --output_dir "data" --train_chunks "2014-06-24-14-20-41,2014-05-06-13-14-58" --val_chunks "2014-05-14-13-59-05"
# "data/lidar_2d"
python src/imdb/calculate_stats.py --lidar_files "${OUTPUT_DIR}" --pgm_height "${PGM_HEIGHT}" --pgm_width "${PGM_WIDTH}" --channels "6"