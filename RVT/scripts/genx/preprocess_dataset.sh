NUM_PROCESSES=20  # set to the number of parallel processes to use
DATA_DIR=/data/scratch1/nzubic/datasets/gen1_tar/
DEST_DIR=/data/scratch1/nzubic/datasets/RVT/gen1_frequencies/gen1_200hz/
FREQUENCY=conf_preprocess/extraction/frequencies/const_duration_200hz.yaml

python preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml ${FREQUENCY} \
conf_preprocess/filter_gen1.yaml -ds gen1 -np ${NUM_PROCESSES}
