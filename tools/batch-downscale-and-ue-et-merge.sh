#!/bin/bash

PATH_TO_DATASETS_DIR="/media/kamil/DATA/ACOMO-DATASETS"

# subjects base names (without _ir or _bgr)
SUBJECTS_LIST=('kamil-17-light'
			'kamil-18-light'
			'wojtus-3-light'
			'gabrys-4-light'
			)

for subject in "${SUBJECTS_LIST[@]}"
do
	BASE_PATH="${PATH_TO_DATASETS_DIR}/${subject}"
	IR_DIR_PATH="${BASE_PATH}_ir"
	IR_DOWNSCALED_DIR_PATH="${BASE_PATH}_ir-downscaled"
	UE_FILE_PATH="${BASE_PATH}-ue.csv"	
	eval python "./downscale_images_cut_padding.py ${IR_DIR_PATH}"
	eval python "./ue-et-time-data.py ${UE_FILE_PATH}"
	mv ${IR_DOWNSCALED_DIR_PATH} ${BASE_PATH} &

done
wait
echo ".. downscaled ir, merged eu+et datas, deleted non-used-downscaled images !"
