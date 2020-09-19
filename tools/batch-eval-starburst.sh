#!/bin/bash

PATH_TO_DATASETS_DIR="/media/kamil/DATA/NVGAZE-DATASETS/"
ACOMOEYE_ET_BIN="/home/kamil/dependencies/econ-grabFrames/e-con_Modified_OpenCV/build/acomoeye-et"

# subjects base names (without _ir or _bgr)
SUBJECTS_LIST=('15'
			'16'
			'17'
			'18'
			'19'
			'20'
			'21'
			'22'
			'23'
			'24'
			'25'
			'26'
			'27'
			'28'
			'29'
			'30'
			'31'
			'32'
			'33'
			'34'
			'35'
			'36'
			'37'
			'38'
			'39'
			'40'
			'41'
			'42'
			)

for subject in "${SUBJECTS_LIST[@]}"
do
	BASE_PATH="${PATH_TO_DATASETS_DIR}/${subject}"
	STAR_PATH="${BASE_PATH}-starburst.csv"
	IR_PATH="${BASE_PATH}_ir-starburst.csv"
	FILTERED_PATH="${BASE_PATH}-starburst-filtered"
	UE_FILE_PATH="${BASE_PATH}-ue.csv"	
	eval ${ACOMOEYE_ET_BIN} "${BASE_PATH}/ -o -d -e" 
	mv ${STAR_PATH} ${IR_PATH}
	eval python "./rip-starburst-to-keypoints-data.py ${IR_PATH} -eyeID L"
	eval python "./rip-starburst-to-keypoints-data.py ${IR_PATH} -eyeID R"
	eval python "./validate-starburst.py ${FILTERED_PATH}-L.csv"
	eval python "./validate-starburst.py ${FILTERED_PATH}-R.csv" &

done
wait
echo ".. downscaled ir, merged eu+et datas, deleted non-used-downscaled images !"
