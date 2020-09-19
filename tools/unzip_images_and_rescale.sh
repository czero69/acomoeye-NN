#!/bin/bash

# this script is written to be memory-in-place efficient
# it takes one 50GB zip at a time, copy it form external HDD to local drive, 
# then it unpacks only eye-images.png without other metadatas like segmentation masks 
# (this is relatively smaller subset than full zip archive, like 60GB/350GB). 
# Then It downscales full-size images 1280x1000 to smaller resolution, 127x127
# Then it deletes downloaded zip and download next zip in a loop

PATH_TO_ZIPS_DIR="/d/ET-datasets/syntetic_dataset" # Path with nvgaze syntetic zips in one dir: ex: {nvgaze_female_02_public_50K_4, ..}
PATH_DESTINATION="/q/DATASETS/NVGAZE_SYN"

echo $PATH_TO_ZIPS_DIR " and.." $PATH_DESTINATION

SUBJECTS_DIR=('nvgaze_male_01_public_50K_1' 
			  'nvgaze_male_01_public_50K_2'
			  'nvgaze_male_01_public_50K_3'
			  'nvgaze_male_01_public_50K_4'
			  'nvgaze_male_02_public_50K_1' 
			  'nvgaze_male_02_public_50K_2'
			  'nvgaze_male_02_public_50K_3'
			  'nvgaze_male_02_public_50K_4'
			  'nvgaze_male_03_public_50K_1' 
			  'nvgaze_male_03_public_50K_2'
			  'nvgaze_male_03_public_50K_3'
			  'nvgaze_male_03_public_50K_4'
			  'nvgaze_male_04_public_50K_1' 
			  'nvgaze_male_04_public_50K_2'
			  'nvgaze_male_04_public_50K_3'
			  'nvgaze_male_04_public_50K_4'
			  'nvgaze_male_05_public_50K_1' 
			  'nvgaze_male_05_public_50K_2'
			  'nvgaze_male_05_public_50K_3'
			  'nvgaze_male_05_public_50K_4'
			  )
for subject in "${SUBJECTS_DIR[@]}"
do
	ZIP_DIR="${PATH_TO_ZIPS_DIR}/${subject}"
	SUBJECT_DIR="${PATH_DESTINATION}/${subject}"
	echo "copying: ${ZIP_DIR}"
	cp -R $ZIP_DIR $PATH_DESTINATION
	echo "extracting ..."
	#find . -name "*.zip" -type f -exec unzip -jd "images/{}" "{}" "*.jpg" "*.png" "*.gif" \;
	unzip -q -j "${SUBJECT_DIR}/footage_image_data.zip" 'type_img_frame_*.png' -d "${SUBJECT_DIR}/subjectID"
	wait # here wait before we finish prev resizing
	echo "resizing ..."
	#eval echo "python ./downscale_images.py ${SUBJECT_DIR}/subjectID -target_size 127"
	eval python "./downscale_images.py ${SUBJECT_DIR}/subjectID -target_size 127" & # start copying in parallel
	echo "removing *.zip"
	rm "${SUBJECT_DIR}/footage_image_data.zip"
done
wait
echo "...all Done!"
