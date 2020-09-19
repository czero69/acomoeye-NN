#!/usr/bin/python
# -*- coding: utf-8 -*-
#Example:
#python3 merge-csv-dataset.py /path/to/list-of-csvs.txt

import os
import sys
import random
# importing shutil module  
import shutil
import time

# for radians 
import math 

directory = sys.argv[1] if len(sys.argv) > 1 else 'Q:/DATASETS/NVGAZE_SYN'
folderNameToOpen = "subjectID"
csvNameToRead = "footage_description.csv"

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     print(filename)
     if(filename.split('_')[0] == "nvgaze"):
     	subjectID= filename.split('_')[2] # get subject id
     	setID = filename.split('_')[5]
     	sex = filename.split('_')[1] # Male or Female
     	print("processing subjectID: ",sex,",",subjectID,",",setID)
     	newNameDir = sex + "_" + subjectID + "_" + setID
     	local_folderToOpen = folderNameToOpen
     	if( os.path.exists(os.path.join(directory, filename,str((int(subjectID)-1)*4+int(setID)).zfill(2)))):
     		local_folderToOpen = str((int(subjectID)-1)*4+int(setID)).zfill(2) # in case of folder named 01, 06, 10, 12 instead of "subjectID"
     	print("to open: ", local_folderToOpen)
     	source_dir_path = os.path.join(directory, filename,local_folderToOpen)
     	destination_dir_path = os.path.join(directory, newNameDir) # move .. and change name so .csv name will be unique and will identify which dataset it is easily
     	### shutil.move(source_dir_path, destination_dir_path)
     	print("moved to: ", destination_dir_path)
     	csv_read = os.path.join(directory, filename, csvNameToRead)
     	csv_save = os.path.join(directory, newNameDir+".csv") # saved one dir higher ..
     	gazePointsList = []
     	# write header first
     	for i in range(12-1):
     		gazePointsList.append("# header\n")
     	gazePointsList.append("imagefile,eye,gaze_x,gaze_y\n")
     	with open(csv_read, 'r') as read_file_handler:
     		csvAllLines = read_file_handler.readlines()[1:]
     		for i in range(len(csvAllLines)):
     			img_fname = csvAllLines[i].split(',')[1]
     			gazeX_deg = math.radians(float(csvAllLines[i].split(',')[9]))
     			gazeY_deg = math.radians(float(csvAllLines[i].split(',')[10]))
     			eyeID = csvAllLines[i].split(',')[4]
     			#print("g xy eyeID: ",str(gazeX_deg), gazeY_deg, eyeID)
     			gazePointsList.append(img_fname+","+eyeID+","+str(gazeX_deg)+","+str(gazeY_deg)+"\n")
     	with open(csv_save, 'w') as save_file_handler:
     		save_file_handler.writelines(gazePointsList)
     	print("done: ", csv_save)


     #if filename.endswith(".asm") or filename.endswith(".py"): 
         # print(os.path.join(directory, filename))
     #    continue
     #else:
     #    continue


