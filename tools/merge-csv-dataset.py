#!/usr/bin/python
# -*- coding: utf-8 -*-
#Example:
#python3 merge-csv-dataset.py /path/to/list-of-csvs.txt

import os
import sys
import random

only_one_eyeID = True # if true monocular dataset, else binocular dataset will be created
selectedEyeID = 'R' # valid only if only_one_eyeID is True; R - right Eye; L - left Eye
div_by_2 = False      # in case of nvgaze real dataset you may want to divide angels by 2, so it matches 'standard projection definition' used in nvgaze synthetic.

cvsListPath = sys.argv[1] if len(sys.argv) > 1 else '/media/kamil/DATA/ACOMO-DATASETS/Train6.txt'
doShuffle = True if len(sys.argv) > 2 else False
csvDir = os.path.dirname(cvsListPath)

saveFileName = saveFileName = os.path.splitext(cvsListPath)[0] + "-merged"
if(only_one_eyeID):
	saveFileName += "-monocular-"+selectedEyeID
if(doShuffle):
	saveFileName += "-shuffled"
saveFileName += ".csv"

def divide_gaze_vals(line):
	splitted = line.split(','); splitted[2] = gazeX; splitted[3] = gazeY; return ','.join(splitted)+'\n';

with open(saveFileName, 'w') as write_file_handler:
	with open(cvsListPath) as txtlist:
		data = []
		csvFiles = txtlist.readlines()
		for csv_file in csvFiles:
			csv_file = csv_file.rstrip("\n")
			folderLabel = csv_file.split('.')[0]		
			with open(os.path.join(csvDir, csv_file), 'r') as f:
				csvAllLines = f.readlines()[12:] # skip headers
				csvAllLinesPerEyeID = []
				for i in range(len(csvAllLines)):
					gazeX = csvAllLines[i].split(',')[2]
					gazeY = csvAllLines[i].split(',')[3]
					if(div_by_2):
						gazeX, gazeY = str(float(gazeX)/2), str(float(gazeY)/2)
						csvAllLines[i] = divide_gaze_vals(csvAllLines[i])
					if( csvAllLines[i].split(',')[1] == selectedEyeID):
						csvAllLinesPerEyeID.append(folderLabel + '/' + csvAllLines[i])
					csvAllLines[i] = folderLabel + '/' + csvAllLines[i]
				if(only_one_eyeID):
					data.extend(csvAllLinesPerEyeID)
				else:
					data.extend(csvAllLines)
		if(doShuffle):
			random.shuffle(data)
		write_file_handler.writelines(data)

print('done')
