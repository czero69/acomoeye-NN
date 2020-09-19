#!/usr/bin/python
# -*- coding: utf-8 -*-
#Example:
#python3 merge-csv-dataset.py /path/to/list-of-csvs.txt --shuffle

# split the dataset into train/test, dividing each individual gaze points into separate gaze point train/test subsets
# this means that same subject will be present in both train and test but different gaze points will be present
# for dividing dataset which do not use same subject for train and test, create manually csv list and use merge-csv-dataset.py

import os
import sys
import random

only_one_eyeID = True # if true monocular dataset, else binocular dataset will be created
selectedEyeID = 'R'   # valid only if only_one_eyeID is True; R - right Eye; L - left Eye
div_by_2 = False      # in case of nvgaze real dataset you may want to divide angels by 2, so it matches 'standard projection definition' used in nvgaze synthetic.
trainTestRatio = 0.8  # for each subject, for particular subject (ex. 01) divide all gaze points into train and test subsets 

cvsListPath = sys.argv[1] if len(sys.argv) > 1 else '/media/kamil/DATA/ACOMO-DATASETS/Train6.txt'
doShuffle = True if len(sys.argv) > 2 else False
csvDir = os.path.dirname(cvsListPath)

saveFileName = saveFileName = os.path.splitext(cvsListPath)[0] + "-merged"
if(only_one_eyeID):
	saveFileName += "-monocular-"+selectedEyeID
if(doShuffle):
	saveFileName += "-shuffled"

saveFileName_train = saveFileName
saveFileName_test = saveFileName

saveFileName_train += "-train.csv"
saveFileName_test += "-test.csv"

def divide_gaze_vals(line):
	splitted = line.split(','); splitted[2] = gazeX; splitted[3] = gazeY; return ','.join(splitted)+'\n';

with open(saveFileName_train, 'w') as write_file_handler_train:
	with open(saveFileName_test, 'w') as write_file_handler_test:
		with open(cvsListPath) as txtlist:
			csvFiles = txtlist.readlines()
			data_train = []
			data_test = []
			for csv_file in csvFiles:
				csv_file = csv_file.rstrip("\n")
				folderLabel = csv_file.split('.')[0]		
				with open(os.path.join(csvDir, csv_file), 'r') as f:
					dict_gazePoints = {}
					keyList = []
					csvAllLines = f.readlines()[12:]
					for i in range(len(csvAllLines)):
						doProcess = False
						if(only_one_eyeID):
							if(csvAllLines[i].split(',')[1] == selectedEyeID):
								doProcess = True
						else:
							doProcess = True
						
						if(doProcess):
							# get gazeX, gazeY
							csvAllLines[i] = folderLabel + '/' + csvAllLines[i]
							gazeX = csvAllLines[i].split(',')[2]
							gazeY = csvAllLines[i].split(',')[3]
							if(div_by_2):
								gazeX, gazeY = str(float(gazeX)/2), str(float(gazeY)/2)
								csvAllLines[i] = divide_gaze_vals(csvAllLines[i])
							key = tuple([gazeX, gazeY])
							#print(key)
							if key in dict_gazePoints.keys():
								dict_gazePoints[key].append(csvAllLines[i])
							else:
								dict_gazePoints[key] = [csvAllLines[i]]	
								keyList.append(key)
					# shuffle keyList per subject, randomly choose which gazePoints for target subject will be in train/test
					random.shuffle(keyList) 
					for k in range(len(keyList)):
						if (k < len(keyList)*trainTestRatio):
							for l in dict_gazePoints[keyList[k]]:
								data_train.append(l)
						else:
							for l in dict_gazePoints[keyList[k]]:
								data_test.append(l)
			
			if(doShuffle):
				random.shuffle(data_train) # shuffle only train and not test
			write_file_handler_train.writelines(data_train)
			write_file_handler_test.writelines(data_test)
print('... done!')
