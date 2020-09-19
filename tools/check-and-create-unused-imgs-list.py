#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random

cvsListPath = sys.argv[1] if len(sys.argv) > 1 else '/media/kamil/DATA/ACOMO-DATASETS/Train6.txt'
csvDir = os.path.dirname(cvsListPath)
saveFileName = os.path.splitext(cvsListPath)[0] + "-list-to-remove.txt"

with open(saveFileName, 'w') as write_file_handler:
	with open(cvsListPath) as txtlist:
		data = []
		csvFiles = txtlist.readlines()
		to_remove_list = []
		for csv_file in csvFiles:
			csv_file = csv_file.rstrip("\n")
			folderLabel = csv_file.split('.')[0]
			realFiles = []
			# r=root, d=directories, f = files
			for r, d, f in os.walk(os.path.join(csvDir, folderLabel)):
				for file in f:
					#print("file issss: ", file)
					filename, extension = os.path.splitext(file)
					if (extension==".png" or extension==".jpg"):
						realFiles.append(file)
			csvAllLines = []
			dict_csv_files = {}
			with open(os.path.join(csvDir, csv_file), 'r') as f:
				csvAllLines = f.readlines()[0:] # skip headers here if any
			for c in csvAllLines:
				fname = c.split(',')[0]
				if fname in dict_csv_files.keys():
					print("warning, key exist: ", fname , "that means same image is listed more than once in .cvs")
				else:
					dict_csv_files[fname] = fname
			
			for r in realFiles:
				if(not(r in dict_csv_files.keys())):
					to_remove_list.append(os.path.join(csvDir, folderLabel, r, "\n"))
			print("realFiles num: ", len(realFiles))
			print("files in csv list num: ", len(csvAllLines))
			print("dict from csv list num: ", len(dict_csv_files.keys()))
			print("files to be removed so far num: ", len(to_remove_list))
	write_file_handler.writelines(to_remove_list)
	print('done')
