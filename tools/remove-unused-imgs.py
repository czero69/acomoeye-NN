#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random

cvsListPath = sys.argv[1] if len(sys.argv) > 1 else '/media/kamil/DATA/ACOMO-DATASETS/Train6-list-to-remove.txt'

with open(cvsListPath) as txtlist:
	csvFiles = txtlist.readlines()
	for csv_file in csvFiles:
		csv_file = csv_file.rstrip("\n")
		os.remove(csv_file)
print('done')
