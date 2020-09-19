#!/usr/bin/python
# -*- coding: utf-8 -*-
#Example:
#python3 merge-csv-dataset.py /path/to/list-of-csvs.txt

import os
import sys
import random
from itertools import tee
import copy

only_one_eyeID = True # if true monocular dataset, else binocular dataset will be created
selectedEyeID = 'R' # valid only if only_one_eyeID is True; R - right Eye; L - left Eye

# dataset csv lines for each subject must be sorted for gazeX then for gazeY, and data points must be in chessboard patterns
cvsListPath = sys.argv[1] if len(sys.argv) > 1 else '/media/kamil/DATA/ACOMO-DATASETS/Train6.txt'   # txt file with subjects list which will be processed
doShuffle = True if len(sys.argv) > 2 else False
csvDir = os.path.dirname(cvsListPath)

saveFileNameFrames1 = saveFileName = os.path.join(os.path.dirname(cvsListPath), "stage1_triplets", "eyes-stage1-t-frame1.csv")
saveFileNameFrames2 = saveFileName = os.path.join(os.path.dirname(cvsListPath), "stage1_triplets", "eyes-stage1-t-frame2.csv")
saveFileNameFrames3 = saveFileName = os.path.join(os.path.dirname(cvsListPath), "stage1_triplets", "eyes-stage1-t-frame3.csv")

def getNextElemIt(myit):
	try:
		prev_it = copy.copy(myit)
	except:
		print("none passed to getNextElemIt")
		return
	try:
		current_X = next(myit).split(',')[2]
		current_X_ = current_X
		while(current_X == current_X_):
			prev_it = copy.copy(myit)
			current_X_ = next(myit).split(',')[2]
	except StopIteration:
		myit = (prev_it)
		return myit
		
	myit = (prev_it)
	return myit

# get next keypoint from the same row
def getNextRow(myit):
	if (myit==None):
		return None, False
	try:
		prev_it = copy.copy(myit)
	except:
		print("none row passed")
		return None, False
	try:
		current_X, current_Y = next(myit).split(',')[2:4]
		current_X_ = current_X
		current_Y_ = current_Y
		while(current_X == current_X_):
			prev_it = copy.copy(myit)
			current_X_, current_Y_ = next(myit).split(',')[2:4]
		if(current_Y_ != current_Y): # it is not in the same row
			#print('hit END!')
			myit = (prev_it)
			return prev_it, False # end of a row
	except StopIteration:	    # end of all datapoints
		myit = (prev_it)
		return prev_it, False
		
	myit = (prev_it)
	return prev_it, True

def getNextCol(myit):
	if (myit==None):
		return None, False
	try:
		prev_it = copy.copy(myit)
	except:
		print("none col passed")
		return None, False
	try:
		current_X, current_Y = next(myit).split(',')[2:4]
		current_X_ = current_X
		current_Y_ = current_Y
		while(current_Y == current_Y_):
			prev_it = copy.copy(myit)
			current_X_, current_Y_ = next(myit).split(',')[2:4]
		while(current_X_ != current_X):
			prev_it = copy.copy(myit)
			current_X_, current_Y_ = next(myit).split(',')[2:4]
	except StopIteration: 	# end of a col will be end of datapoints anyway
		myit = (prev_it)
		return prev_it, False
	myit = (prev_it)
	return prev_it, True

def getNextDiag(myit):
	if (myit==None):
		return None, False
	try:
		prev_it = copy.copy(myit)
	except:
		print("none diag passed")
		return None, False
	try:
		current_X, current_Y = next(myit).split(',')[2:4]
		current_Y_ = current_Y
		current_X_ = current_X
		while(current_Y == current_Y_):
			prev_it = copy.copy(myit)
			current_X_, current_Y_ = next(myit).split(',')[2:4]
		current_Y = current_Y_
		while(current_X != current_X_):
			prev_it = copy.copy(myit)
			current_X_, current_Y_ = next(myit).split(',')[2:4]
		while(current_X == current_X_):
			prev_it = copy.copy(myit)
			current_X_, current_Y_ = next(myit).split(',')[2:4]
		if(current_Y_ != current_Y): # it is not in the same row
			#print("diag not exist!")
			myit = (prev_it)
			return prev_it, False # end of a row	
	except StopIteration:
		myit = (prev_it)
		return prev_it, False
	myit = (prev_it)
	return prev_it, True

def getXCoord(elem):
	return elem[2]

def getNextTriplets(csvAllLines, skip=5):
	triplets_up_row = []
	triplets_right_col = []
	triplets_diag_l2r_u2d = []
	triplets_diag_l2r_d2u = []
	#triplets = []

	prev_it = iter(csvAllLines)
	#it_row = iter(csvAllLines)
	#it_col = iter(csvAllLines)
	#it_diag = iter(csvAllLines)
	#it_nn_row = iter(csvAllLines) #next,next row
	#it_nn_col = iter(csvAllLines) #next,next col
	#it_nn_diag = iter(csvAllLines) #next,next diag

	#print('12722')
	first_time = 0

	try:

		while(1):
			
			myit = copy.copy(prev_it)
			it_row = copy.copy(myit)
			it_col = copy.copy(myit)
			it_diag = copy.copy(myit)
			it_nn_row = copy.copy(myit) #next,next row
			it_nn_col = copy.copy(myit) #next,next col
			it_nn_diag = copy.copy(myit) #next,next diag

			#print('XX')
			it_row, w1 = getNextRow(it_row)
			it_col, w2 = getNextCol(it_col)
			it_diag, w3 = getNextDiag(it_diag)
			it_nn_row, w4 = getNextRow(it_nn_row)
			it_nn_row, w5 = getNextRow(it_nn_row)
			it_nn_col, w7 = getNextCol(it_nn_col)
			it_nn_col, w8 = getNextCol(it_nn_col)
			it_nn_diag, w9 = getNextDiag(it_nn_diag)
			it_nn_diag, w0 = getNextDiag(it_nn_diag)

			if(not(w1 and w2 and w3 and w4 and w5 and w7 and w8 and w9 and w0)):
				elem = next(myit)
				prev_it = getNextElemIt(prev_it)
				myit = copy.copy(prev_it)
				continue
			
			elem = next(myit)
			elem_row = next(it_row)
			elem_col = next(it_col)
			elem_diag = next(it_diag)
			elem_nn_row = next(it_nn_row)
			elem_nn_col = next(it_nn_col)
			elem_nn_diag = next(it_nn_diag)
			
			if(not first_time):
				first_time = 1
				#print("elem_row: ", elem_row)
			# if X changed it means we are on the next keypoint so we need to skip it
			first_X = getXCoord(elem)
			first_X_row = getXCoord(elem_row)
			first_X_col = getXCoord(elem_col)
			first_X_diag = getXCoord(elem_diag)
			first_X_nn_row = getXCoord(elem_nn_row)
			first_X_nn_col = getXCoord(elem_nn_col)
			first_X_nn_diag = getXCoord(elem_nn_diag)

			counter = 0
			# unrolled first sample
			if( counter % skip == 0):
					triplets_up_row.append([elem, elem_row, elem_nn_row])
					triplets_right_col.append([elem, elem_col, elem_nn_col])
					triplets_diag_l2r_d2u.append([elem, elem_diag, elem_nn_diag])
					triplets_diag_l2r_u2d.append([elem_nn_row, elem_diag, elem_nn_col])
					#triplets.append([elem, elem_diag, elem_nn_diag])

			while( first_X == getXCoord(elem) and first_X_row == getXCoord(elem_row) and first_X_col == getXCoord(elem_col) and first_X_diag == getXCoord(elem_diag) and first_X_nn_row == getXCoord(elem_nn_row) and first_X_nn_col == getXCoord(elem_nn_col) and first_X_nn_diag == getXCoord(elem_nn_diag) ):

				counter = counter + 1
				if( counter % skip == 0):
					triplets_up_row.append([elem, elem_row, elem_nn_row])
					triplets_right_col.append([elem, elem_col, elem_nn_col])
					triplets_diag_l2r_d2u.append([elem, elem_diag, elem_nn_diag])
					triplets_diag_l2r_u2d.append([elem_nn_row, elem_diag, elem_nn_col]) 
					#triplets.append([elem, elem_diag, elem_nn_diag])
				elem = next(myit)
				elem_row = next(it_row)
				elem_col = next(it_col)
				elem_diag = next(it_diag)
				elem_nn_row = next(it_nn_row)
				elem_nn_col = next(it_nn_col)
				elem_nn_diag = next(it_nn_diag)

			prev_it = getNextElemIt(prev_it)
			myit = copy.copy(prev_it)
						
	
	except StopIteration:
		print("that would be out of bound, time to return triplets")
		return triplets_up_row, triplets_right_col, triplets_diag_l2r_d2u, triplets_diag_l2r_u2d		
		
	return triplets_up_row, triplets_right_col, triplets_diag_l2r_d2u, triplets_diag_l2r_u2d
	
with open(saveFileNameFrames1, 'w') as write_file_handler_frames1:
  with open(saveFileNameFrames2, 'w') as write_file_handler_frames2:
    with open(saveFileNameFrames3, 'w') as write_file_handler_frames3:
      with open(cvsListPath) as txtlist:
      	csvFiles = txtlist.readlines()
      	for csv_file in csvFiles:
      		csv_file = csv_file.rstrip("\n")
      		with open(os.path.join(csvDir, csv_file), 'r') as f:
		      	csvAllLines = f.readlines()[0:] # skip headers
      			csv_file_no_ext = os.path.splitext(csv_file)[0]
		      	triplets = []
					
		      	triplets_up_row, triplets_right_col, triplets_diag_l2r_d2u, triplets_diag_l2r_u2d = getNextTriplets(csvAllLines)

		      	#print("triplets[0]: ", triplets_up_row[0])
		      	#print("triplet1[-1]: ", triplets_up_row[-1])

      			#print("triplets size: ", len(triplets_up_row))
      			#print(triplets_up_row)

      			for i in range(len(triplets_up_row)):
      				write_file_handler_frames1.write(csv_file_no_ext+"/"+triplets_up_row[i][0])
      				write_file_handler_frames2.write(csv_file_no_ext+"/"+triplets_up_row[i][1])
	      			write_file_handler_frames3.write(csv_file_no_ext+"/"+triplets_up_row[i][2])
		      		write_file_handler_frames1.write(csv_file_no_ext+"/"+triplets_right_col[i][0])
      				write_file_handler_frames2.write(csv_file_no_ext+"/"+triplets_right_col[i][1])
      				write_file_handler_frames3.write(csv_file_no_ext+"/"+triplets_right_col[i][2])
      				write_file_handler_frames1.write(csv_file_no_ext+"/"+triplets_diag_l2r_d2u[i][0])
      				write_file_handler_frames2.write(csv_file_no_ext+"/"+triplets_diag_l2r_d2u[i][1])
      				write_file_handler_frames3.write(csv_file_no_ext+"/"+triplets_diag_l2r_d2u[i][2])
      				write_file_handler_frames1.write(csv_file_no_ext+"/"+triplets_diag_l2r_u2d[i][0])
      				write_file_handler_frames2.write(csv_file_no_ext+"/"+triplets_diag_l2r_u2d[i][1])
      				write_file_handler_frames3.write(csv_file_no_ext+"/"+triplets_diag_l2r_u2d[i][2])


print('done')
