import argparse
import os
import cv2
import numpy as np
import random
import math
import operator
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

CLUSTER_COUNT = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)

def parse_args():
  """
    Parse command arguments.
  """
  parser = argparse.ArgumentParser(description='validate data from starbust algo I for test (ex. chessboard test)')
  parser.add_argument('path', help='path to starburst filtered file')
  return parser.parse_args()

def main():

	fig = plt.figure()
	ax = fig.add_subplot(111)
	args = parse_args()
	star_path = args.path
	# checking is ue, et correct
	if(os.path.basename(star_path).split('-')[:-1] == "-starburst-filtered.csv"):
		raise ValueError('filename is not ending with -starburst-filtered.csv ')
	with open(star_path, 'r') as star_read_handler:
		starAllLines = star_read_handler.readlines() # should be img-name-sorted
		dict_gazePoints = {}
  		keyList = []
		star_it = iter(starAllLines)
		while(1):
	  			try:
	  				eyeID, imgName, posX, posY, area, gazeX, gazeY = next(star_it).rstrip().split(',')
	  			except StopIteration: 
	  				print("Warning: starburst-filtered list has finished")						
	  				break
	  			key = tuple([gazeX, gazeY])
	  			if key in dict_gazePoints.keys():
	  				dict_gazePoints[key].append([eyeID, imgName, float(posX), float(posY), float(area), float(gazeX), float(gazeY)])
	  			else:
	  				dict_gazePoints[key] = [[eyeID, imgName, float(posX), float(posY), float(area), float(gazeX), float(gazeY)]]
					keyList.append(key)
		random.shuffle(keyList)
		print("WF")
		counter_K = 0
		for k in range(len(keyList)):
			print("k: ",k)
			if(not(dict_gazePoints[keyList[k]][0][6]>-0.2 and dict_gazePoints[keyList[k]][0][5]<0.2 and dict_gazePoints[keyList[k]][0][5]>-0.2)): # to reduce gazeVectors with a lot of bad ellipses
	  			continue
			counter_K += 1
			currentData = dict_gazePoints[keyList[k]]
			startNumber = int(os.path.splitext(currentData[0][1])[0])
			X = []
			Y = []
			if(counter_K>4):
				break

			'''
			currentData_filtered = []
			currentData = dict_gazePoints[keyList[k]]
		  	posSamples = np.empty(shape=(len(currentData),2), dtype=np.float32)
		  	for j in range(len(currentData)):
			  	posSamples[j][0] = currentData[j][2] # cpy posX
				posSamples[j][1] = currentData[j][3] # cpy posY
  			# Set flags (Just to avoid line break in the code)
  			# Apply KMeans
  			cluster_energy = np.zeros(CLUSTER_COUNT, dtype=np.int)
  			compactness,labels,centers = cv2.kmeans(posSamples,CLUSTER_COUNT,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
			# find most popuar center
			#print("centers: ", centers)
  			for l in labels:
				cluster_energy[l] = cluster_energy[l] + 1
				#print("cluster_energy: ", cluster_energy)
	  		best_cluster_index = np.argmax(cluster_energy)
			print("len l: ", len(labels))
			print("best_cluster_index: ", best_cluster_index)
	  		if(centers[best_cluster_index][0] != 0.0 and centers[best_cluster_index][0] != 0.0):
	  			print("point is ok")
	  		else:
	  			print("point is really bad and will be ommited:!", " Point k:", k, keyList[k])
	  			print(compactness)
	  			print(centers)
	  			print(cluster_energy)
	  			print(best_cluster_index)
			for i in range(len(labels)):
				if(labels[i] == best_cluster_index):
					currentData_filtered.append(currentData[i])
			'''

			for j in range(len(currentData)):
				currNumber = int(os.path.splitext(currentData[j][1])[0]) # get base name
				timeSlice = float(currNumber - startNumber) # 300hz cam, get back time from foto Id's (loseless, but more or less ok)
				#timeSlice*= 3.000000 # 300hz cam, get back time from foto Id's (loseless, but more or less ok)
				area = currentData[j][4]
				X.append(timeSlice)
				Y.append(area)
			# 300 represents number of points to make between T.min and T.max
			xnew = np.linspace(min(X), max(X), 300)  # ???
			spl = make_interp_spline(X, Y, k=3)  # type: BSpline
			smooth = spl(xnew)
			#plt.plot(X, Y, 'b-', label=str(keyList[k]))
			ax.plot(xnew, smooth, label=str(keyList[k]))
		ax.legend(loc='best')
		ax.set_xlabel('czas')
		ax.set_ylabel('pole pow. elipsy')
		#ax.text(3, 130, '---- gazeX, gazeY', style='italic',
        	#bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
		plt.show()

if __name__ == "__main__":
  main()
