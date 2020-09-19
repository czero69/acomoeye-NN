import argparse
import os
import cv2
import numpy as np
import random
import math
import operator

# this is per-subject test for haar+starburst+ellipseFitting against gazeX,gazeY labels with polynomial matching.

metrics = "NVGaze" # "NVGaze" or "Acomo"

trainTestRatio = 0.8 # for particular subject divide all gaze points into train and test subsets
CLUSTER_COUNT = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)

repeat_tests = 100

reject_if_less_than_minY = True
minY = -0.3	# angle below which test samples will be rejected if 'reject_if_less_than_minY' was set to True

# leas square method to minimize 2D polynomial fit for gazeX and gazeY accordingly: gazeX = polyfit1(x,y); gazeY = polyfit2(x,y)

def create_poly(X, Y):
	return np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
	#return np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y, X**3, Y**3, X**3*Y**3, X**3*Y, X**3*Y*Y**2, Y**3*X, Y**3*X**2]).T

def parse_args():
  """
    Parse command arguments.
  """
  parser = argparse.ArgumentParser(description='validate data from starbust algo I for test (ex. chessboard test)')
  parser.add_argument('path', help='path to starburst filtered file')
  return parser.parse_args()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def yaw_matrix(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                     [math.sin(yaw), math.cos(yaw), 0.0],
                     [0.0, 0.0, 1.0]])

def pitch_matrix(pitch):
    return np.array([[math.cos(pitch), 0.0, math.sin(pitch)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(pitch), 0.0, math.cos(pitch)]])

def roll_matrix(roll):
    return np.array([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

# for measuring nvidia nvgaze, gaze angle as projection to perpendicular planes
def accuracy_projection(curr_val, curr_label):
  label_len = len(curr_label)

  err = 0
  num = 0

  for i in range(label_len): # iterate batches

      # gaze angles are horizontal and vertical angles as 'one-sided visual angle'
      gaze_val = [math.tan(curr_val[i][0]), math.tan(curr_val[i][1]), 2]
      gaze_label = [math.tan(curr_label[i][0]), math.tan(curr_label[i][1]), 2]

      # angle between val and label (angular error in radians)
      e = math.atan2(np.linalg.norm(np.cross(gaze_label,gaze_val)),np.dot(gaze_label,gaze_val));
      e_degrees = math.degrees(e)
      err += e_degrees
      num += 1
  return float(err), num

# for measuring acomo gaze angle as ordered yaw, pitch rotation
def accuracy_yaw_pitch(curr_val, curr_label):
    label_len = len(curr_label)

    err = 0
    num = 0
    AXIS_Z = [0,0,1] # for x-horizontal-rotations
    AXIS_X = [1,0,0] # for y-vertical-rotations
    AXIS_Y = [0,1,0] # for y-vertical-rotations
    gazeX_init_val = [1,0,0]
    gazeY_init_val = [0,1,0]
    #print(curr_label)
    #print(curr_val)
    for i in range(label_len): # iterate batches

        ### -- this is alternative way to calculate rotations : --
        #val_x_around_z = np.dot(rotation_matrix(AXIS_Z, curr_val[i][0]), gazeX_init_val)
        #val_y_around_z = np.dot(rotation_matrix(AXIS_Z, curr_val[i][0]), gazeY_init_val)
        #gaze_val = np.dot(rotation_matrix(val_y_around_z, -1*curr_val[i][1]), val_x_around_z)

        #lab_x_around_z = np.dot(rotation_matrix(AXIS_Z, curr_label[i][0]), gazeX_init_val)
        #lab_y_around_z = np.dot(rotation_matrix(AXIS_Z, curr_label[i][0]), gazeY_init_val)
        #gaze_label = np.dot(rotation_matrix(lab_y_around_z, -1*curr_label[i][1]), lab_x_around_z)
        ### -- alternative way to calculate rotations --

        ###
        R_val = np.dot(yaw_matrix(curr_val[i][0]) , pitch_matrix(curr_val[i][1]))
        R_lab = np.dot(yaw_matrix(curr_label[i][0]) , pitch_matrix(curr_label[i][1]))

        gaze_val = np.dot(R_val, gazeX_init_val)
        gaze_label = np.dot(R_lab, gazeX_init_val)
        ###
        # angle between val and label (angular error in radians)
        e = math.atan2(np.linalg.norm(np.cross(gaze_label,gaze_val)),np.dot(gaze_label,gaze_val));
        e_degrees = math.degrees(e)
        err += e_degrees
        num += 1
    return float(err), num

def main():
  if(metrics == "NVGaze"):
  	accuracy = accuracy_projection
  else:
  	accuracy = accuracy_yaw_pitch
  global reject_if_less_than_minY
  args = parse_args()
  star_path = args.path
  # checking is ue, et correct
  if(os.path.basename(star_path).split('-')[:-1] == "-starburst-filtered.csv"):
  	raise ValueError('filename is not ending with -starburst-filtered.csv ')
  base_path = os.path.join(os.path.dirname(star_path),os.path.basename(star_path)[:-4]) # remove .csv
  save_results_path = base_path + '-eval-results.txt'

  print("star_path, base_path, save_results_path :", star_path, base_path, save_results_path)
  if(reject_if_less_than_minY):
  			print("WARNING: reject_if_less_than_minY is set to true, angles less than minY was rejected;")

  # for now chessBoard support only
  with open(save_results_path, 'a') as write_file_handler:
  	with open(star_path, 'r') as star_read_handler:
  		starAllLines = star_read_handler.readlines() # should be img-name-sorted
  		dict_gazePoints = {}
  		keyList = []
  		star_it = iter(starAllLines)
  		imgName_star = ""
  		eyeID_star = ""
  		posX_star, posY_star, area_star = "","",""
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
  		# randomize test/train keypoints
  		for testSwap in range(2):
  			reject_if_less_than_minY = operator.not_(reject_if_less_than_minY)
			global_cumm_err = 0
			global_cumm_num = 0
			global_cumm_best_centroids_err = 0
			global_cumm_best_centroids_num = 0
	  		for testID in range(repeat_tests):
		  		random.shuffle(keyList)
		  		calibration_points = [] 
		  		test_points_only_from_best_clusters = []
		  		for k in range(len(keyList)):
		  			if (k < len(keyList)*trainTestRatio): # TRAIN DATA
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
		  				if(centers[best_cluster_index][0] != 0.0 and centers[best_cluster_index][0] != 0.0):
		  					calibration_points.append( [centers[best_cluster_index], [currentData[0][5], currentData[0][6]]])
		  				else:
		  					print("point is really bad and will be ommited:!", " Point k:", k, keyList[k])
		  					print(compactness)
		  					print(centers)
		  					print(cluster_energy)
		  					print(best_cluster_index)
  						#print("calibration_points: ", calibration_points)
		  			else:					# TEST DATA
		  				currentData = dict_gazePoints[keyList[k]]
		  				posSamples = np.empty(shape=(len(currentData),2), dtype=np.float32)
		  				for j in range(len(currentData)):
		  					posSamples[j][0] = currentData[j][2] # cpy posX
		  					posSamples[j][1] = currentData[j][3] # cpy posY
		  				# Set flags (Just to avoid line break in the code)
		  				# Apply KMeans
		  				cluster_energy = np.zeros(CLUSTER_COUNT, dtype=np.int)
		  				compactness,labels,centers = cv2.kmeans(posSamples,CLUSTER_COUNT,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
						# find most popular center
		  				for l in labels:
		  					cluster_energy[l] = cluster_energy[l] + 1
		  				best_cluster_index = np.argmax(cluster_energy)
		  				if(centers[best_cluster_index][0] != 0.0 and centers[best_cluster_index][0] != 0.0):
		  					test_points_only_from_best_clusters.append( [centers[best_cluster_index], [currentData[0][5], currentData[0][6]]])
		  				else:
		  					print("point is really bad and will be ommited:!", " Point k:", k, keyList[k])
		  					print(compactness)
		  					print(centers)
		  					print(cluster_energy)
		  					print(best_cluster_index)
		  			#write_file_handler.writelines(outputLines)
		  		#print(calibration_points)
		  		#print(len(calibration_points))
		  		z_x_out = np.empty(shape=(len(calibration_points),1), dtype=np.float32)
		  		z_y_out = np.empty(shape=(len(calibration_points),1), dtype=np.float32)
		  		XPOS = np.empty(shape=(len(calibration_points),1), dtype=np.float32)
		  		YPOS = np.empty(shape=(len(calibration_points),1), dtype=np.float32)
		  		for c in range(len(calibration_points)):
		  			XPOS[c] = calibration_points[c][0][0]
		  			YPOS[c] = calibration_points[c][0][1]
		  			z_x_out[c] = calibration_points[c][1][0]
		  			z_y_out[c] = calibration_points[c][1][1]
		  		#print(xy_pos)
		  		#print(z_x_out)
		  		#print(z_y_out)
		  		poly = create_poly(XPOS.flatten(), YPOS.flatten())
		  		calib_vec_x, _, _, _ = np.linalg.lstsq(poly, z_x_out.flatten())
		  		calib_vec_y, _, _, _ = np.linalg.lstsq(poly, z_y_out.flatten())
		  		#print(calib_vec_x)
		  		#print(calib_vec_y)
		  		#print("calibration complete!")
				
				## make tests for only clustered test-points ##
		  		cumm_err = 0
		  		cumm_num = 0
		  		for t in test_points_only_from_best_clusters:
		  			xpos_test = t[0][0]
		  			ypos_test = t[0][1]
		  			label_x = t[1][0]
		  			label_y = t[1][1]
					if(reject_if_less_than_minY and label_y < minY):
							continue
		  			poly_test = create_poly(xpos_test.flatten(), ypos_test.flatten())
		  			#print("poly test:", poly_test)
		  			inferred = [[np.dot(poly_test,calib_vec_x)[0], np.dot(poly_test,calib_vec_y)[0]]]
		  			label_test = [[label_x, label_y]]
		  			#print("inferred: ", inferred)
		  			#print("label_test: ", label_test)
		  			err, num = accuracy(inferred,label_test)
		  			############ print("acc for best-cluster test: ", t[1], "acc: ", err)
		  			cumm_err += err
		  			cumm_num += num
		  		print("accuracy for best test clusters, step: ", cumm_err/cumm_num, testID)

		  		global_cumm_best_centroids_err += cumm_err
		  		global_cumm_best_centroids_num += cumm_num

		  		## make tests for all test-points ##
		  		cumm_err = 0
		  		cumm_num = 0
		  		for k in range(len(keyList)):
		  			if (k >= len(keyList)*trainTestRatio):	# TEST DATA
		  				currentData = dict_gazePoints[keyList[k]]
		  				posSamples = np.empty(shape=(len(currentData),2), dtype=np.float32)
		  				point_cumm_err = 0
		  				point_cumm_num = 0
						if(reject_if_less_than_minY and currentData[0][6] < minY):
							continue
		  				for j in range(len(currentData)):
		  					xpos_test = currentData[j][2] # cpy posX
		  					ypos_test = currentData[j][3] # cpy posY
		  					label_x = currentData[j][5]
		  					label_y = currentData[j][6]
		  					poly_test = create_poly(np.array(xpos_test, dtype=np.float32).flatten(), np.array(ypos_test, dtype=np.float32).flatten())
		  					inferred = [[np.dot(poly_test,calib_vec_x)[0], np.dot(poly_test,calib_vec_y)[0]]]
		  					label_test = [[label_x, label_y]]
		  					err, num = accuracy(inferred,label_test)
		  					point_cumm_err += err
		  					point_cumm_num += num
		  				cumm_err += point_cumm_err
		  				cumm_num += point_cumm_num
		  				############ print("accuracy for the point:", k,  keyList[k], "acc: ", point_cumm_err/point_cumm_num)
		  				# Set flags (Just to avoid line break in the code)
		  		print("accuracy for full test, acc, step: ", cumm_err/cumm_num, testID)
		  		global_cumm_err += cumm_err
		  		global_cumm_num += cumm_num
	  		print("test complete for mode rejection: ", reject_if_less_than_minY)
	  		print("accuracy for best test clusters test, acc: ", global_cumm_best_centroids_err/global_cumm_best_centroids_num)
	  		print("accuracy for full test, acc: ", global_cumm_err/global_cumm_num)
	  		#write_file_handler.write("tested for repeat_tests, reject_if_less_than_minY, minY: " + str(repeat_tests) + "," + str(reject_if_less_than_minY) + "," + str(minY) +"\n")
	  		#write_file_handler.write("accuracy for best test clusters test, acc: " + str(global_cumm_best_centroids_err/global_cumm_best_centroids_num) + "\n")
	  		#write_file_handler.write("accuracy for full test, acc: " + str(global_cumm_err/global_cumm_num) + "\n")
	  		#write_file_handler.write("====================================================================" + "\n\n")

					

if __name__ == "__main__":
  main()
