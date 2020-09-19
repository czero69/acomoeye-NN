import argparse
import os
#from sets import Set
# Run it in PY 3.X
# from ctypes import c_longlong as ll # time points may be even long long, EDIT: python long does not have a defined limit which is named just int in PY 3.X

m_timeToNextTestCase = 7000
CHESSBAORD_TIME_START = int(m_timeToNextTestCase/8) # in ms. time after switching to new keypoint that will matched. Everything earlier will be rejected.
CHESSBOARD_TIME_END = int(m_timeToNextTestCase*6/10) # in ms. time till matching will last after switching to new keypoint. Everything after that will be rejected.
DELETE_NOT_USED_FILES = True # will delete files that are in -downscaled dir and are not used.

#CHESSBAORD: warning(1), recorded ueTimePoint means ENDING of the sequence (everything that was before has angles: gazeX, gazeY)

def parse_args():
  """
    Parse command arguments.
  """
  parser = argparse.ArgumentParser(description='merge ue-et datasets by time')
  parser.add_argument('path', help='Path to ue file')
  parser.add_argument('-test-type', type=str, default='chessBoard',
                      help='type of test to be merged: {chessBoard, smooth}')
  return parser.parse_args()

def chessBoardTest(outputLines, ueAllLines, myit, write_file_handler):
	etTimePoint = int(0)	
	for u in ueAllLines:
		eyeID_ue, gazeX, gazeY, ueTimePoint_str = u.split(',')
		# first - rewind iter to start condition
		while(etTimePoint < (int(ueTimePoint_str) - m_timeToNextTestCase + CHESSBAORD_TIME_START) ): # -m_timeToNextTestCase, see warning(1) above 
			try: 
				eyeID_et, imgName, etTimePoint_str = next(myit).split(',')
				etTimePoint = int(etTimePoint_str)
			except StopIteration: 
				print("Warning: et list has finished early and it lacks data")						
				break
		# record data till we reach end condition
		while(etTimePoint < (int(ueTimePoint_str) - m_timeToNextTestCase + CHESSBOARD_TIME_END) ):
			try: 
				eyeID_et, imgName, etTimePoint_str = next(myit).split(',')
				etTimePoint = int(etTimePoint_str)
			except StopIteration: 
				print("Warning: et list has finished early and it lacks data")
				break
			if(eyeID_et == eyeID_ue):
				outputLines.append(imgName+","+eyeID_et+","+gazeX+","+gazeY+"\n")
	

def main():
  args = parse_args()
  ue_path = args.path
  # checking is ue, et correct
  if(os.path.basename(ue_path).split('-')[:-1] == "ue.csv"):
  	raise ValueError('filename is not ending with -ue.csv ')
  base_path = os.path.join(os.path.dirname(ue_path),os.path.basename(ue_path)[:-7]) # remove -ue.csv
  et_path = base_path + '-et.csv'
  save_path = base_path + '.csv'
  downscaled_path = base_path + '_ir-downscaled'

  print("ue_path, base_path, et_path, save_path :", ue_path, base_path, et_path, save_path)
  test_type = args.test_type

  # for now chessBoard support only
  outputLines = []
  with open(save_path, 'w') as write_file_handler:
  	with open(et_path, 'r') as et_read_handler:
  		with open(ue_path, 'r') as ue_read_handler:
  			etAllLines = et_read_handler.readlines() # should be time-sorted
  			ueAllLines = ue_read_handler.readlines()[1:] # skip header, should be time-sorted
  			myit = iter(etAllLines)
  			if(test_type == "chessBoard"):
  				chessBoardTest(outputLines, ueAllLines, myit, write_file_handler)
  			write_file_handler.writelines(outputLines)
  			print(" ...done merging ue and et data")
  			if(DELETE_NOT_USED_FILES):
  				print(" ...now deleting not used images in -downscaled dir")
  				filenames_set = set()
  				filenames_et = []
  				for ol in outputLines:
  					filenames_set.add(ol.split(',')[0])
  				for et in etAllLines:
  					filenames_et.append(et.split(',')[1])
  				for fe in filenames_et:
  					if(not(fe in filenames_set)):
  						path_fe = os.path.join(downscaled_path,fe)
  						if os.path.exists(path_fe):
  							os.remove(path_fe)
  						else:
  							print("The file does not exist and cannot be deleted: ", path_fe)
				
  				print(" ...done deleting")
				
					

if __name__ == "__main__":
  main()
