import argparse
import os

one_eye = True
selected_eye_id = 'L'

def parse_args():
  """
    Parse command arguments.
  """
  parser = argparse.ArgumentParser(description='rip off data from starbust output so it matches keypoints in defined test (ex. chessboard test)')
  parser.add_argument('path', help='path to starburst file')
  parser.add_argument('-eyeID', type=str, default="N", help='L or R')
  return parser.parse_args()

def main():
  global selected_eye_id
  args = parse_args()
  if(args.eyeID == "L"):
  	selected_eye_id = 'L'
  if(args.eyeID == "R"):
  	selected_eye_id = 'R'
  star_path = args.path
  # checking is ue, et correct
  if(os.path.basename(star_path).split('-')[:-1] == "starburst.csv"):
  	raise ValueError('filename is not ending with -starburst.csv ')
  base_path = os.path.join(os.path.dirname(star_path),os.path.basename(star_path)[:-17]) # remove -starburst.csv
  keypoints_path = base_path + '.csv'	# path for file with all data filtered to match keypoints test 
  save_path = base_path + '-starburst-filtered.csv'
  if(one_eye):
    save_path = save_path[:-4] + '-' + selected_eye_id + '.csv'

  print("star_path, base_path, keypoints_path, save_path :", star_path, base_path, keypoints_path, save_path)

  # for now chessBoard support only
  outputLines = []
  with open(save_path, 'w') as write_file_handler:
  	with open(keypoints_path, 'r') as keypoints_read_handler:
  		with open(star_path, 'r') as star_read_handler:
  			keypointsAllLines = keypoints_read_handler.readlines()[12:] # should be img-name-sorted, but it's not. (Racing when many-threads saving files.)
  			keypointsAllLines = sorted(keypointsAllLines)
  			starAllLines = star_read_handler.readlines() # should be img-name-sorted
  			myit = iter(keypointsAllLines)
  			star_it = iter(starAllLines)
  			imgName_star = ""
  			eyeID_star = ""
  			posX_star, posY_star, area_star = "","",""
  			while(1):
  				try: 
  					imgName_et, eyeID_et, gazeX_et, gazeY_et = next(myit).split(',')
  				except StopIteration: 
  					print("Warning: keypoints list has finished")						
  					break
  				while( imgName_et != imgName_star):
	  				try: 
	  					eyeID_star, imgName_star, posX_star, posY_star, area_star = next(star_it).split(',')
	  				except StopIteration: 
	  					print("Warning: starburst list has finished")						
	  					break
  				if(one_eye):
  					if(eyeID_et == selected_eye_id):
  						outputLines.append(','.join([eyeID_et, imgName_star, posX_star, posY_star, area_star.rstrip(), gazeX_et, gazeY_et]))			
  			write_file_handler.writelines(outputLines)
				
					

if __name__ == "__main__":
  main()
