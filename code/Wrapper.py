import argparse 
import sys
import os
from face_detection import face_detection_main




def Wrapper_main():

	#Get all the Images 
	parser = argparse.ArgumentParser(prog="Wrapper",
		description="Wrapper to face detection",
		epilog="Saves Time",
		argument_default=argparse.SUPPRESS,
		allow_abbrev=False)

	#Arguments
	parser.add_argument("directory_of_images")
	#Flags
	parser.add_argument("-d", "--dir", action="store_true", required=True, help="Path to directory containing Images")

	args = parser.parse_args()

	files_list = os.listdir(args.directory_of_images)

	#Other Arguments
	data = "--data"
	faces = "--faces"
	result_dir = "/Users/ltc/Documents/Python Scripts/CS763/22M1079_22M1177_22M2119_Task02/results/faceDetection"
	type_var = "--type"
	value = "1"


	for file in files_list:

		#Get the absolute path
		file_path = os.path.join(args.directory_of_images, file)
		#Modifying sys argv before sending to face_detection 
		sys.argv = [sys.argv[0], data, file_path, faces, result_dir, type_var, value]
		face_detection_main()


if __name__ == "__main__":
	Wrapper_main()