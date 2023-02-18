import face_recognition as fr
import os
import numpy as np
import cv2 as cv
import argparse 



def CheckFileExists(path):

	"""
	Check if the given file exists or not
	path : Relative Path of file
	"""

	try:
		os.path.exists(path)
	except OSError:
		raise SystemError(f"File don't Exists: {path}")


def main():

	#Argument Parsing
	parser = argparse.ArgumentParser(prog="face-detection",
		description="Get the faces from images",
		epilog="Thank you for using !!",
		argument_default=argparse.SUPPRESS,
		allow_abbrev=False,
		fromfile_prefix_chars="@")

	#Arguments
	parser.add_argument("path_to_image")
	parser.add_argument("path_to_save_faces")
	parser.add_argument("value")


	#Flags
	parser.add_argument("-d", "--data", action="store_true", required=True, help="Path to Image")
	parser.add_argument("-f", "--faces", action="store_true", required=True, help="Path to save Faces")
	parser.add_argument("-t", "--type", action="store_true", required=True, help="File application")

	args = parser.parse_args()
	# print(f'NameSpace : {args}')

	para = args.value

	if para == "1" :

		#Read the image path from the CLI
		path = args.path_to_image

		CheckFileExists(path)

		#Detect Faces
		img = fr.load_image_file(path)

		#Give face co-ordinates as (y1,x1, y2, x2)
		face_locations = fr.face_locations(img)


		#Extract Faces
		mask = np.zeros_like(img, dtype=np.uint8)
		for face_location in face_locations:
			#Bounding Box
			y1,x1,y2,x2 = face_location
			

			pt1 = (x1,y1)
			pt2 = (x2,y2)
			# print(pt1,pt2)
			mask = cv.rectangle(mask, pt1, pt2, (255,255,255), 1)

		#Output image
		out = np.zeros_like(img, dtype=np.uint8)

		#To do matching in all 3 channels
		dim = 3
		for j in range(dim):
			#To make a red box(0,0,255)
			if j != 2:
				out[:,:,j] = np.where(mask[:,:,j] == out[:,:,j], img[:,:,j], 0)
			else:
				out[:,:,j] = np.where(mask[:,:,j] == out[:,:,j], img[:,:,j], 0)


		window_name = os.path.basename(path).split(".")[0]


		#Change color mapping
		out = out[:,:,[2,1,0]]
		cv.imshow(window_name,out)
		cv.waitKey()
		cv.destroyAllWindows()

	else:
		pass


	#Save the Extracted Face



if __name__ == "__main__":
	main()