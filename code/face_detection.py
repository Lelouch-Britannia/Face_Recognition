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


def CheckDirExists(path):

	"""
	Check if the given Directory exists or not 
	if not make it
	path : Relative Path of file
	"""
	try:
		if not os.path.exists(path):
			os.makedirs(path)
	except OSError:
		raise SystemError(f"Directory don't Exists: {path}")

def FaceDetectionExtraction(img, face_locations, results_path, filename):
	"""
	For Bounding box around all the faces in the given image
	and extraction all the faces and saving
	img 			: Image to be analysed
	face_locations 	: Location of faces in the image
	filename 		: For saving the faces
	"""

	#Creating mask 
	mask = np.zeros_like(img, dtype=np.uint8)
	for i in range(len(face_locations)):

		#Face Co-ordinates
		y1,x1,y2,x2 = face_locations[i]
		# print(x1,y1,x2,y2)

		#Save the Extracted Face
		#Output Face(Careful with coordinates!!)
		face = img[y1:y2, x2:x1, :]
		face = face[:,:,[2,1,0]]
		
		
		#Check the number of Faces
		if len(face_locations) > 1:
			suffix = f"face{filename}suffix0{i+1}"
		else:
			suffix = f"{filename}"

		#Check if the Dir exists
		CheckDirExists(results_path)
		#Save the results
		output_img_loc = os.path.join(results_path, f"{suffix}.jpg")
		cv.imwrite(output_img_loc, face)

		mask = cv.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), 1)

	#Output image
	out = np.zeros_like(img, dtype=np.uint8)

	#To do matching in all 3 channels
	dim = 3
	for j in range(dim):
		#To make a red box(0,0,255)
		if j == 0:
			out[:,:,j] = np.where(mask[:,:,j] == out[:,:,j], img[:,:,j], 255)
		else:
			out[:,:,j] = np.where(mask[:,:,j] == out[:,:,j], img[:,:,j], 0)


	return out



def face_detection_main():

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
		results_path = args.path_to_save_faces

		CheckFileExists(path)

		#Detect Faces
		img = fr.load_image_file(path)

		height, width, dim = img.shape
		# print(img.shape)
		# if height < width:

		# 	#Transposed Image
		# 	new_image = np.zeros((width, height, dim), dtype=np.uint8)

		# 	# Copy the original image onto the new image with the dimensions swapped
		# 	cv.transpose(img, new_image)
		# 	cv.flip(new_image, 1, new_image)
		# 	img = new_image
		# 	print(img.shape)


		#Give faces co-ordinates as (y1,x1, y2, x2)
		face_locations = fr.face_locations(img, number_of_times_to_upsample=3)

		#If no face is found in image
		if len(face_locations) == 0:
			#Transposed Image
			new_image = np.zeros((width, height, dim), dtype=np.uint8)

			# Copy the original image onto the new image with the dimensions swapped
			cv.transpose(img, new_image)
			cv.flip(new_image, 1, new_image)
			img = new_image
			face_locations = fr.face_locations(img, number_of_times_to_upsample=3)


		filename = os.path.basename(path).split(".")[0]

		out = FaceDetectionExtraction(img, face_locations, results_path,filename)
		
		window_name = os.path.basename(path).split(".")[0]

		#Change color mapping
		out = out[:,:,[2,1,0]]
		cv.imshow(window_name,out)
		cv.waitKey()
		cv.destroyAllWindows()

	else:
		pass


	



if __name__ == "__main__":
	face_detection_main()