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
    # Check if the file exists at the given path, and raise an error if it doesn't
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")

def CheckDirExists(path):
    """
    Check if the given Directory exists or not 
    if not make it
    path : Relative Path of file
    """
    # Check if the directory exists at the given path, and create it if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)

import os
import cv2 as cv
import numpy as np


def FaceDetectionExtraction(img, face_locations, scaling, results_path=None, filename=None, extraction=False):
    """
    Extracts faces from an image and optionally saves the extracted faces and/or a copy of the image with
    bounding boxes drawn around the faces.
    
    Parameters:
    img (numpy.ndarray)			: The image to be analyzed.
    face_locations (list)		: The location of the faces in the image.
    scaling (float)			: The scaling factor used for face detection.
    results_path (str, optional)	: The path to the directory where extracted faces and/or the output image should be saved.
    filename (str, optional)		: The name of the output file to be saved. If multiple faces are detected, a suffix with the face index will be appended to the filename.
    extraction (bool, optional)		: Whether or not to extract and save the faces.
    
    Returns:
    numpy.ndarray : The output image with bounding boxes drawn around the faces.
    """

    # Create a black mask with the same dimensions as the input image.
    mask = np.zeros_like(img, dtype=np.uint8)

    # Loop over each detected face.
    for i, face_location in enumerate(face_locations):

        # Extract the coordinates of the face bounding box.
        y1, x1, y2, x2 = face_location

        # Scale the face bounding box back up to the original image size.
        y1 = int(y1 * (scaling ** -1))
        x1 = int(x1 * (scaling ** -1))
        y2 = int(y2 * (scaling ** -1))
        x2 = int(x2 * (scaling ** -1))

        # If requested, extract and save the face.
        if extraction:

            # Extract the face from the original image.
            face = img[y1:y2, x2:x1, :]
            # face = face[:, :, [2, 1, 0]]  # Reorder color channels from BGR to RGB.

            # Determine the suffix for the output filename, based on the number of detected faces.
            if len(face_locations) > 1:
                suffix = f"face{filename}suffix0{i+1}"
            else:
                suffix = f"{filename}"

            # Ensure that the output directory exists.
            CheckDirExists(results_path)

            # Save the extracted face.
            output_img_loc = os.path.join(results_path, f"{suffix}.jpg")
            cv.imwrite(output_img_loc, face)

        # Draw a white bounding box around the face on the mask.
        mask = cv.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Create an output image with the same dimensions as the input image.
    out = np.zeros_like(img, dtype=np.uint8)

    # Iterate over each color channel in the output image.
    for j in range(3):

        # Set the output image pixels to red (0, 0, 255) where the corresponding pixel in the mask is white.
        if j == 2:
            out[:, :, j] = np.where(mask[:, :, j] == 255, 255, img[:, :, j])
        else:
            out[:, :, j] = np.where(mask[:, :, j] == 255, 0, img[:, :, j])

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
	parser.add_argument("path_to_data")
	parser.add_argument("path_to_save_result")
	parser.add_argument("value")


	#Flags
	parser.add_argument("-d", "--data", action="store_true", required=True, help="Path to Data")
	parser.add_argument("-f", "--faces", action="store_true", required=True, help="Path to save results")
	parser.add_argument("-t", "--type", action="store_true", required=True, help="File application")

	args = parser.parse_args()
	# print(f'NameSpace : {args}')

	para = args.value

	if para == "1" :

		#Read the image path from the CLI
		path = args.path_to_data
		results_path = args.path_to_save_result

		CheckFileExists(path)

		#Detect Faces
		img = fr.load_image_file(path)

		height, width, dim = img.shape

		#Resize the frame of video to 1/4 for faster face detection 
		small_img = cv.resize(img, (0,0), fx=1.0, fy=1.0)

		#Get all the faces co-ordinates as (y1,x1, y2, x2)
		face_locations = fr.face_locations(small_img)

		#If no face is found in image
		if len(face_locations) == 0:
			#Transposed Image
			new_image = np.zeros((width, height, dim), dtype=np.uint8)

			# Copy the original image onto the new image with the dimensions swapped
			cv.transpose(img, new_image)
			cv.flip(new_image, 1, new_image)
			img = new_image

			#Resize the frame of video to 1/4 for faster face detection 
			small_img = cv.resize(img, (0,0), fx=1.0, fy=1.0)

			face_locations = fr.face_locations(small_img)


		filename = os.path.basename(path).split(".")[0]
		#Change color mapping
		img = img[:,:,::-1]

		out = FaceDetectionExtraction(img, face_locations, 1.0, results_path, filename, True)
		
		window_name = os.path.basename(path).split(".")[0]

		cv.imshow(window_name,out)
		cv.waitKey()
		cv.destroyAllWindows()

	elif para == "2":

		#Read the video from CLI
		video_path = args.path_to_data

		CheckFileExists(video_path)

		video = cv.VideoCapture(video_path)

		#Grab a single frame from the video
		# video.set(cv.CAP_PROP_POS_MSEC,1)
		ret, frame = video.read()
		
		#Path to save results
		if os.path.basename(video_path) == "arnold.mp4":
			result_path = os.path.join(args.path_to_save_result, os.path.basename(video_path))
		else:
			file_name = os.path.basename(video_path).split(".")[0]
			result_path = os.path.join(args.path_to_save_result, f"{file_name}Output?.mp4")

		output = cv.VideoWriter(result_path, cv.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))

		process_frame = True
		#Read the Video Frame by Frame
		while True:


			#Only process every other frame from the video to save time
			if process_frame:

				#Resize the frame of video to 1/4 for faster face detection 
				small_frame = cv.resize(frame, (0,0), fx=1.0, fy=1.0)

				#convert image from BGR (which opencv uses) to RGB (which face_recognition uses)
				small_frame = small_frame[:,:, ::-1]


				#Get all the faces co-ordinates as (y1,x1, y2, x2)
				face_locations = fr.face_locations(small_frame)

			process_frame = not process_frame

			out = FaceDetectionExtraction(frame, face_locations, 1.0)

			# out = out[:,:,::-1]

			#Save the result 
			output.write(out)
			ret, frame = video.read()
			if ret == False:
				break
			# cv.waitKey()

		video.release()
		output.release()
		cv.destroyAllWindows()

	else:
		pass

if __name__ == "__main__":

	
	face_detection_main()