import os
import face_recognition as fr
import pickle
import numpy as np
import cv2 as cv
import argparse


def CalculateIou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box1 and box2 are tuples of the form (top, right, bottom, left) representing bounding boxes.
    """
    # Calculate the coordinates of the intersection rectangle
    left 	= max(box1[3], box2[3])
    top 	= max(box1[0], box2[0])
    right 	= min(box1[1], box2[1])
    bottom 	= min(box1[2], box2[2])

    # If the boxes do not intersect, return 0
    if right < left or bottom < top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (right - left) * (bottom - top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[1] - box1[3]) * (box1[2] - box1[0])
    box2_area = (box2[1] - box2[3]) * (box2[2] - box2[0])

   
    if intersection_area == box1_area:
    	 #box1 is contained inside box2
    	union_area = box2_area
    
    elif intersection_area == box2_area:
    	#box2 in contained inside box1
    	union_area = box1_area
    else:
    	# Calculate the Union area by subtracting the intersection area from the sum of both areas
    	union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU as the intersection over the union
    iou = intersection_area / union_area

    return iou


def CheckFileExists(path):
    """
    Check if the given file exists or not
    path : Relative Path of file
    """
    # Check if the file exists at the given path, and raise an error if it doesn't
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")


def ChangeFormat(faces_cords):
    """
    Convert the format of face coordinates to match that of the face_locations list.

    Args:
        faces_cords (list): A list of face coordinates.

    Returns:
        list: A list of face coordinates in the format of (top, right, bottom, left).

    Note:
        - The input face coordinate can be either a numpy array or a tuple of two points.
        - The output face coordinate is a tuple of four points in the format of (top, right, bottom, left).
    """
    # Create an empty list to store the converted face coordinates
    arr = []

    # Loop through each face coordinate in the input list
    for face_cords in faces_cords:
        # If the face coordinate is a numpy array
        if isinstance(face_cords, np.ndarray):
            # Extract the x, y, width, and height values from the array
            x, y, w, h = face_cords
            # Calculate the x1, y1, x2, and y2 values based on the x, y, width, and height values
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
        # If the face coordinate is a tuple of two points
        else:
            # Extract the x and y values from the two points
            x1, y1 = face_cords[0]
            x2, y2 = face_cords[1]

        # Calculate the top, bottom, right, and left values based on the x1, y1, x2, and y2 values
        top = min(y1, y2)
        bottom = max(y1, y2)
        right = max(x1, x2)
        left = min(x1, x2)

        # Create a tuple of the face coordinates in the format of (top, right, bottom, left) and append it to the arr list
        face_cords = (top, right, bottom, left)
        arr.append(face_cords)

    # Return the arr list containing the converted face coordinates
    return arr


def GetFaces(img, faces):
    """
    Extract the faces from the image based on their locations and display them in a window.

    Args:
        img (numpy.ndarray): An image.
        faces (list): A list of face locations in the format of (y2, x2, y1, x1).

    Returns:
        None.
    """
    # Loop through each face location in the faces list
    for loc in faces:
        # Extract the y2, x2, y1, and x1 values from the face location
        y2, x2, y1, x1 = loc
        # Extract the face from the image based on the location
        face = img[y2:y1, x1:x2, :]
        # Display the face in a window using OpenCV
        cv.imshow("Faces", face)
        cv.waitKey()
    # Destroy all windows created by OpenCV
    cv.destroyAllWindows()




def main():


	#Parsing
	parser = argparse.ArgumentParser(prog="compare.py",
		description="Compare Vj and HOG via Intersection over Union",
		epilog="Thank you for using!!",
		argument_default=argparse.SUPPRESS,
		allow_abbrev=False,
		fromfile_prefix_chars="@")


	#Arguments
	parser.add_argument("path_to_image")

	#Flags
	parser.add_argument("-d","--data",action="store_true", required=True, help="Path to Image")


	args = parser.parse_args()


	CheckFileExists(args.path_to_image)


	img = fr.load_image_file(args.path_to_image)
	img = img[:,:, ::-1]
	height, width, dim = img.shape

	#Get the Ground truth from the pickle file

	#Get the corresponding pickle
	file_name = os.path.basename(args.path_to_image).split(".")[0]
	dirname = os.path.dirname(args.path_to_image)
	pickle_file = os.path.join(dirname, f"{file_name}.txt")

	#Unpickle to get the bounding box information
	with open(pickle_file, "rb") as f:
		content = f.read(1000)
		faces_cords = pickle.loads(content)

	
	#Changing Format
	faces_cords = faces_cords[1]
	faces_cords = ChangeFormat(faces_cords)

	#Get the ground truth Co-ordinates 

	#Sort faces L to R to (top,right, bottom, left)
	faces_cords = sorted(faces_cords, key=lambda x : x[3])

	#Get the bounding box information for VJ

	#Converting to gray scale
	gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	#Get the VJ Faces
	
	#Getting the Haar Cascade from xml file
	haarcascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

	vj_faces_cords = haarcascade.detectMultiScale(gray_img, 1.4, 9)

	#Changing Format to (top,right, bottom, left)
	vj_faces_cords = ChangeFormat(vj_faces_cords)

	#Sort L to R
	vj_faces_cords = sorted(vj_faces_cords, key=lambda x : x[3])

	#Get the bounding box information for HOG

	#Get the HOG Faces

	#Detect Faces

	#Resize the frame of video to 1/4 for faster face detection 
	# small_img = cv.resize(img, (0,0), fx=0.5, fy=0.5)

	#Get all the faces co-ordinates as (top,right, bottom, left))
	face_locations = fr.face_locations(img)

	#Sort faces from L to R
	face_locations = sorted(face_locations, key=lambda x : x[3])

	iou1 = 0
	iou2 = 0
	no_of_faces = len(faces_cords)
	#Calculate Intersection over Union
	for i in range(no_of_faces):
		iou1 += CalculateIou(faces_cords[i], face_locations[i])
		iou2 += CalculateIou(faces_cords[i], vj_faces_cords[i])

	#Taking the avg of all the iou scores 
	avg_iou1 = iou1/no_of_faces
	avg_iou2 = iou2/no_of_faces
	print(f"IoU Score VJ  : {avg_iou2:.2f}")
	print(f"IoU Score HOG : {avg_iou1:.2f}")




if __name__ == "__main__":
	main()