import numpy as np
import cv2 as cv
import os
import skimage

frame_dir = "image_2/"
semantic_dir = "gt_image_2/"
frame_directory = os.listdir(frame_dir)
semantic_directory = os.listdir(semantic_dir)
frame_directory.sort()
semantic_directory.sort()
label_types = ["um", "umm", "uu"]
label_lengths = [94, 95, 97]

def get_frame(i, label):
	i = "0" * (6 - len(str(i))) + str(i)

	if label == "um":
		semantic = "{}{}_lane_{}.png".format(semantic_dir, label, i)
	else:
		semantic = "{}{}_road_{}.png".format(semantic_dir, label, i)
	frame = "{}{}_{}.png".format(frame_dir, label, i)

	return cv.imread(frame), cv.imread(semantic)

def merge_frame(frame, semantic_frame):
	return cv.addWeighted(frame, 0.5, semantic_frame, 0.5, 0.0)

keypress = ""
i = 0

while (keypress != ord("q") and i < 95):
	frame, semantic_frame = get_frame(i, "umm")
	frame = skimage.img_as_float(frame)
	semantic_frame = skimage.img_as_float(semantic_frame)

	img = merge_frame(frame, semantic_frame)
	cv.imshow("img", img)
	keypress = cv.waitKey(200)

	if keypress == ord("l"):
		i = i + 1
	elif keypress == ord("j"):
		if i != 0:
			i = i - 1
	i = i + 1

cv.destroyAllWindows()
