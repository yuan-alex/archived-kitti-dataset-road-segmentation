import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import datetime
from termcolor import colored
import keras
from keras.models import load_model
from sklearn.preprocessing import minmax_scale
from numba import jit

frame_dir = "image_2/"
semantic_dir = "gt_image_2/"

def get_frame(i, label="umm"):
	i = "0" * (6 - len(str(i))) + str(i)

	frame = cv.imread("{}{}_{}.png".format(frame_dir, label, i))
	semantic = cv.imread("{}{}_road_{}.png".format(semantic_dir, label, i))

	return frame, semantic

def preprocess_frame(frame):
	frame = frame[1:, :, :]
	if frame.shape != (374, 1242, 3):
		frame = cv.resize(frame, dsize=(1242, 374), interpolation = cv.INTER_AREA)
	return cv.cvtColor(frame, cv.COLOR_BGR2RGB) / 127.5 - 1.0

def postprocess_frame(frame):
	return cv.cvtColor(frame * 127.5 + 1.0, cv.COLOR_RGB2BGR)

def convert_to_float(frame):
	return keras.backend.cast_to_floatx(frame)

def merge_frame(frame, semantic_frame):
	return cv.addWeighted(frame, 0.5, semantic_frame, 1, 0.0)

keras.backend.set_floatx("float32")

model = load_model("model.h5")
keypress = ""

i = 0
while (keypress != ord("q") and i < 95):
	print(colored("[LOG] Predicting frame on index: {}".format(i), "white"))

	frame, ground_truth = get_frame(i)

	X_test = convert_to_float(preprocess_frame(frame))
	X_test = np.expand_dims(X_test, axis=0)

	prediction = model.predict(X_test)[0]
	prediction = postprocess_frame(prediction)

	demo_frame = frame.copy()
	demo_frame = demo_frame[1:, :, :]
	if demo_frame.shape != (374, 1242, 3):
		demo_frame = cv.resize(demo_frame, dsize=(1242, 374), interpolation = cv.INTER_AREA)
	demo_frame = convert_to_float(demo_frame)
	# this is because of how OpenCV does thier shitty imshow
	demo_frame = demo_frame / 255

	img = merge_frame(demo_frame, (prediction - np.min(prediction))/np.ptp(prediction))

	cv.imshow("prediction", prediction)
	cv.imshow("ground_truth", ground_truth)
	cv.imshow("frame", frame)
	cv.imshow("img", img)

	keypress = cv.waitKey(0)
	if keypress == ord("l"):
		i = i + 1
	elif keypress == ord("j"):
		if i != 0:
			i = i - 1

cv.destroyAllWindows()
