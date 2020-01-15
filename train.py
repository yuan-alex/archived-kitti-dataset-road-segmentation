import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import datetime
from termcolor import colored
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from sklearn.preprocessing import minmax_scale

frame_dir = "image_2/"
semantic_dir = "gt_image_2/"

def build_model():
	input = Input(shape=(374, 1242, 3), dtype="float32")

	# ENCODER NETWORK
	encoder_conv1 = Conv2D(64, (4, 4), activation="relu", padding="same")(input)
	encoder_pool1 = MaxPooling2D((2, 2))(encoder_conv1)
	encoder_conv2 = Conv2D(32, (4, 4), activation="relu", padding="same")(encoder_pool1)

	#DECODER NETWORK
	decoder_conv1 = Conv2D(32, (4, 4), activation="relu", padding="same")(encoder_conv2)
	decoder_sample1 = UpSampling2D((2, 2))(decoder_conv1)
	decoder_conv2 = Conv2D(64, (4, 4), activation="relu", padding="same")(decoder_sample1)

	output = Conv2D(3, (4,4), activation="linear", padding="same")(decoder_conv2)

	# BUILD MODEL
	model = Model(input, output)
	model.compile(loss="mse", optimizer="adam")

	return model

def get_frame(i, label="umm"):

	i = "0" * (6 - len(str(i))) + str(i)

	frame = cv.imread("{}{}_{}.png".format(frame_dir, label, i))
	semantic = cv.imread("{}{}_road_{}.png".format(semantic_dir, label, i))
	return frame, semantic

def preprocess_frame(frame):
	frame = frame[1:, :, :]
	if frame.shape != (374, 1242, 3):
		frame = cv.resize(frame, dsize=(1242, 374), interpolation = cv.INTER_AREA)
	return np.expand_dims(cv.cvtColor(frame, cv.COLOR_BGR2RGB) / 127.5 - 1.0, axis=0)

def convert_to_float(frame):
	return keras.backend.cast_to_floatx(frame)

keras.backend.set_floatx("float32")

model = build_model()
epochs = 200

for epoch in range(0, epochs):
	print(colored("\n[EPOCH] Beginning epoch #{} at {}".format(epoch + 1, datetime.datetime.now()), "green"))

	for i in range(0, 95):
		frame, semantic = get_frame(i)
		X_train = convert_to_float(preprocess_frame(frame))
		y_train = convert_to_float(preprocess_frame(semantic))

		model.train_on_batch(X_train, y_train)

	if epoch == int(epochs / 2):
		print(colored("\n[HALF WAY] Half way point reached, saving model for backup; time: ".format(datetime.datetime.now())))
		model.save("model.h5")

print(colored("\n[SUCCESS] Model has been trained; time: {}".format(datetime.datetime.now()), "green"))
model.save("model.h5")
