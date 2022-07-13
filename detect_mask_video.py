# import the necessary packages
import os
import cv2
import time
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
from imutils.video import VideoStream

#function that takes input frame, face_detector, mask_detector and returns the face location and probability of face mask being worn
def detect_and_predict_mask(frame, faceNet, maskNet):
	# creating a blob from each frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# pass the blob through the network to obtain face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the probability associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding rectangle for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensuring that the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# resizing and preprocessing of face object
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# prediction made if atleast one face is detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a tuple of face locations(x,y coords) and probability of face mask 
	return (locs, preds)

# load the face detector model(ResNet) from disk
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the mask detector model from disk
maskNet = load_model("mask_detector.model")

# satrt the video stream
vs = VideoStream(src=0).start()

while True:
	# read the frame and resize it to 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and the probability of mask being worn
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determining the class label and color for the bounding rectangle and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# including the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# displaying the label and bounding rectangle on the output frame
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# exiting from the loop on pressing the "q" key
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()