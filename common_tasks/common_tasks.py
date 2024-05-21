import os
import sys
script_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)
import cvzone
import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time
import datetime
from config import (
    SHOW_DETECT,
    DATA_RECORD,
    RE_CHECK,
    RE_START_TIME,
    RE_END_TIME,
    SD_CHECK,
    SHOW_VIOLATION_COUNT,
    SHOW_TRACKING_ID,
    SOCIAL_DISTANCE,
    SHOW_PROCESSING_OUTPUT,
    YOLO_CONFIG,
    VIDEO_CONFIG,
    DATA_RECORD_RATE,
    ABNORMAL_CHECK,
    ABNORMAL_ENERGY,
    ABNORMAL_THRESH,
    ABNORMAL_MIN_PEOPLE,
    FRAME_SIZE,
    TRACK_MAX_AGE,
    MIN_CONF,
    NMS_THRESH,
)
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from typing import Any, Dict, List, Tuple, Union, Sequence
segmentor: SelfiSegmentation = SelfiSegmentation()



def remove_background_dummy(frame: cv2.typing.MatLike) -> np.ndarray[Any]:
	green: Tuple[int, int, int] = (0, 255, 0)
	imgNoBg: np.ndarray[Any] = segmentor.removeBG(frame, green, cutThreshold = 1)
	return imgNoBg

def remove_background(frame: cv2.typing.MatLike) -> np.ndarray[Any]:
	green: Tuple[int, int, int] = (0, 255, 0)
	imgNoBg: np.ndarray[Any] = segmentor.removeBG(frame, green, cutThreshold = 0.5)
	return imgNoBg

def detect_human (net: cv2.dnn.Net, ln: Sequence[str], frame, encoder: np.ndarray[np.float32], tracker: Tracker, time: Union[int, datetime.datetime]) -> List[List[Tracker]]:
# Get the dimension of the frame
	(frame_height, frame_width) = frame.shape[:2]
	frame_height: int
	frame_width: int
	# Initialize lists needed for detection
	boxes = []
	centroids = []
	confidences = []

	# Construct a blob from the input frame 
	blob: cv2.typing.MatLike = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

	# Perform forward pass of YOLOv3, output are the boxes and probabilities

	net.setInput(blob)
	layer_outputs: Sequence[cv2.typing.MatLike] = net.forward(ln)

	# For each output
	for output in layer_outputs:
		output: cv2.typing.MatLike
		# For each detection in output 
		for detection in output:
			detection: np.ndarray
			# Extract the class ID and confidence 
			scores: np.ndarray = detection[5:]
			class_id: Union[int, np.ndarray] = np.argmax(scores)
			confidence = scores[class_id]
			# Class ID for person is 0, check if the confidence meet threshold
			if class_id == 0 and confidence > MIN_CONF:
				# Scale the bounding box coordinates back to the size of the image
				box: np.ndarray[Any] = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
				(center_x, center_y, width, height) = box.astype("int")
				# Derive the coordinates for the top left corner of the bounding box
				x: int = int(center_x - (width / 2))
				y: int = int(center_y - (height / 2))
				# Add processed results to respective list
				boxes.append([x, y, int(width), int(height)])
				centroids.append((center_x, center_y))
				confidences.append(float(confidence))
	# Perform Non-maxima suppression to suppress weak and overlapping boxes
	# It will filter out unnecessary boxes, i.e. box within box
	# Output will be indexs of useful boxes
	idxs: Sequence[int] = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	tracked_bboxes: List[Tracker] = []
	expired = []
	if len(idxs) > 0:
		del_idxs = []
		for i in range(len(boxes)):
			if i not in idxs:
				del_idxs.append(i)
		for i in sorted(del_idxs, reverse=True):
			del boxes[i]
			del centroids[i]
			del confidences[i]

		boxes: np.ndarray = np.array(boxes)
		centroids: np.ndarray = np.array(centroids)
		confidences: np.ndarray = np.array(confidences)
		features: np.ndarray = np.array(encoder(frame, boxes))
		detections: List[Detection] = [Detection(bbox, score, centroid, feature) for bbox, score, centroid, feature in zip(boxes, scores, centroids, features)]

		tracker.predict()
		expired = tracker.update(detections, time)


		# Obtain info from the tracks
		for track in tracker.tracks:
			track: Track
			if not track.is_confirmed() or track.time_since_update > 5:
					continue 
			tracked_bboxes.append(track)
	return [tracked_bboxes, expired]