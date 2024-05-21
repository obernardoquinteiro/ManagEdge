import os
import sys
import time
import datetime
import numpy as np
import socket
import ssl
import pickle
import psutil
import csv
import json
import cvzone
import argparse
script_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)
import multiprocessing
import multiprocessing.shared_memory
import imutils
import cv2
from pynput import keyboard
from math import ceil
from scipy.spatial.distance import euclidean
from util import rect_distance, progress, kinetic_energy
from colors import RGB_COLORS
from config import (
	Setup,
	CLIENT_KEY_PATH,
	CLIENT_CERT_PATH,
	MODEL_FILENAME_PATH,
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
	MIN_THRESHOLD_CAMSERVER_IMBALANCE,
	MAX_THRESHOLD_CAMSERVER_IMBALANCE,
	MIN_THRESHOLD_CENTRAL_IMBALANCE,
	MAX_THRESHOLD_CENTRAL_IMBALANCE,
	MAX_SIZE_LISTEN_LIST,
	SERVER_HOST,
	SERVER_PORT,
	find_git_root
)
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker, Track
from deep_sort import generate_detections as gdet
from common_tasks.common_tasks import remove_background, detect_human, remove_background_dummy

from cvzone.SelfiSegmentationModule import SelfiSegmentation
IS_CAM: bool = VIDEO_CONFIG["IS_CAM"]
HIGH_CAM: bool = VIDEO_CONFIG["HIGH_CAM"]

from enum import Enum
from typing import Any, Dict, List, Tuple, Union, Sequence





listenList: List = []
#currentSetup: int = 1
totalDataTransferVolume: int = 0
numTransfers: int = 0

#current_setup_shm = multiprocessing.shared_memory.SharedMemory(create=True, size=4)  # 4 bytes for an integer
#initial_setup_value = 1  # Assuming the initial setup value is 1
#current_setup_array = np.array([initial_setup_value], dtype=np.int32)
#current_setup_shm.buf[:] = current_setup_array.tobytes()

lock = multiprocessing.Lock()



segmentor: SelfiSegmentation = SelfiSegmentation()
client: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client: ssl.SSLSocket = ssl.wrap_socket(client, keyfile=CLIENT_KEY_PATH, certfile=CLIENT_CERT_PATH)

terminate_signal_sent: bool = False

def terminate_video():
	print("Sending termination signal ('QUIT')")
	client.sendall(b"QUIT")

def on_key_press(key: str) -> None:
    global terminate_signal_sent
    try:
        if key.char == 'q':
            terminate_video()
            terminate_signal_sent = True
    except AttributeError:
        pass


def _record_movement_data(movement_data_writer: csv.writer, movement: Track)-> None:
	track_id: int = movement.track_id 
	entry_time = movement.entry 
	exit_time = movement.exit			
	positions = movement.positions
	positions = np.array(positions).flatten()
	positions = list(positions)
	data = [track_id] + [entry_time] + [exit_time] + positions
	movement_data_writer.writerow(data)

def _record_crowd_data(time, human_count, violate_count, restricted_entry, abnormal_activity, crowd_data_writer):
	data = [time, human_count, violate_count, int(restricted_entry), int(abnormal_activity)]
	crowd_data_writer.writerow(data)

def _end_video(tracker, frame_count, movement_data_writer):
	for t in tracker.tracks:
		if t.is_confirmed():
			t.exit = frame_count
			_record_movement_data(movement_data_writer, t)
		

def client_init(cap: cv2.VideoCapture, frame_size: int, net: cv2.dnn.Net, ln: Sequence[str], encoder: np.ndarray[np.float32], tracker: Tracker, movement_data_writer: csv.writer, crowd_data_writer: csv.writer, host: str, port: int, currentSetup: int):
	try:
		
		#print("Connecting to host ", host, " and port ", port)
		client.connect((host, port))
		video_process(cap, frame_size, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer, currentSetup)
	except Exception as e:
		print(f"Error: {e}")

def checkForImbalance(currentSetupDict, lock):
	global listenList
	imbalanceSum:float = 0
	for x in listenList:
		imbalanceSum+=x
	imbalanceIndex:float = float(imbalanceSum/MAX_SIZE_LISTEN_LIST)
	if currentSetupDict[0] == Setup.ALL_EDGE.value:
		if imbalanceIndex<=MIN_THRESHOLD_CAMSERVER_IMBALANCE:
			with lock:
				print(imbalanceIndex)
				print("Change 1 Edge")
				currentSetupDict[0] = Setup.HALF_EDGE_SERVER.value
			listenList = []
	elif currentSetupDict[0]  == Setup.HALF_EDGE_SERVER.value:
		if imbalanceIndex<=MIN_THRESHOLD_CAMSERVER_IMBALANCE:
			with lock:
				print(imbalanceIndex)
				print("Change 2 Edge")
				currentSetupDict[0]  = Setup.ALL_SERVER.value
			listenList = []
		elif imbalanceIndex>=MAX_THRESHOLD_CAMSERVER_IMBALANCE:
			with lock:
				print(imbalanceIndex)
				print("Change 0 Edge")
				currentSetupDict[0] = Setup.ALL_EDGE.value
			listenList = []
	elif currentSetupDict[0]  == Setup.ALL_SERVER.value:
		if imbalanceIndex>=MAX_THRESHOLD_CAMSERVER_IMBALANCE:
			with lock:
				print(imbalanceIndex)
				print("Change 1 Edge")
				currentSetupDict[0] = Setup.HALF_EDGE_SERVER.value
			listenList = []
	if currentSetupDict[0] == Setup.BACK_SERVER.value:
		if imbalanceIndex<=MIN_THRESHOLD_CAMSERVER_IMBALANCE:
			with lock:
				print(imbalanceIndex)
				print("Change 5 Edge")
				currentSetupDict[0] = Setup.BACK_EDGE.value
			listenList = []
	elif currentSetupDict[0] == Setup.BACK_EDGE.value:
		if imbalanceIndex>=MAX_THRESHOLD_CAMSERVER_IMBALANCE:
			with lock:
				print(imbalanceIndex)
				print("Change 4 Edge")
				currentSetupDict[0] = Setup.BACK_SERVER.value
			listenList = []


def listenToServer(currentSetupDict, lock):
	global listenList
	while True:	
		data = b""
		while True:
			# Receive a chunk of data
			try:
				chunk = client.recv(4096)
				data += chunk
			except Exception as e:
					data = b"QUIT"
					break
			# Check if the end of the frame is reached
			if data[-2:] == b"\xff\xd9":
				break	
			if data == b"QUIT":
				print("Received Termination Signal ListenToServer")
				break
		if data == b"QUIT":
			break
		data = data[:-2]
		received_data = pickle.loads(data)
		imbalanceIndex:float = received_data["imbalanceIndex"]
		if len(listenList) == MAX_SIZE_LISTEN_LIST:
			listenList.pop(0)
			listenList.append(imbalanceIndex)
			checkForImbalance(currentSetupDict, lock)
		else:
			listenList.append(imbalanceIndex)


def video_process(cap: cv2.VideoCapture, frame_size: int, net: cv2.dnn.Net, ln: Sequence[str], encoder: np.ndarray[np.float32], tracker: Tracker, movement_data_writer: csv.writer, crowd_data_writer: csv.writer, currentSetup: int):
	totalDataTransferVolume: int = 0
	numTransfers: int = 0
	listener: keyboard = keyboard.Listener(on_press=on_key_press)
	listener.start()
	def _calculate_FPS() -> None:
		t1: float = time.time() - t0
		VID_FPS = frame_count / t1

	if IS_CAM:
		VID_FPS = None
		DATA_RECORD_FRAME: int = 1
		TIME_STEP: int = 1
		t0: float = time.time()
	else:
		VID_FPS: float = cap.get(cv2.CAP_PROP_FPS)
		DATA_RECORD_FRAME: int = int(VID_FPS / DATA_RECORD_RATE)
		TIME_STEP: int = DATA_RECORD_FRAME/VID_FPS

	frame_count: int = 0
	display_frame_count: int = 0
	re_warning_timeout: int = 0
	sd_warning_timeout: int = 0
	ab_warning_timeout: int = 0

	RE: bool = False
	ABNORMAL: bool = False
	

	manager = multiprocessing.Manager()
	currentSetupDict = manager.dict()
	processListen = multiprocessing.Process(target=listenToServer, args=(currentSetupDict, lock))
	processListen.daemon = True  # Daemonize the process
	processListen.start()
	currentSetupDict[0] = currentSetup
	START_TIME: float = time.time()
	try:
		while not terminate_signal_sent:
			ITERATE_START_TIME = time.time()
			currentSetup = currentSetupDict[0]
			#print("currentSetupDict ", currentSetupDict[0])
			#print("currentSetup ", currentSetup)
			(ret_aux, frame_aux) = cap.read()
			ret: bool = ret_aux
			frame: cv2.typing.MatLike = frame_aux
			#frame = remove_background_dummy(frame_aux)
			#cv2.imshow("Captured Frame", frame)
			# Stop the loop when video ends
			if not ret:
				terminate_video()
				_end_video(tracker, frame_count, movement_data_writer)
				if not VID_FPS:
					_calculate_FPS()
				break
			#if currentSetup == 5:
			if currentSetup == Setup.BACK_EDGE.value:
				frame= remove_background(frame)
				#cv2.imshow("Processed Frame", frame)
			"""else:
				frame = remove_background_dummy(frame)"""
			#cv2.imshow("Client Frame", frame)	
			#if currentSetup == 2 or currentSetup == 4 or currentSetup == 5:
			if currentSetup == Setup.ALL_SERVER.value or currentSetup == Setup.BACK_SERVER.value or currentSetup == Setup.BACK_EDGE.value:
				#SEND TO SERVER
				frame_count +=1
				#print("Sending Frame Shape:", frame.shape)
				frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
				ITERATE_END_TIME = time.time()
				data_to_send = {
					"frame": frame_bytes,
					"setup": currentSetup,
					"processingTime": float(ITERATE_END_TIME-ITERATE_START_TIME)
				}
				#DEBUG_THREADING DELETE AFTER
				# Convert the data to bytes using pickle

				data_bytes = pickle.dumps(data_to_send)
				new_bytes = bytes.fromhex('ffd9')
				data_bytes += new_bytes
				#print(data_bytes[-2:])
				# Send the data to the server
				totalDataTransferVolume += len(data_bytes)
				numTransfers += 1
				#print(len(data_bytes))
				client.sendall(data_bytes)
				continue

			# Update frame count
			if frame_count > 1000000:
				if not VID_FPS:
					_calculate_FPS()
				frame_count: int = 0
				display_frame_count: int = 0
			frame_count += 1
			
			# Skip frames according to given rate
			if frame_count % DATA_RECORD_FRAME != 0:
				continue

			display_frame_count += 1

			# Resize Frame to given size
			frame: np.ndarray[Any] = imutils.resize(frame, width=frame_size)

			# Get current time
			current_datetime: datetime.datetime = datetime.datetime.now()

			# Run detection algorithm
			record_time: Union[int, datetime.datetime]
			if IS_CAM:
				record_time = current_datetime
			else:
				record_time = frame_count
			#DEBUG TESTING THREADING
			# Run tracking algorithm
			[humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, record_time)
			humans_detected: List[Tracker]
			expired: List[Tracker]

			#if currentSetup == 1:
			if currentSetup == Setup.HALF_EDGE_SERVER.value:
				frame_bytes: bytes = cv2.imencode(".jpg", frame)[1].tobytes()
				ITERATE_END_TIME = time.time()
				# Serialize the data to be sent
				data_to_send: Dict[Any, Any] = {
					"frame": frame_bytes,
					"humans_detected": humans_detected,
					"expired": expired,
					"setup": currentSetup,
					"processingTime": float(ITERATE_END_TIME-ITERATE_START_TIME)
				}

				# Convert the data to bytes using pickle
				data_bytes: bytes = pickle.dumps(data_to_send)
				new_bytes: bytes = bytes.fromhex('ffd9')
				data_bytes += new_bytes
				# Send the data to the server
				totalDataTransferVolume += len(data_bytes)
				numTransfers += 1
				#print(len(data_bytes))
				client.sendall(data_bytes)
				continue

			# Record movement data
			for movement in expired:
				movement: Tracker
				_record_movement_data(movement_data_writer, movement)
			
			# Check for restricted entry
			if RE_CHECK:
				RE: bool = False
				if (current_datetime.time() > RE_START_TIME) and (current_datetime.time() < RE_END_TIME) :
					if len(humans_detected) > 0:
						RE = True
				
			# Initiate video process loop
			if SHOW_PROCESSING_OUTPUT or SHOW_DETECT or SD_CHECK or RE_CHECK or ABNORMAL_CHECK:
				# Initialize set for violate so an individual will be recorded only once
				violate_set = set()
				# Initialize list to record violation count for each individual detected
				violate_count = np.zeros(len(humans_detected))

				# Initialize list to record id of individual with abnormal energy level
				abnormal_individual = []
				ABNORMAL = False
				for i, track in enumerate(humans_detected):
					# Get object bounding box
					[x, y, w, h] = list(map(int, track.to_tlbr().tolist()))
					# Get object centroid
					[cx, cy] = list(map(int, track.positions[-1]))
					# Get object id
					idx = track.track_id
					# Check for social distance violation
					if SD_CHECK:
						if len(humans_detected) >= 2:
							# Check the distance between current loop object with the rest of the object in the list
							for j, track_2 in enumerate(humans_detected[i+1:], start=i+1):
								if HIGH_CAM:
									[cx_2, cy_2] = list(map(int, track_2.positions[-1]))
									distance = euclidean((cx, cy), (cx_2, cy_2))
								else:
									[x_2, y_2, w_2, h_2] = list(map(int, track_2.to_tlbr().tolist()))
									distance = rect_distance((x, y, w, h), (x_2, y_2, w_2, h_2))
								if distance < SOCIAL_DISTANCE:
									# Distance between detection less than minimum social distance 
									violate_set.add(i)
									violate_count[i] += 1
									violate_set.add(j)
									violate_count[j] += 1

					# Compute energy level for each detection
					if ABNORMAL_CHECK:
						ke = kinetic_energy(track.positions[-1], track.positions[-2], TIME_STEP)
						if ke > ABNORMAL_ENERGY:
							abnormal_individual.append(track.track_id)

					# If restrited entry is on, draw red boxes around each detection
					if RE:
						cv2.rectangle(frame, (x + 5 , y + 5 ), (w - 5, h - 5), RGB_COLORS["red"], 5)
		
					cv2.putText(frame, str(int(display_frame_count)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["red"], 2)
					# Draw yellow boxes for detection with social distance violation, green boxes for no violation
					# Place a number of violation count on top of the box
					if i in violate_set:
						cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["yellow"], 2)
						if SHOW_VIOLATION_COUNT:
							cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
					elif SHOW_DETECT and not RE:
						cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["green"], 2)
						if SHOW_VIOLATION_COUNT:
							cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
					
					if SHOW_TRACKING_ID:
						cv2.putText(frame, str(int(idx)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
				
				# Check for overall abnormal level, trigger notification if exceeds threshold
				if len(humans_detected)  > ABNORMAL_MIN_PEOPLE:
					if len(abnormal_individual) / len(humans_detected) > ABNORMAL_THRESH:
						ABNORMAL = True

			# Place violation count on frames
			if SD_CHECK:
				# Warning stays on screen for 10 frames
				if (len(violate_set) > 0):
					sd_warning_timeout = 10
				else: 
					sd_warning_timeout -= 1
				# Display violation warning and count on screen
				if sd_warning_timeout > 0:
					text = "Violation count: {}".format(len(violate_set))
					cv2.putText(frame, text, (200, frame.shape[0] - 30),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

			# Place restricted entry warning
			if RE_CHECK:
				# Warning stays on screen for 10 frames
				if RE:
					re_warning_timeout = 10
				else: 
					re_warning_timeout -= 1
				# Display restricted entry warning and count on screen
				if re_warning_timeout > 0:
					if display_frame_count % 3 != 0 :
						cv2.putText(frame, "RESTRICTED ENTRY", (200, 100),
							cv2.FONT_HERSHEY_SIMPLEX, 1, RGB_COLORS["red"], 3)

			# Place abnormal activity warning
			if ABNORMAL_CHECK:
				if ABNORMAL:
					# Warning stays on screen for 10 frames
					ab_warning_timeout = 10
					# Draw blue boxes over the the abnormally behave detection if abnormal activity detected
					for track in humans_detected:
						if track.track_id in abnormal_individual:
							[x, y, w, h] = list(map(int, track.to_tlbr().tolist()))
							cv2.rectangle(frame, (x , y ), (w, h), RGB_COLORS["blue"], 5)
				else:
					ab_warning_timeout -= 1
				if ab_warning_timeout > 0:
					if display_frame_count % 3 != 0:
						cv2.putText(frame, "ABNORMAL ACTIVITY", (130, 250),
							cv2.FONT_HERSHEY_SIMPLEX, 1.5, RGB_COLORS["blue"], 5)

			# Display crowd count on screen
			if SHOW_DETECT:
				text = "Crowd count: {}".format(len(humans_detected))
				cv2.putText(frame, text, (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
			
			# Record crowd data to file
			if DATA_RECORD:
				_record_crowd_data(record_time, len(humans_detected), len(violate_set), RE, ABNORMAL, crowd_data_writer)

			# Display video output or processing indicator
			if SHOW_PROCESSING_OUTPUT:
				#SEND TO SERVER
				frame_count +=1
				#print("Sending Frame Shape:", frame.shape)
				frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
				ITERATE_END_TIME = time.time()
				data_to_send = {
					"frame": frame_bytes,
					"setup": currentSetup,
					"processingTime": float(ITERATE_END_TIME-ITERATE_START_TIME)
				}
				#DEBUG_THREADING DELETE AFTER
				#print("Single iteration time: "+ str(ITERATE_END_TIME-ITERATE_START_TIME))
				# Convert the data to bytes using pickle

				data_bytes = pickle.dumps(data_to_send)
				new_bytes = bytes.fromhex('ffd9')
				data_bytes += new_bytes
				#print(data_bytes[-2:])
				# Send the data to the server
				totalDataTransferVolume += len(data_bytes)
				numTransfers += 1
				#print(len(data_bytes))
				client.sendall(data_bytes)
				continue
				#cv2.imshow("Processed Output", frame)
			else:
				progress(display_frame_count)

			# Press 'Q' to stop the video display
			if cv2.waitKey(1) & 0xFF == ord('q'):
				# Record the movement when video ends
				_end_video(tracker, frame_count, movement_data_writer)
				# Compute the processing speed
				if not VID_FPS:
					_calculate_FPS()
				break
	except KeyboardInterrupt:
		print("Keyboard interrupt. Killing client")
	except Exception as e:
		print(f"Error server: {e}")
	END_TIME: float = time.time()
	PROCESS_TIME: float = END_TIME - START_TIME
	print(f"Average data transfer volume = {totalDataTransferVolume/numTransfers}")
	print("Time elapsed: ", PROCESS_TIME)
	print("Frames sent per second: ", str(float(numTransfers/PROCESS_TIME)))
	cv2.destroyAllWindows()
	listener.stop()
	listener.join()
	client.close()
	processListen.join()


def main() -> None:
	parser = argparse.ArgumentParser(description='Server for video processing')
	parser.add_argument('--host', type=str, default=None, help='Host IP to bind')
	parser.add_argument('--port', type=int, default=None, help='Port number to listen on')
	parser.add_argument('--video', type=str, default=None, help='Video to be processed')
	parser.add_argument('--setup', type=int, default=None, help='Setup to start with')
	args = parser.parse_args()

	host = args.host if args.host is not None else SERVER_HOST
	port = args.port if args.port is not None else SERVER_PORT
	currentSetup = args.setup if args.setup is not None else Setup.BACK_EDGE.value

	if FRAME_SIZE > 1920:
		print("Frame size is too large!")
		quit()
	elif FRAME_SIZE < 480:
		print("Frame size is too small! You won't see anything")
		quit()

	# Read from video
	IS_CAM = VIDEO_CONFIG["IS_CAM"]
	if args.video:
		git_root = find_git_root(os.path.dirname(os.path.abspath(__file__)))
		if git_root:
			print(f"Git repository root found at: {git_root}")
		else:
			print("No Git repository root found.")
		cap: cv2.VideoCapture = cv2.VideoCapture(os.path.join(git_root, args.video))
	else:
		cap: cv2.VideoCapture = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
	#cap: cv2.VideoCapture = cv2.VideoCapture(0)
	# Check if the camera opened successfully
	if not cap.isOpened():
		print("Error: Could not open camera.")
		exit()
		
	# Load YOLOv3-tiny weights and config
	WEIGHTS_PATH: str = YOLO_CONFIG["WEIGHTS_PATH"]
	CONFIG_PATH: str = YOLO_CONFIG["CONFIG_PATH"]

	# Load the YOLOv3-tiny pre-trained COCO dataset 
	net: cv2.dnn.Net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
	# Set the preferable backend to CPU since we are not using GPU
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	# Get the names of all the layers in the network
	ln: Sequence[str] = net.getLayerNames()
	# Filter out the layer names we dont need for YOLO
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

	# Tracker parameters
	max_cosine_distance: float = 0.7
	nn_budget = None

	#initialize deep sort object
	if IS_CAM: 
		max_age: int = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
	else:
		max_age: int = DATA_RECORD_RATE * TRACK_MAX_AGE
		if max_age > 30:
			max_age = 30
	model_filename: str = MODEL_FILENAME_PATH
	encoder: np.ndarray[np.float32] = gdet.create_box_encoder(model_filename, batch_size=1)
	metric: nn_matching.NearestNeighborDistanceMetric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker: Tracker = Tracker(metric, max_age=max_age)

	if not os.path.exists('processed_data'):
		os.makedirs('processed_data')

	movement_data_file = open('processed_data/movement_data.csv', 'w') 
	crowd_data_file = open('processed_data/crowd_data.csv', 'w')
	# sd_violate_data_file = open('sd_violate_data.csv', 'w')
	# restricted_entry_data_file = open('restricted_entry_data.csv', 'w')
	
	movement_data_writer: csv.writer = csv.writer(movement_data_file)
	crowd_data_writer: csv.writer = csv.writer(crowd_data_file)
	# sd_violate_writer = csv.writer(sd_violate_data_file)
	# restricted_entry_data_writer = csv.writer(restricted_entry_data_file)

	if os.path.getsize('processed_data/movement_data.csv') == 0:
		movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
	if os.path.getsize('processed_data/crowd_data.csv') == 0:
		crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry', 'Abnormal Activity'])


	client_init(cap, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer, host, port, currentSetup)
	cv2.destroyAllWindows()
	movement_data_file.close()
	crowd_data_file.close()


	cap.release()


if __name__ == "__main__":
	main()
