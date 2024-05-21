from config import (
	MIN_THRESHOLD_CAMSERVER_IMBALANCE,
	MAX_THRESHOLD_CAMSERVER_IMBALANCE,
	MAX_SIZE_LISTEN_LIST
)

class VideoProcessor:
    def __init__(self, connection, client_address):
        self.frames = []
        self.connection = connection
        self.client_address = client_address
        self.imbalances30 = []
        self.imbalances60 = []
        self.imbalances120 = []
        #self.load_dependencies()

    def appendImbalanceIndex(self, imbalanceIndex: float):
        self.imbalances30.append(imbalanceIndex)
        if len(self.imbalances30) > 30:
            self.imbalances30.pop(0)
        self.imbalances60.append(imbalanceIndex)
        if len(self.imbalances60) > 60:
            self.imbalances60.pop(0)
        self.imbalances120.append(imbalanceIndex)
        if len(self.imbalances120) > 120:
            self.imbalances120.pop(0)

    def record_movement_data(self, movement_data_writer, movement):
        import numpy as np
        track_id = movement.track_id 
        entry_time = movement.entry 
        exit_time = movement.exit			
        positions = movement.positions
        positions = np.array(positions).flatten()
        positions = list(positions)
        data = [track_id] + [entry_time] + [exit_time] + positions
        movement_data_writer.writerow(data)

    def record_crowd_data(self, time, human_count, violate_count, restricted_entry, abnormal_activity, crowd_data_writer):
        data = [time, human_count, violate_count, int(restricted_entry), int(abnormal_activity)]
        crowd_data_writer.writerow(data)

    def end_video(self, tracker, frame_count, movement_data_writer):
        for t in tracker.tracks:
            if t.is_confirmed():
                t.exit = frame_count
                self.record_movement_data(movement_data_writer, t)

    def create_video(self, frames):
        import cv2
        import os
        fps: int = 30
        if frames:
            output_filename: str = "output_video_id" + str(os.getpid()) + ".mp4"
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)

            out.release()
            print(f"Video '{output_filename}' created successfully.")
            return len(frames)
        else:
            print("No frames")

    def process_video(self):
        import debugpy
        import os
        import sys
        import time
        import datetime
        import multiprocessing
        import socket
        import numpy as np
        import imutils
        import psutil
        import cv2
        import socket
        import ssl
        from math import ceil
        from scipy.spatial.distance import euclidean
        #import pprint

        script_directory = os.path.dirname(__file__)
        parent_directory = os.path.dirname(script_directory)
        sys.path.append(parent_directory)

        #pprint.pprint(sys.path)


        from util import rect_distance, progress, kinetic_energy
        from colors import RGB_COLORS
        from config import (
            Setup,
            SERVER_KEY_PATH,
            SERVER_CERT_PATH,
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
        )
        from deep_sort import nn_matching
        from deep_sort.detection import Detection
        from deep_sort.tracker import Tracker
        from deep_sort import generate_detections as gdet
        from common_tasks.common_tasks import remove_background, detect_human
        import csv
        import pickle
        import json
        import tensorflow as tf
        import cvzone
        from cvzone.SelfiSegmentationModule import SelfiSegmentation
        from typing import Any, Dict, List, Tuple, Union, Sequence
        from enum import Enum
        
        def calculate_FPS():
            t1 = time.time() - t0
            VID_FPS = frame_count / t1

        IS_CAM: bool = VIDEO_CONFIG["IS_CAM"]
        HIGH_CAM: bool = VIDEO_CONFIG["HIGH_CAM"]
        START_TIME: float = time.time()
        END_TIME: float = time.time()
        PROCESS_TIME: float = time.time()

        if FRAME_SIZE > 1920:
            print("Frame size is too large!")
            quit()
        elif FRAME_SIZE < 480:
            print("Frame size is too small! You won't see anything")
            quit()
    
        # Load YOLOv3-tiny weights and config
        WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
        CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]

        # Load the YOLOv3-tiny pre-trained COCO dataset 
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        # Set the preferable backend to CPU since we are not using GPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the names of all the layers in the network
        ln = net.getLayerNames()
        # Filter out the layer names we dont need for YOLO
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # Tracker parameters
        max_cosine_distance = 0.7
        nn_budget = None

        #initialize deep sort object
        if IS_CAM: 
            max_age = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
        else:
            max_age = DATA_RECORD_RATE * TRACK_MAX_AGE
            if max_age > 30:
                max_age = 30
        model_filename = MODEL_FILENAME_PATH
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric, max_age=max_age)

        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')

        movement_data_string: str = 'processed_data/movement_data' + str(os.getpid()) + '.csv'
        crowd_data_string: str = 'processed_data/crowd_data' + str(os.getpid()) + '.csv'

        movement_data_file = open(movement_data_string, 'w') 
        crowd_data_file = open(crowd_data_string, 'w')

        movement_data_writer = csv.writer(movement_data_file)
        crowd_data_writer = csv.writer(crowd_data_file)

        if os.path.getsize(movement_data_string) == 0:
            movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
        if os.path.getsize(crowd_data_string) == 0:
            crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry', 'Abnormal Activity'])

        VID_FPS = None
        DATA_RECORD_FRAME = 1
        TIME_STEP = 1
        t0 = time.time()

        frame_count = 0
        display_frame_count = 0
        re_warning_timeout = 0
        sd_warning_timeout = 0
        ab_warning_timeout = 0

        RE = False
        ABNORMAL = False
        try:
            START_TIME = time.time()
            while True:
                ITERATE_START_TIME = time.time()
                data = b""
                while True:
                    # Receive a chunk of data
                    try:
                        chunk = self.connection.recv(4096)
                        data += chunk
                    except Exception as e:
                        data = b"QUIT"
                    # Check if the end of the frame is reached
                    if data[-2:] == b"\xff\xd9":
                        break
                    if data == b"QUIT":
                        print("Termination signal received")
                        self.end_video(tracker, frame_count, movement_data_writer)
                        break
                if data == b"QUIT":
                    break    
                # Convert the received bytes to a NumPy array (frame)
                try:
                    # Deserialize the received data
                    data = data[:-2]
                    received_data = pickle.loads(data)
                    #frame_bytes = received_data["frame"]
                    setup:int = received_data["setup"]
                    frame = cv2.imdecode(np.frombuffer(received_data["frame"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    #cv2.imshow("Server Frame", frame)
                    iterateTimeClient = received_data["processingTime"]
                    #print("Setup: ", setup)
                    if setup==Setup.HALF_EDGE_SERVER.value:
                        try:
                            humans_detected = received_data["humans_detected"]
                            expired = received_data["expired"]
                        except Exception as e:
                            print(f"Error decoding humans_detected: {e}")
                            break
                    if setup == Setup.ALL_EDGE.value:
                        self.frames.append(frame)
                        ITERATE_TIME = time.time() - ITERATE_START_TIME
                        imbalanceIndex: float = float(ITERATE_TIME/iterateTimeClient)
                        data_to_send: Dict[float] = {
                            "imbalanceIndex": imbalanceIndex
                        }
                        data_bytes: bytes = pickle.dumps(data_to_send)
                        new_bytes: bytes = bytes.fromhex('ffd9')
                        data_bytes += new_bytes
                        self.connection.sendall(data_bytes)
                        self.appendImbalanceIndex(imbalanceIndex)
                        continue
                    if setup == Setup.BACK_SERVER.value:
                        new_frame = remove_background(frame)
                        self.frames.append(new_frame)
                        ITERATE_TIME = time.time() - ITERATE_START_TIME
                        imbalanceIndex: float = float(ITERATE_TIME/iterateTimeClient)
                        data_to_send: Dict[float] = {
                            "imbalanceIndex": imbalanceIndex
                        }
                        data_bytes: bytes = pickle.dumps(data_to_send)
                        new_bytes: bytes = bytes.fromhex('ffd9')
                        data_bytes += new_bytes
                        self.connection.sendall(data_bytes)
                        self.appendImbalanceIndex(imbalanceIndex)
                        continue
                    if setup == Setup.BACK_EDGE.value:
                        self.frames.append(frame)
                        ITERATE_TIME = time.time() - ITERATE_START_TIME
                        imbalanceIndex: float = float(ITERATE_TIME/iterateTimeClient)
                        data_to_send: Dict[float] = {
                            "imbalanceIndex": imbalanceIndex
                        }
                        data_bytes: bytes = pickle.dumps(data_to_send)
                        new_bytes: bytes = bytes.fromhex('ffd9')
                        data_bytes += new_bytes
                        self.connection.sendall(data_bytes)
                        self.appendImbalanceIndex(imbalanceIndex)
                        continue
                except Exception as e:
                    print(f"Error decoding frame: {e}")
                    continue

                ret = frame is not None and frame.size > 0
                # Stop the loop when video ends
                if not ret:
                    self.end_video(tracker, frame_count, movement_data_writer)
                    if not VID_FPS:
                        self.calculate_FPS()
                    break
                #frames.append(frame)

                # Update frame count
                if frame_count > 1000000:
                    if not VID_FPS:
                        self.calculate_FPS()
                    frame_count = 0
                    display_frame_count = 0

                # Skip frames according to the given rate
                if frame_count % DATA_RECORD_FRAME != 0:
                    continue

                display_frame_count += 1

                frame = imutils.resize(frame, width=FRAME_SIZE)

                # Get current time
                current_datetime = datetime.datetime.now()

                # Run detection algorithm
                if IS_CAM:
                    record_time = current_datetime
                else:
                    record_time = frame_count

                # Run tracking algorithm
                if setup == Setup.ALL_SERVER.value:
                    [humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, record_time)

                # Record movement data
                for movement in expired:
                    self.record_movement_data(movement_data_writer, movement)

                # Check for restricted entry
                if RE_CHECK:
                    RE = False
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

                # Display current time on screen
                # current_date = str(current_datetime.strftime("%b-%d-%Y"))
                # current_time = str(current_datetime.strftime("%I:%M:%S %p"))
                # cv2.putText(frame, (current_date), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                # cv2.putText(frame, (current_time), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                # Record crowd data to file
                if DATA_RECORD:
                    self.record_crowd_data(record_time, len(humans_detected), len(violate_set), RE, ABNORMAL, crowd_data_writer)

                # Display video output or processing indicator
                if SHOW_PROCESSING_OUTPUT:
                    self.frames.append(frame)
                    ITERATE_TIME = time.time() - ITERATE_START_TIME
                    imbalanceIndex = float(ITERATE_TIME/iterateTimeClient)
                    data_to_send: Dict[float] = {
                        "imbalanceIndex": imbalanceIndex
                    }
                    data_bytes: bytes = pickle.dumps(data_to_send)
                    new_bytes: bytes = bytes.fromhex('ffd9')
                    data_bytes += new_bytes
                    self.connection.sendall(data_bytes)
                    self.appendImbalanceIndex(imbalanceIndex)
                else:
                    progress(display_frame_count)

                # Press 'Q' to stop the video display
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Record the movement when video ends
                    self.end_video(tracker, frame_count, movement_data_writer)
                    # Compute the processing speed
                    if not VID_FPS:
                        self.calculate_FPS()
                    break
        except KeyboardInterrupt:
            print("Keyboard interrupt. Processing frames and creating video.")
        except Exception as e:
            print(f"Error: {e}")
        if self.connection is not None:
            self.connection.sendall(b"QUIT")
            self.connection.close()
        END_TIME = time.time()
        PROCESS_TIME = END_TIME - START_TIME
        print("Time elapsed: ", PROCESS_TIME)
        frame_count = self.create_video(self.frames)
        print("Processed Frames per second: ", float(frame_count/PROCESS_TIME))
        testSum: int = 0
        for x in self.imbalances30:
            testSum += x
        print("Imbalances30 Average = ", str(float(testSum/len(self.imbalances30))))
        testSum = 0
        for x in self.imbalances60:
            testSum += x
        print("Imbalances60 Average = ", str(float(testSum/len(self.imbalances60))))
        testSum = 0
        for x in self.imbalances120:
            testSum += x
        print("Imbalances120 Average = ", str(float(testSum/len(self.imbalances120))))
        cv2.destroyAllWindows()
        movement_data_file.close()
        crowd_data_file.close()

    def run(self):
        try:
            print("Run")
            self.process_video()
        except KeyboardInterrupt:
            print("Keyboard interrupt. Processing frames and creating video.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.connection is not None:
                self.connection.close()




def start_video_processing(connection, client_address):
    # Instantiate the VideoProcessor class with the provided arguments
    processor = VideoProcessor(connection, client_address)
    # Run the video processing logic
    processor.run()
