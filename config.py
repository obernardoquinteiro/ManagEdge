import os
import sys
import datetime
from enum import Enum

from typing import Any, Dict

class Setup(Enum):
	ALL_EDGE = 0
	HALF_EDGE_SERVER = 1
	ALL_SERVER = 2
	BACK_SERVER = 4
	BACK_EDGE = 5

SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 60001
CLIENT_HOST: str = "0.0.0.0"
CLIENT_PORT: int = 60001
#Threshold below which the edge nodes are overloaded compared to the server
MIN_THRESHOLD_CENTRAL_IMBALANCE: float = 0.7
#Threshold above which the server is overloaded compared to the edge nodes
MAX_THRESHOLD_CENTRAL_IMBALANCE: float = 1.3
#Threshold below which the camera is overloaded compared to the server
MIN_THRESHOLD_CAMSERVER_IMBALANCE: float = 0.5
#Threshold above which the server is overloaded compared to the camera
MAX_THRESHOLD_CAMSERVER_IMBALANCE: float = 1.5
#Maximum size of listenList: list that stores the last X workload imbalance indexes recorded 
MAX_SIZE_LISTEN_LIST: int = 10
# Show individuals detected
SHOW_PROCESSING_OUTPUT: bool = True
# Show individuals detected
SHOW_DETECT: bool = True
# Data record
DATA_RECORD: bool = True
# Data record rate (data record per frame)
DATA_RECORD_RATE: int = 5
# Check for restricted entry
RE_CHECK: bool = False
# Restricted entry time (H:M:S)
RE_START_TIME: datetime.time = datetime.time(0,0,0) 
RE_END_TIME: datetime.time = datetime.time(23,0,0)
# Check for social distance violation
SD_CHECK: bool = False
# Show violation count
SHOW_VIOLATION_COUNT: bool = False
# Show tracking id
SHOW_TRACKING_ID: bool = False
# Threshold for distance violation
SOCIAL_DISTANCE: int = 50
# Check for abnormal crowd activity
ABNORMAL_CHECK: bool = True
# Min number of people to check for abnormal
ABNORMAL_MIN_PEOPLE: int = 5
# Abnormal energy level threshold
ABNORMAL_ENERGY: int = 1866
# Abnormal activity ratio threhold
ABNORMAL_THRESH: float = 0.66
# Threshold for human detection minumun confindence
MIN_CONF: float = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH: float = 0.2
# Resize frame for processing
FRAME_SIZE: int = 1080
# Tracker max missing age before removing (seconds)
TRACK_MAX_AGE: int = 3

def find_git_root(start_directory):
    """
    Traverse upwards from 'start_directory' until a directory containing
    a '.git' directory is found, indicating the root of a Git repository.

    Args:
    start_directory (str): The starting directory path.

    Returns:
    str: The path to the root of the Git repository, or None if not found.
    """
    current_directory = start_directory

    # Keep moving up in the directory structure until reaching the system root
    while current_directory != os.path.dirname(current_directory):
        # Check if this directory contains a .git folder
        if os.path.isdir(os.path.join(current_directory, '.git')):
            return current_directory
        # Move up one directory level
        current_directory = os.path.dirname(current_directory)

    # If we reach here, no Git root has been found
    return None


# Start the search from the directory where the script is running
git_root = find_git_root(os.path.dirname(os.path.abspath(__file__)))
if git_root:
     print(f"Git repository root found at: {git_root}")
else:
      print("No Git repository root found.")
      sys.exit(1)
SERVER_KEY_PATH: str = os.path.join(git_root, "certificates", "server-key.pem")
SERVER_CERT_PATH: str = os.path.join(git_root, "certificates", "server-cert.pem")
CLIENT_KEY_PATH: str = os.path.join(git_root, "certificates", "client-key.pem")
CLIENT_CERT_PATH: str = os.path.join(git_root, "certificates", "client-cert.pem")
# Load YOLOv3-tiny weights and config
YOLO_CONFIG: Dict[str, str] = {
     "WEIGHTS_PATH" : os.path.join(git_root, "YOLOv4-tiny", "yolov4-tiny.weights"),
      "CONFIG_PATH" : os.path.join(git_root, "YOLOv4-tiny", "yolov4-tiny.cfg"),
}
# Video Path
VIDEO_CONFIG: Dict[str, Any] = {
	"VIDEO_CAP" : os.path.join(git_root, "full.webm"),
	"IS_CAM" : False,
	"CAM_APPROX_FPS": 3,
	"HIGH_CAM": False,
	"START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}
MODEL_FILENAME_PATH: str = os.path.join(git_root, "model_data", "mars-small128.pb")
