import os
import sys
import multiprocessing
import socket
import psutil
import socket
import ssl
import argparse
script_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)

from config import (
    SERVER_KEY_PATH,
    SERVER_CERT_PATH,
    SERVER_HOST,
	SERVER_PORT
)
from common_tasks.process_video_common_class_decentralized_lb import start_video_processing

from typing import Any, Dict, List, Tuple, Union, Sequence


"""IS_CAM: bool = VIDEO_CONFIG["IS_CAM"]
HIGH_CAM: bool = VIDEO_CONFIG["HIGH_CAM"]
START_TIME: float = time.time()
END_TIME: float = time.time()
PROCESS_TIME: float = time.time()"""
"""HOST: str = "192.168.3.8"
PORT: int = 60000"""

server: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 100000)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server: ssl.SSLSocket = ssl.wrap_socket(
    server, server_side=True, keyfile=SERVER_KEY_PATH, certfile=SERVER_CERT_PATH
)

connection = None 

		
def server_init(host, port):
    global connection  # Use the global connection variable
    
    server.bind((host, port))
    server.listen(0)
    print("SERVER OPEN ON IP", host, " AND PORT ", port)
    process_list: List = []
    
    while True:
        """print("CPU Percent", psutil.cpu_percent())
        print("Memory Percent", psutil.virtual_memory().percent)"""
        try:
            connection, client_address = server.accept()
            print(f"Connection from {client_address}")
            process = multiprocessing.Process(target=start_video_processing, args=(connection, client_address))
            process.daemon = True  # Daemonize the process
            process.start()
            process_list.append(process)
        except KeyboardInterrupt:
            print("Keyboard interrupt. Killing server")
            for p in process_list:
                p.join()
            break
        except Exception as e:
            print(f"Error server: {e}")
            for p in process_list:
                p.join()
            break

def main():
    parser = argparse.ArgumentParser(description='Server for video processing')
    parser.add_argument('--host', type=str, default=None, help='Host IP to bind')
    parser.add_argument('--port', type=int, default=None, help='Port number to listen on')
    args = parser.parse_args()

    host = args.host if args.host is not None else SERVER_HOST
    port = args.port if args.port is not None else SERVER_PORT

    server_init(host, port)
    print("END")


if __name__ == "__main__":
    #multiprocessing.set_start_method('fork')
    main()