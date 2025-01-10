import socket
import cv2
import numpy as np
import pickle
import struct
from typing import List
import depthai as dai
from camera import Camera
from birdseyeview import BirdsEyeView
import config


class VideoServer:
    def __init__(self, host='192.168.1.7', port=9999):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

    def start(self):
        # Initialize cameras
        device_infos = dai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No devices found!")
        print("Found", len(device_infos), "devices")

        device_infos.sort(key=lambda x: x.getMxId(), reverse=True)
        cameras: List[Camera] = []

        friendly_id = 0
        for device_info in device_infos:
            friendly_id += 1
            # Set show_video to False since we don't want local display
            cameras.append(Camera(device_info, friendly_id, show_video=False))

        birds_eye_view = BirdsEyeView(cameras, config.size[0], config.size[1], config.scale)

        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr}")

            try:
                while True:
                    # Update all cameras
                    for camera in cameras:
                        camera.update()
                        if camera.frame_rgb is not None:
                            # Encode camera frame
                            _, frame_data = cv2.imencode('.jpg', camera.frame_rgb)
                            camera_data = pickle.dumps({
                                'type': 'camera',
                                'id': camera.friendly_id,
                                'frame': frame_data
                            })
                            # Send camera frame size and data
                            message_size = struct.pack("L", len(camera_data))
                            client_socket.sendall(message_size + camera_data)

                    # Update and send birds eye view
                    birds_eye_view.render()
                    _, birds_eye_data = cv2.imencode('.jpg', birds_eye_view.img)
                    view_data = pickle.dumps({
                        'type': 'birds_eye',
                        'frame': birds_eye_data
                    })
                    message_size = struct.pack("L", len(view_data))
                    client_socket.sendall(message_size + view_data)

            except (ConnectionResetError, BrokenPipeError):
                print(f"Client {addr} disconnected")
                client_socket.close()


if __name__ == "__main__":
    server = VideoServer()
    server.start()