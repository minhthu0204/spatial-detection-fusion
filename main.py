import socket
import pickle
import struct
import cv2
import depthai as dai
from camera import Camera
from birdseyeview import BirdsEyeView
import config
from typing import List
import threading


class Server:
    def __init__(self, host=config.SERVER_HOST, port=config.SERVER_PORT):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(10)
        print(f"Server listening on {host}:{port}")

        # Khởi tạo cameras
        self.cameras = self.initialize_cameras()
        self.birds_eye_view = BirdsEyeView(self.cameras, config.size[0], config.size[1], config.scale)

        self.running = True

    def initialize_cameras(self) -> List[Camera]:
        device_infos = dai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No devices found!")
        print("Found", len(device_infos), "devices")

        device_infos.sort(key=lambda x: x.getMxId(), reverse=True)
        return [Camera(device_info, i + 1) for i, device_info in enumerate(device_infos)]

    def send_data(self, client_socket, data):
        try:
            serialized_data = pickle.dumps(data)
            message_size = struct.pack("L", len(serialized_data))
            client_socket.sendall(message_size + serialized_data)
        except Exception as e:
            print(f"Error sending data: {e}")
            return False
        return True

    def handle_client(self, client_socket, addr):
        print(f"New connection from {addr}")

        try:
            while self.running:
                # Update all cameras
                all_data = {
                    'cameras_data': [],
                    'birds_eye_view': None
                }

                for camera in self.cameras:
                    camera_data = camera.update()
                    if camera_data:
                        all_data['cameras_data'].append(camera_data)

                # Update birds eye view
                self.birds_eye_view.render()
                all_data['birds_eye_view'] = self.birds_eye_view.img

                # Send data to client
                if not self.send_data(client_socket, all_data):
                    break

        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
            print(f"Connection closed with {addr}")

    def run(self):
        try:
            while self.running:
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, addr)
                )
                client_thread.start()
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.running = False
            self.server_socket.close()


if __name__ == "__main__":
    server = Server()
    server.run()