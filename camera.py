from cgitb import enable

import depthai as dai
import blobconverter
import cv2
import numpy as np
from typing import List, Optional
from detection import Detection
import config
import os
import socket
import pickle
import threading


class CameraTCPServer:
    def __init__(self, host: str = 'localhost', port: int = 8485):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.clients: List[socket.socket] = []
        self.running = True
        # Start server thread
        self.server_thread = threading.Thread(target=self.accept_connections)
        self.server_thread.daemon = True
        self.server_thread.start()

    def accept_connections(self):
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                self.clients.append(client_socket)
                print(f"New client connected. Total clients: {len(self.clients)}")
            except:
                break

    def send_frame(self, frame):
        # Encode frame
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        data = pickle.dumps(encoded_frame)
        message_size = len(data)

        # Remove disconnected clients
        disconnected = []
        for client in self.clients:
            try:
                # Send message size first
                client.sendall(message_size.to_bytes(4, byteorder='big'))
                # Then send the frame data
                client.sendall(data)
            except:
                disconnected.append(client)

        for client in disconnected:
            self.clients.remove(client)
            client.close()
            print(f"Client disconnected. Remaining clients: {len(self.clients)}")

    def cleanup(self):
        self.running = False
        for client in self.clients:
            client.close()
        self.server_socket.close()


class Camera:
    label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = True, enable_tcp: bool = True):
        self.show_video = show_video
        self.show_detph = False
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.still_queue = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        self.control_queue = self.device.getInputQueue(name="control")
        self.nn_queue = self.device.getOutputQueue(name="nn", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 360)

        # Initialize TCP server if enabled
        self.tcp_server: Optional[CameraTCPServer] = None
        if enable_tcp:
            self.tcp_server = CameraTCPServer(port=8485 + friendly_id)


        self.frame_rgb = None
        self.detected_objects: List[Detection] = []

        self._load_calibration()

        print("=== Connected to " + self.device_info.getMxId())
        if enable_tcp:
            print(f"=== TCP server started on port {8485 + friendly_id}")

    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())

    def _load_calibration(self):
        path = os.path.join(os.path.dirname(__file__), f"{config.calibration_data_dir}")
        try:
            extrinsics = np.load(f"{path}/extrinsics_{self.mxid}.npz")
            self.cam_to_world = extrinsics["cam_to_world"]
        except:
            raise RuntimeError(f"Could not load calibration data for camera {self.mxid} from {path}!")

    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")

        # Depth cam -> 'depth'
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_stereo = pipeline.create(dai.node.StereoDepth)
        cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth map to the perspective of RGB camera, on which inference is done
        cam_stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        # Spatial detection network -> 'nn'
        spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        spatial_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
        spatial_nn.setConfidenceThreshold(0.6)
        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.2)
        spatial_nn.setDepthLowerThreshold(100)
        spatial_nn.setDepthUpperThreshold(5000)
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        cam_rgb.preview.link(spatial_nn.input)
        # cam_rgb.preview.link(xout_rgb.input)
        cam_stereo.depth.link(spatial_nn.inputDepth)
        spatial_nn.passthrough.link(xout_rgb.input)
        spatial_nn.passthroughDepth.link(xout_depth.input)
        spatial_nn.out.link(xout_nn.input)


        # Still encoder -> 'still'
        still_encoder = pipeline.create(dai.node.VideoEncoder)
        still_encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
        cam_rgb.still.link(still_encoder.input)
        xout_still = pipeline.createXLinkOut()
        xout_still.setStreamName("still")
        still_encoder.bitstream.link(xout_still.input)

        # Camera control -> 'control'
        control = pipeline.create(dai.node.XLinkIn)
        control.setStreamName('control')
        control.out.link(cam_rgb.inputControl)

        self.pipeline = pipeline

    def update(self):
        in_rgb = self.rgb_queue.tryGet()
        in_nn = self.nn_queue.tryGet()
        in_depth = self.depth_queue.tryGet()

        if in_rgb is None or in_depth is None:
            return

        depth_frame = in_depth.getFrame() # depthFrame values are in millimeters
        depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)
        depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

        self.frame_rgb = in_rgb.getCvFrame()

        if self.show_detph:
            visualization = depth_frame_color.copy()
        else:
            visualization = self.frame_rgb.copy()
        visualization = cv2.resize(visualization, (640, 360), interpolation = cv2.INTER_NEAREST)

        height = visualization.shape[0]
        width  = visualization.shape[1]

        detections = []
        if in_nn is not None:
            detections = in_nn.detections

        self.detected_objects = []

        for detection in detections:
            roi = detection.boundingBoxMapping.roi
            roi = roi.denormalize(width, height)
            top_left = roi.topLeft()
            bottom_right = roi.bottomRight()
            xmin = int(top_left.x)
            ymin = int(top_left.y)
            xmax = int(bottom_right.x)
            ymax = int(bottom_right.y)

            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = self.label_map[detection.label]
            except:
                label = detection.label

            if self.cam_to_world is not None:
                pos_camera_frame = np.array([[detection.spatialCoordinates.x / 1000, -detection.spatialCoordinates.y / 1000, detection.spatialCoordinates.z / 1000, 1]]).T
                # pos_camera_frame = np.array([[0, 0, detection.spatialCoordinates.z/1000, 1]]).T
                pos_world_frame = self.cam_to_world @ pos_camera_frame

                self.detected_objects.append(Detection(label, detection.confidence, pos_world_frame, self.friendly_id))

            cv2.rectangle(visualization, (xmin, ymin), (xmax, ymax), (100, 0, 0), 2)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(visualization, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # Send frame through TCP if server is enabled
        if self.tcp_server:
            self.tcp_server.send_frame(visualization)
        if self.show_video:
            cv2.imshow(self.window_name, visualization)