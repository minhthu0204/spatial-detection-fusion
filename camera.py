import depthai as dai
import blobconverter
import cv2
import numpy as np
from typing import List
from detection import Detection
import config
import os

class Camera:
    label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = False):
        self.show_video = show_video
        self.show_depth = False
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
        #self.rgb_high_pixel_queue = self.device.getOutputQueue(name="rgb_high_pixel", maxSize=1, blocking=False)

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 360)

        self.frame_rgb = None
        self.frame_depth = None
        self.detected_objects: List[Detection] = []

        self._load_calibration()

        print("=== Connected to " + self.device_info.getMxId())

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

        # RGB cam -> 'rgb_high_pixel' (high resolution video output)
        # xout_rgb_high_pixel = pipeline.createXLinkOut()
        # xout_rgb_high_pixel.setStreamName("rgb_high_pixel")
        # cam_rgb.video.link(xout_rgb_high_pixel.input)  # Link high-resolution video output

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

    # In camera.py
    def update(self):
        in_rgb = self.rgb_queue.tryGet()
        in_nn = self.nn_queue.tryGet()
        in_depth = self.depth_queue.tryGet()

        if in_rgb is None or in_depth is None:
            return None  # Return None explicitly when no new frames

        self.frame_rgb = in_rgb.getCvFrame()
        self.depth_frame = in_depth.getFrame()  # depthFrame values are in millimeters

        detections = []
        if in_nn is not None:
            detections = in_nn.detections

        self.detected_objects = []
        detection_info = []

        for detection in detections:
            try:
                label = self.label_map[detection.label]
            except:
                label = detection.label

            if self.cam_to_world is not None:
                pos_camera_frame = np.array([[detection.spatialCoordinates.x / 1000,
                                              -detection.spatialCoordinates.y / 1000,
                                              detection.spatialCoordinates.z / 1000, 1]]).T
                pos_world_frame = self.cam_to_world @ pos_camera_frame

                self.detected_objects.append(Detection(label, detection.confidence, pos_world_frame, self.friendly_id))

            # Store detection information for client
            detection_info.append({
                'label': label,
                'confidence': detection.confidence,
                'bbox': {
                    'xmin': detection.xmin,
                    'ymin': detection.ymin,
                    'xmax': detection.xmax,
                    'ymax': detection.ymax
                },
                'spatial': {
                    'x': detection.spatialCoordinates.x,
                    'y': detection.spatialCoordinates.y,
                    'z': detection.spatialCoordinates.z
                }
            })

        return {
            'frame_rgb': self.frame_rgb,
            'depth_frame': self.depth_frame,
            'detections': detection_info,
            'friendly_id': self.friendly_id
        }