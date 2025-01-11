import depthai as dai
import blobconverter
import cv2
import numpy as np
from typing import List, Dict
from detection import Detection
import config
import os

class Camera:
    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.nn_queue = self.device.getOutputQueue(name="nn", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        self.frame_rgb = None
        self.detected_objects: List[Detection] = []
        self.cam_to_world = None

        self._load_calibration()
        print(f"=== Connected to {self.device_info.getMxId()}")

    def _load_calibration(self):
        path = os.path.join(os.path.dirname(__file__), f"{config.calibration_data_dir}")
        try:
            extrinsics = np.load(f"{path}/extrinsics_{self.mxid}.npz")
            self.cam_to_world = extrinsics["cam_to_world"]
        except:
            raise RuntimeError(f"Could not load calibration data for camera {self.mxid} from {path}!")

    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB cam
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)

        # Depth cam
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        cam_stereo = pipeline.create(dai.node.StereoDepth)
        cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        cam_stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        # Neural Network
        spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        spatial_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
        spatial_nn.setConfidenceThreshold(0.6)
        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.2)
        spatial_nn.setDepthLowerThreshold(100)
        spatial_nn.setDepthUpperThreshold(5000)

        # Linking
        cam_rgb.preview.link(spatial_nn.input)
        cam_stereo.depth.link(spatial_nn.inputDepth)

        # Create outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        spatial_nn.passthrough.link(xout_rgb.input)

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        spatial_nn.out.link(xout_nn.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        spatial_nn.passthroughDepth.link(xout_depth.input)

        self.pipeline = pipeline

    def update(self) -> Dict:
        in_rgb = self.rgb_queue.tryGet()
        in_nn = self.nn_queue.tryGet()
        in_depth = self.depth_queue.tryGet()

        if in_rgb is None or in_depth is None:
            return None

        frame = in_rgb.getCvFrame()
        height = frame.shape[0]
        width = frame.shape[1]

        detections = []
        detection_data = []

        if in_nn is not None:
            # Chỉ lấy person detections (label 15)
            detections = [det for det in in_nn.detections if det.label == 15]

            for detection in detections:
                if self.cam_to_world is not None:
                    pos_camera_frame = np.array([[
                        detection.spatialCoordinates.x / 1000,
                        -detection.spatialCoordinates.y / 1000,
                        detection.spatialCoordinates.z / 1000,
                        1
                    ]]).T
                    pos_world_frame = self.cam_to_world @ pos_camera_frame

                    self.detected_objects.append(Detection(
                        "person",
                        detection.confidence,
                        pos_world_frame,
                        self.friendly_id
                    ))

                detection_data.append({
                    'confidence': detection.confidence,
                    'xmin': int(detection.xmin * width),
                    'ymin': int(detection.ymin * height),
                    'xmax': int(detection.xmax * width),
                    'ymax': int(detection.ymax * height),
                    'spatial_coords': {
                        'x': int(detection.spatialCoordinates.x),
                        'y': int(detection.spatialCoordinates.y),
                        'z': int(detection.spatialCoordinates.z)
                    }
                })

        return {
            'frame': frame,
            'detections': detection_data,
            'detected_objects': self.detected_objects
        }

    def __del__(self):
        self.device.close()
        print(f"=== Closed {self.device_info.getMxId()}")