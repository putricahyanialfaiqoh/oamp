import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CONTOUR_GROUPS = (
        ((10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109), (0, 255, 0)),
        ((70, 63, 105, 66, 107, 55, 65, 52, 53, 46), (0, 255, 0)),
        ((336, 296, 334, 293, 300, 285, 295, 282, 283, 276), (0, 255, 0)),
        ((33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
            159, 160, 161, 246), (0, 0, 255)),
        ((263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385,
            386, 387, 388, 466), (0, 0, 255)),
        ((61, 40, 37, 0, 267, 270, 409, 291, 321, 375, 308, 324, 318,
            402, 317, 14, 87, 178, 88, 95, 185, 40, 39), (0, 0, 255)),
)


def draw_mesh_overlay(frame, points):
        height, width, _ = frame.shape
        for x, y in points:
                if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

        for indices, color in CONTOUR_GROUPS:
                contour = [points[index] for index in indices if index < len(points)]
                if len(contour) >= 2:
                        cv2.polylines(frame, [np.asarray(contour, dtype=np.int32)], False, color, 1)

        valid_points = [point for point in points if 0 <= point[0] < width and 0 <= point[1] < height]
        if valid_points:
                x_values, y_values = zip(*valid_points)
                cv2.rectangle(frame, (min(x_values), min(y_values)), (max(x_values), max(y_values)), (0, 0, 255), 1)

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        model_path = Path(__file__).parent / "facemape" / "face_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(f"Model landmark tidak ditemukan: {model_path}")

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(image)

        if not result.face_landmarks:
            return False, None

        landmarks = result.face_landmarks[0]
        height, width, _ = frame.shape
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]

        xmin = max(0, min(x for x, _ in points))
        ymin = max(0, min(y for _, y in points))
        xmax = min(width, max(x for x, _ in points) + 1)
        ymax = min(height, max(y for _, y in points) + 1)

        face_width = xmax - xmin
        face_height = ymax - ymin
        padding_x = int(face_width * 0.2)
        padding_y = int(face_height * 0.2)
        xmin = max(0, xmin - padding_x)
        ymin = max(0, ymin - padding_y)
        xmax = min(width, xmax + padding_x)
        ymax = min(height, ymax + padding_y)
        face_crop = frame[ymin:ymax, xmin:xmax].copy()

        draw_mesh_overlay(frame, points)

        return True, face_crop