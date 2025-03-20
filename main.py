import cv2
import numpy as np
from rknn.api import RKNNLite
from norfair import Detection, Tracker, draw_tracked_objects
from filterpy.kalman import KalmanFilter

# Load RKNN model for YOLO
rknn = RKNNLite()
rknn.load_rknn("yolov8n.rknn")

# Stereo Camera Parameters
focal_length = 700  # Pixels
baseline = 0.1  # Meters
cx, cy = 320, 240  # Image center

# Enable OpenCL for depth processing
cv2.ocl.setUseOpenCL(True)

# OpenCL-accelerated StereoSGBM
stereo = cv2.StereoSGBM_create(
    numDisparities=96,
    blockSize=15,
    P1=8 * 3 * 3**2,
    P2=32 * 3 * 3**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# Separate trackers for two object types
tracker_type1 = Tracker(distance_function="euclidean", distance_threshold=30)
tracker_type2 = Tracker(distance_function="euclidean", distance_threshold=30)

# Kalman Filters for smoothing depth
kf_type1 = KalmanFilter(dim_x=4, dim_z=2)
kf_type1.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf_type1.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf_type1.P *= 1000  # Initial uncertainty
kf_type1.R = np.array([[5, 0], [0, 5]])

kf_type2 = KalmanFilter(dim_x=4, dim_z=2)
kf_type2.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf_type2.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf_type2.P *= 1000
kf_type2.R = np.array([[5, 0], [0, 5]])

# Open cameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

while cap_left.isOpened() and cap_right.isOpened():
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not (retL and retR):
        break

    # Convert to grayscale for stereo depth
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Compute disparity map using OpenCL-accelerated SGBM
    disparity = stereo.compute(grayL, grayR)
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Preprocess for YOLO RKNN
    input_data = cv2.resize(frameL, (640, 640)).astype(np.uint8)

    # Run YOLO inference on RKNN
    outputs = rknn.inference(inputs=[input_data])

    # Separate detections for two object types
    detections_type1 = []
    detections_type2 = []

    for det in outputs[0]:  # Assuming first output contains boxes
        x1, y1, x2, y2, conf, class_id = det
        if conf < 0.5:
            continue  # Skip low-confidence detections
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Get depth from disparity map
        disparity_value = disparity[center_y, center_x]
        if disparity_value > 0:
            Z = (focal_length * baseline) / disparity_value
            X = ((center_x - cx) * Z) / focal_length
            Y = ((center_y - cy) * Z) / focal_length

            # Kalman filtering
            if class_id == 0:  # Non-spherical object
                kf_type1.predict()
                kf_type1.update([Z, X])
                Z, X = kf_type1.x[:2]
                detections_type1.append(Detection(points=np.array([X, Y, Z])))

                # Compute bounding box aspect ratio for orientation (yaw estimation)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height
                orientation_yaw = np.arctan(aspect_ratio)  # Yaw angle from aspect ratio

                # Estimate pitch and roll based on the object's bounding box orientation
                box_angle = np.arctan2(
                    y2 - y1, x2 - x1
                )  # Angle relative to the image plane
                pitch = np.degrees(box_angle)  # Assuming the object is upright
                roll = np.degrees(
                    np.arctan(height / width)
                )  # Approximate roll based on aspect ratio

                # Visualize orientation
                cv2.putText(
                    frameL,
                    f"Yaw: {np.degrees(orientation_yaw):.2f}°",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frameL,
                    f"Pitch: {pitch:.2f}°",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frameL,
                    f"Roll: {roll:.2f}°",
                    (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                color = (255, 0, 0)  # Blue for Non-Spherical
            else:  # Algae (Spherical object)
                kf_type2.predict()
                kf_type2.update([Z, X])
                Z, X = kf_type2.x[:2]
                detections_type2.append(Detection(points=np.array([X, Y, Z])))
                color = (0, 255, 0)  # Green for Spherical

            # Draw bounding boxes & depth info
            cv2.rectangle(frameL, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frameL,
                f"Z={Z:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Update separate trackers
    tracked_objects_type1 = tracker_type1.update(detections=detections_type1)
    tracked_objects_type2 = tracker_type2.update(detections=detections_type2)

    # Draw tracked objects with different colors
    draw_tracked_objects(frameL, tracked_objects_type1, color=(255, 0, 0))
    draw_tracked_objects(frameL, tracked_objects_type2, color=(0, 255, 0))

    # Show frames
    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Depth Map", disparity)

    # Break loop on key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
rknn.release()
