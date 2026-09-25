import time
import numpy as np
import cv2
import pyzed.sl as sl

from your_module import (
    Detection, CameraIntrinsics, SafetyConfig,
    detections_from_result, choose_detection,
    median_depth, unproject_pixel, transform_point, safety_check,
    _load_model,
)

# fixed transform: measure or get from robot spec/CAD
CAMERA_TO_BODY = np.array([
    [0, 0, 1, X_OFFSET],
    [-1, 0, 0, Y_OFFSET],
    [0, -1, 0, Z_OFFSET],
    [0, 0, 0, 1],
], dtype=np.float64)

def main():
    model = _load_model("Kota0612/okra11n-seg-v5", "output/okra_finetune_v5/weights/best.pt")

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    intrinsics = CameraIntrinsics(fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy)

    image = sl.Mat()
    depth = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    while True:
        t0 = time.time()
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # meters, HxW array

        frame = image.get_data()[:, :, :3]
        depth_m = depth.get_data()

        results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
        detections = [d for r in results for d in detections_from_result(r)]
        selected = choose_detection(detections, minimum_confidence=0.5)
        if selected is None:
            print("no detection")
            continue

        d = median_depth(depth_m, selected.centroid_u, selected.centroid_v, radius_px=3)
        if d is None:
            print("no valid depth at centroid")
            continue

        point_cam = unproject_pixel(selected.centroid_u, selected.centroid_v, d, intrinsics)
        point_body = transform_point(point_cam, CAMERA_TO_BODY)

        ok, reason = safety_check(
            selected, point_body,
            frame_age_s=time.time() - t0,
            config=SafetyConfig(min_confidence=0.6, max_distance_m=1.0),
        )
        print(f"conf={selected.confidence:.2f} point_body={point_body} ok={ok} ({reason})")

        # STOP HERE for now — no motion, just observe

if __name__ == "__main__":
    main()