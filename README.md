# Multi-Camera Fusion for Pingpong

0. Run extrinsic calibration
   
    a. put intrinsic calibration (yaml) in `config/camera/`

    b. in `tests/camera_calibration_tests/test_camera_pose.py` run `test_camera_pose_estimation()`. This will write the extrinsics in the original yaml file

1. Run ball detection

    a. put `obj.data`, `obj.names`, `yolo.weights`, `yolo.cfg` in `config/darknet/`

    b. run `test_all_images()` in `tests/yolo_tests/test_yolo_detection.py`. This will do detection through all the images in `data/images/**/*.jpg`. The detection result will append to the annotations in `json`. Meanwhile, a `data/debug/` folder will be created for debugging purpose.

    