from pointcloud import DepthAIPointcloud
import numpy as np
import depthai as dai

def create_pipeline():
	# StereoDepth config options.
	lrcheck = True  # Better handling for occlusions
	extended = False  # Closer-in minimum depth, disparity range is doubled
	subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
	left_right_check_threshold = 5
	confidence_threshold = 125
	
	pipeline = dai.Pipeline()

	# Define sources and outputs
	left = pipeline.createMonoCamera()
	right = pipeline.createMonoCamera()
	stereo = pipeline.createStereoDepth()
	depthOut = pipeline.createXLinkOut()

	left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	left.setBoardSocket(dai.CameraBoardSocket.LEFT)
	left.setFps(60.0);
	
	right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
	right.setFps(60.0);

	stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
	stereo.initialConfig.setLeftRightCheckThreshold(left_right_check_threshold);
	stereo.setLeftRightCheck(lrcheck)
	stereo.setExtendedDisparity(extended)
	stereo.setSubpixel(subpixel)
	stereo.setRectifyEdgeFillColor(0)

	depthOut.setStreamName("depth")

	# Linking
	left.out.link(stereo.left)
	right.out.link(stereo.right)
	stereo.depth.link(depthOut.input)

	return pipeline

def setup_device(device, color = [1, 0.706, 0]):
	device_calibration_data = device.readCalibration()
	
	depth_queue = device.getOutputQueue("depth", 1, blocking=False)
	
	sample_image	=	depth_queue.get();

	width	=	sample_image.getWidth();
	height	=	sample_image.getHeight();
	print(width)
	print(height)
	intrinsics	=	np.array(device_calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, width, height))
	extrinsics	=	np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]);

	output = DepthAIPointcloud(device, width, height, intrinsics, extrinsics, depth_queue, color)
	
	return output
	