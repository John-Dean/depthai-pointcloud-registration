#!/usr/bin/env python3

# http://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html

import time
import copy
import numpy as np
import depthai as dai
import contextlib
import open3d as o3d

from pointcloud_generator import PointCloudGenerator

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

pipeline = create_pipeline()

# http://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html

def preprocess_point_cloud(pcd, voxel_size):
	# pcd = pcd.voxel_down_sample(voxel_size)
	
	radius_normal = voxel_size * 2
	pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

	radius_feature = voxel_size * 5
	pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
	return pcd, pcd_fpfh

def prepare_dataset(voxel_size, source_pointcloud, target_pointcloud):
	source = copy.deepcopy(source_pointcloud)
	target = copy.deepcopy(target_pointcloud)
	
	smoothing_points_to_check = 30; #More points = better smoothing but worse performance        
	source = source.remove_statistical_outlier(smoothing_points_to_check, 0.5)[0]
	target = target.remove_statistical_outlier(smoothing_points_to_check, 0.5)[0]
	
	source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
	target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
	return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
	distance_threshold = voxel_size * 1.5
	print(":: RANSAC registration on downsampled point clouds.")
	print("   Since the downsampling voxel size is %.3f," % voxel_size)
	print("   we use a liberal distance threshold %.3f." % distance_threshold)
	result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
		source_down, target_down, source_fpfh, target_fpfh, True,
		distance_threshold,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
		3, [
			o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
				0.9),
			o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
				distance_threshold)
		], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
	return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
									 target_fpfh, voxel_size):
	distance_threshold = voxel_size * 0.5
	print(":: Apply fast global registration with distance threshold %.3f" \
			% distance_threshold)
	result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
		source_down, target_down, source_fpfh, target_fpfh,
		o3d.pipelines.registration.FastGlobalRegistrationOption(
			maximum_correspondence_distance=distance_threshold))
	return result

def refine_registration(source, target, voxel_size, current_transform, distance_threshold = None):
	if distance_threshold == None:
		distance_threshold = voxel_size * 0.4
	print(":: Point-to-plane ICP registration is applied on original point")
	print("   clouds to refine the alignment. This time we use a strict")
	print("   distance threshold %.3f." % distance_threshold)
	method = o3d.pipelines.registration.TransformationEstimationPointToPlane();
	# method = o3d.pipelines.registration.TransformationEstimationPointToPoint();
	
	result = o3d.pipelines.registration.registration_icp(
		source, target, distance_threshold, current_transform,
		method)
	return result

# http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
def pairwise_registration(source, target,max_correspondence_distance_coarse, max_correspondence_distance_fine, base_transform = np.identity(4)):
	print("Apply point-to-plane ICP")
	icp_coarse = o3d.pipelines.registration.registration_icp(
		source, target, max_correspondence_distance_coarse, base_transform,
		o3d.pipelines.registration.TransformationEstimationPointToPlane())
	icp_fine = o3d.pipelines.registration.registration_icp(
		source, target, max_correspondence_distance_fine,
		icp_coarse.transformation,
		o3d.pipelines.registration.TransformationEstimationPointToPlane())
	transformation_icp = icp_fine.transformation
	information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
		source, target, max_correspondence_distance_fine,
		icp_fine.transformation)
	return transformation_icp, information_icp

def full_registration(pointclouds, max_correspondence_distance_coarse,
					  max_correspondence_distance_fine, transforms):
	pose_graph = o3d.pipelines.registration.PoseGraph()
	odometry = np.identity(4)
	pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
	n_pointclouds = len(pointclouds)
	for source_id in range(n_pointclouds):
		for target_id in range(source_id + 1, n_pointclouds):
			transformation_icp, information_icp = pairwise_registration(
				pointclouds[source_id], pointclouds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
			print("Build o3d.pipelines.registration.PoseGraph")
			if target_id == source_id + 1:  # odometry case
				odometry = np.dot(transformation_icp, odometry)
				pose_graph.nodes.append(
					o3d.pipelines.registration.PoseGraphNode(
						np.linalg.inv(odometry)))
				pose_graph.edges.append(
					o3d.pipelines.registration.PoseGraphEdge(source_id,
															 target_id,
															 transformation_icp,
															 information_icp,
															 uncertain=False))
			else:  # loop closure case
				pose_graph.edges.append(
					o3d.pipelines.registration.PoseGraphEdge(source_id,
															 target_id,
															 transformation_icp,
															 information_icp,
															 uncertain=True))
	return pose_graph

def align_clouds(all_pointclouds, voxel_size):
	max_correspondence_distance_coarse = voxel_size * 15
	max_correspondence_distance_fine = voxel_size * 1
	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		pose_graph = full_registration(all_pointclouds,
					max_correspondence_distance_coarse,
					max_correspondence_distance_fine)


	option = o3d.pipelines.registration.GlobalOptimizationOption(
		max_correspondence_distance=max_correspondence_distance_fine,
		edge_prune_threshold=0.25)
	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		o3d.pipelines.registration.global_optimization(
			pose_graph,
			o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
			o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
			option)
	
	return pose_graph
				
queue_frames	=	8

final_visualiser = None;

with contextlib.ExitStack() as stack:
	dai_available_devices = dai.Device.getAllAvailableDevices()
	if len(dai_available_devices) == 0:
		raise RuntimeError("No devices found!")
	else:
		print("Found", len(dai_available_devices), "devices")

	devices_info	=	[];

	# Go through each device and set them up with a pipeline
	# Also gather their intrinsics and queues, as well as setting up a pointcloud for them
	for available_device in dai_available_devices:
		device = stack.enter_context(dai.Device(pipeline, available_device))
		device_calibration_data = device.readCalibration()

		device_info = lambda: [p for p in device_info.__dict__.keys()]
		device_info.width	=	640;
		device_info.height	=	400;
		device_info.intrinsics = np.array(device_calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, device_info.width, device_info.height))
		device_info.extrinsics	=	np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]);	# Dummy extrinsics for now, ideally we would work out the actual extrinsics later and replace this
		device_info.pointcloud_generator = PointCloudGenerator(device_info.intrinsics, device_info.width, device_info.height)
		device_info.depth_queue = device.getOutputQueue("depth", queue_frames, blocking=False)
		
		devices_info.append(device_info)
		
		if final_visualiser == None:
			final_visualiser = PointCloudGenerator(device_info.intrinsics, device_info.width, device_info.height)
	
	is_aligned	=	False;
	frame = 0
	transforms = []
	merged_clouds = []
	camera_pointcloud_colors = [ [1, 0.706, 0], [0, 0.651, 0.929], [0.25, 0.25, 0.25]]
	fps_limit = 60;
	
	while True:
		start_time = time.time()
		frame = frame + 1;
		
		for i in range(len(devices_info)):
			device_info = devices_info[i]
			
			if len(camera_pointcloud_colors) <= i:
				color = camera_pointcloud_colors[0]
			else:
				color = camera_pointcloud_colors[i]
			
			# Grab the depth image from the camera
			depth_queue	=	device_info.depth_queue;
			depth_image	=	depth_queue.get();
			depth_frame	=	np.ascontiguousarray(depth_image.getFrame())
			
			# Convert the depth image into a pointcloud and recolour it
			device_info.pointcloud_generator.depth_to_projection(depth_frame, device_info.extrinsics)
			device_info.pointcloud_generator.color_pointcloud(color)
			
			# This merges the first few frames into a massive pointcloud
			# This means registration is done with the biggest dataset possible
			if frame <= queue_frames:
				if len(merged_clouds) <= i:
					merged_clouds.append(o3d.geometry.PointCloud())
				pointcloud = copy.deepcopy(device_info.pointcloud_generator.pcl)
				smoothing_points_to_check = 50; #More points = better smoothing but worse performance        
				pointcloud = pointcloud.remove_statistical_outlier(smoothing_points_to_check, 0.5)[0]
				merged_clouds[i] += pointcloud;			
			
		if frame > queue_frames:
			# Perfrom pointcloud alignment (global registration)
			if is_aligned == False:			
				is_aligned = True
				
				voxel_size = 0.05 # In meteres, so 1.00 is 1m
				all_pointclouds = []
				
				for i in range(len(devices_info)):
					if len(transforms) <= i:
						transforms.append(np.identity(4))
						
					if len(all_pointclouds) <= i:
						all_pointclouds.append(None)
					
					if i == 0:
						transforms[i] = np.identity(4)
					else:
						source = copy.deepcopy(merged_clouds[i])
						target = copy.deepcopy(merged_clouds[0])
						
						source.estimate_normals()
						target.estimate_normals()
						
						all_pointclouds[i] = source.voxel_down_sample(voxel_size);
						all_pointclouds[0] = target.voxel_down_sample(voxel_size);
						
						source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)
						
						result_ransac = execute_global_registration(source_down, target_down,
															source_fpfh, target_fpfh,
																	voxel_size)
															
						print(result_ransac)
						
						transforms[i] = result_ransac.transformation;
						
						# On or off this seems to not help
						result_icp = refine_registration(source, target, voxel_size, transforms[i])
						print(result_icp)
						transforms[i] = result_icp.transformation;
						
						# For kicks, let's try refine it again, but tighter
						result_icp = refine_registration(source, target, voxel_size, transforms[i], voxel_size * 0.1)
						print(result_icp)
						transforms[i] = result_icp.transformation;
						
						
						# And one finalt ime
						result_icp = refine_registration(source, target, voxel_size, transforms[i], voxel_size * 0.01)
						print(result_icp)
						transforms[i] = result_icp.transformation;
				
				# If we actually have >1 camera
				# if len(devices_info) > 1:
				# 	pose_graph = align_clouds(all_pointclouds, voxel_size)
					
				# 	for i in range(len(devices_info)):
				# 		all_pointclouds[i].transform(pose_graph.nodes[i].pose)
				# 		transforms[i].append(pose_graph.nodes[i].pose);
						
						
				
			else:
				voxel_size = 0.01
				combined_cloud = o3d.geometry.PointCloud()
				for i in range(len(devices_info)):
					source = devices_info[i].pointcloud_generator.pcl;
					
					source_temp = copy.deepcopy(source)
					source_temp = source_temp.voxel_down_sample(voxel_size)
					
					source_temp.transform(transforms[i])
				
					combined_cloud +=	source_temp
				
				if final_visualiser.pcl == None:
					final_visualiser.pcl = combined_cloud;
				else:
					final_visualiser.pcl.points = combined_cloud.points
					final_visualiser.pcl.colors = combined_cloud.colors
					
				final_visualiser.visualize_pcd()
			
			
		end_time	=	time.time();
		# print( str(1/(end_time - start_time)) + "fps")
		time.sleep(max(1./fps_limit - (end_time- start_time), 0))
