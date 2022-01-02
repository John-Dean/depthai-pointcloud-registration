#!/usr/bin/env python3

# http://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html

import time
import copy
import numpy as np
import depthai as dai
import contextlib
import open3d as o3d

import dai_pipeline
import pointcloud_alignment


pipeline = dai_pipeline.create_pipeline()

# http://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html


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
				
queue_frames	=	5
visualizer = None
global_pointcloud = o3d.geometry.PointCloud()

with contextlib.ExitStack() as stack:
	dai_available_devices = dai.Device.getAllAvailableDevices()
	if len(dai_available_devices) == 0:
		raise RuntimeError("No devices found!")
	else:
		print("Found", len(dai_available_devices), "devices")

	camera_pointcloud_colors = [ [1, 0.706, 0], [0, 0.651, 0.929], [0.25, 0.25, 0.25]]
	devices	=	[];
	device_number = 0;
	# Go through each device and set them up with a pipeline
	# Also gather their intrinsics and queues, as well as setting up a pointcloud for them
	for available_device in dai_available_devices:
		device = stack.enter_context(dai.Device(pipeline, available_device))
		devices.append(dai_pipeline.setup_device(device, camera_pointcloud_colors[device_number]))
		device_number += 1
	
	is_aligned	=	False;
	frame = 0
	transforms = []
	merged_clouds = []
	fps_limit = 120;
	
	while True:
		start_time = time.time()
		frame = frame + 1;
		
		for i in range(len(devices)):
			device = devices[i]
			device.update_pointcloud()
			
			# This merges the first few frames into a massive pointcloud
			# This means registration is done with the biggest dataset possible
			if frame <= queue_frames:
				if len(merged_clouds) <= i:
					merged_clouds.append(o3d.geometry.PointCloud())
				pointcloud = copy.deepcopy(device.pointcloud)
				smoothing_points_to_check = 50; #More points = better smoothing but worse performance        
				pointcloud = pointcloud.remove_statistical_outlier(smoothing_points_to_check, 0.5)[0]
				merged_clouds[i] += pointcloud;
			else:
				1+1; #This is here so the line below can be commented out as needed
				# device.visualize_pointcloud()
			
		if frame >= queue_frames:
			# Perfrom pointcloud alignment (global registration)
			if is_aligned == False:			
				is_aligned = True
				
				voxel_size = 0.05 # In meteres, so 1.00 is 1m
				all_pointclouds = []
				
				for i in range(len(devices)):
					if len(transforms) <= i:
						transforms.append(np.identity(4))
						
					if len(all_pointclouds) <= i:
						all_pointclouds.append(None)
					
					if i == 0:
						transforms[i] = np.identity(4)
					else:
						transforms[i] = pointcloud_alignment.align_two_pointclouds(merged_clouds[i], merged_clouds[0])[0]
				
				# If we actually have >1 camera
				# if len(devices_info) > 1:
				# 	pose_graph = align_clouds(all_pointclouds, voxel_size)
					
				# 	for i in range(len(devices_info)):
				# 		all_pointclouds[i].transform(pose_graph.nodes[i].pose)
				# 		transforms[i].append(pose_graph.nodes[i].pose);
						
						
				
			else:					
				voxel_size = 0.01
				combined_cloud = o3d.geometry.PointCloud()
				for i in range(len(devices)):
					device = devices[i]
					
					source = device.pointcloud;
					
					source_temp = copy.deepcopy(source)
					source_temp = source_temp.voxel_down_sample(voxel_size)
					
					source_temp.transform(transforms[i])
				
					combined_cloud +=	source_temp
				
				
				global_pointcloud.points = combined_cloud.points;
				global_pointcloud.colors = combined_cloud.colors;
				
				if visualizer == None:
					visualizer = o3d.visualization.Visualizer()
					# Changed to only show the window when we start rendering
					visualizer.create_window()
					visualizer.add_geometry(global_pointcloud)
					origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
					visualizer.add_geometry(origin)
				else:
					visualizer.update_geometry(global_pointcloud)
					visualizer.poll_events()
					visualizer.update_renderer()
			
			
		end_time	=	time.time();
		# print( str(1/(end_time - start_time)) + "fps")
		time.sleep(max(1./fps_limit - (end_time- start_time), 0))
