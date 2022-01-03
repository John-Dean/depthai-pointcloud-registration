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
