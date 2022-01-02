#!/usr/bin/env python3

import numpy as np
import open3d as o3d

class DepthAIPointcloud():
	def __init__(self, device, width, height, intrinsic_matrix, extrinsic_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), depth_queue = None, color = [1, 0.706, 0]):
		self.device	=	device
		self.pointcloud = o3d.geometry.PointCloud()

		self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,height,intrinsic_matrix[0][0],intrinsic_matrix[1][1],intrinsic_matrix[0][2],intrinsic_matrix[1][2])
		self.camera_extrinsic    =   extrinsic_matrix
		self.visualizer = None
		self.depth_queue = depth_queue
		self.color = color
	
	def update_pointcloud(self):
		if self.depth_queue == None:
			return False
		data = self.depth_queue.tryGet()
		if data == None:
			return False
		depth_frame	=	np.ascontiguousarray(data.getFrame())
		
		self.depth_to_projection(depth_frame)
		self.color_pointcloud(self.color)
		return True


	def depth_to_projection(self, depth_map):
		depth_o3d = o3d.geometry.Image(depth_map)
		
		max_depth_meters = 5.0;
		point_cloud_reduction_factor = 1; #Int, 1 is the lowest/least reduction 
		
		pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.camera_intrinsic, self.camera_extrinsic, 1000.0, max_depth_meters, point_cloud_reduction_factor)
		
		if self.pointcloud is None:
			self.pointcloud = pointcloud;
		else:
			self.pointcloud.points = pointcloud.points
			self.pointcloud.colors = pointcloud.colors
		return self.pointcloud
	
	def color_pointcloud(self, color=None):
		if color != None:
			self.pointcloud.paint_uniform_color(color)
	
	def visualize_pointcloud(self):
		if self.visualizer == None:
			self.visualizer = o3d.visualization.Visualizer()
			self.visualizer.create_window()
			self.visualizer.add_geometry(self.pointcloud)
			origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
			self.visualizer.add_geometry(origin)
			self.is_visualizer_started = True
		else:
			self.visualizer.update_geometry(self.pointcloud)
			self.visualizer.poll_events()
			self.visualizer.update_renderer()

	def close_window(self):
		self.visualizer.destroy_window()
