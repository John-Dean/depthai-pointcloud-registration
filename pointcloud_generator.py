#!/usr/bin/env python3

import numpy as np
import open3d as o3d

class PointCloudGenerator():
	def __init__(self, intrinsic_matrix, width, height):
		self.depth_map = None
		self.rgb = None
		self.pcl = None

		self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
																		 height,
																		 intrinsic_matrix[0][0],
																		 intrinsic_matrix[1][1],
																		 intrinsic_matrix[0][2],
																		 intrinsic_matrix[1][2])
		self.dummy_extrinsic    =   np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
		self.vis = o3d.visualization.Visualizer()
		self.isstarted = False


	def depth_to_projection(self, depth_map, extrinsic=None):
		self.depth_map = depth_map
		depth_o3d = o3d.geometry.Image(self.depth_map)
		
		max_depth_meters = 5.0;
		point_cloud_reduction_factor = 1; #Int, 1 is the lowest/least reduction 
		
		if extrinsic is None:
			extrinsic   =   self.dummy_extrinsic;
		
		pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.pinhole_camera_intrinsic, extrinsic, 1000.0, max_depth_meters, point_cloud_reduction_factor)
		
		if self.pcl is None:
			self.pcl = pcd;
		else:
			self.pcl.points = pcd.points
			self.pcl.colors = pcd.colors
		return self.pcl
	
	def color_pointcloud(self, color=None):
		if color != None:
			self.pcl.paint_uniform_color(color)
	
	def visualize_pcd(self, color=None):
		if not self.isstarted:
			# Changed to only show the window when we start rendering
			self.vis.create_window()
			self.vis.add_geometry(self.pcl)
			origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
			self.vis.add_geometry(origin)
			self.isstarted = True
		else:
			self.vis.update_geometry(self.pcl)
			self.vis.poll_events()
			self.vis.update_renderer()

	def close_window(self):
		self.vis.destroy_window()
