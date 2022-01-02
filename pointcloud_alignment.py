import numpy as np
import copy
import open3d as o3d

def preprocess_point_cloud(pointcloud, voxel_size):
	pointcloud.voxel_down_sample(voxel_size)
	
	radius_normal = voxel_size * 2
	pointcloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

	radius_feature = voxel_size * 5
	pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pointcloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
	
	return pointcloud, pointcloud_fpfh

def prepare_dataset(voxel_size, source_pointcloud, target_pointcloud):
	source = copy.deepcopy(source_pointcloud)
	target = copy.deepcopy(target_pointcloud)
	
	smoothing_points_to_check = 30; #More points = better smoothing but worse performance        
	source = source.remove_statistical_outlier(smoothing_points_to_check, 0.5)[0]
	target = target.remove_statistical_outlier(smoothing_points_to_check, 0.5)[0]
	
	temp_source = copy.deepcopy(source)
	temp_target = copy.deepcopy(target)
	
	source_down, source_fpfh = preprocess_point_cloud(temp_source, voxel_size)
	target_down, target_fpfh = preprocess_point_cloud(temp_target, voxel_size)
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
		], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99999))
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
	# method = o3d.pipelines.registration.TransformationEstimationPointToPlane();
	method = o3d.pipelines.registration.TransformationEstimationPointToPoint();
	
	result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, current_transform, method)
	return result

def align_two_pointclouds(pointcloud_to_align, target_pointcloud, voxel_size = 0.05):
	source = copy.deepcopy(pointcloud_to_align)
	target = copy.deepcopy(target_pointcloud)
	
	transform = np.identity(4)
	
	source.estimate_normals()
	target.estimate_normals()
	
	source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)
	
	result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
															
	print(result_ransac)
	
	transform = result_ransac.transformation;
	
	# On or off this seems to not help
	result_icp = refine_registration(source, target, voxel_size, transform)
	print(result_icp)
	transform = result_icp.transformation;
	
	# For kicks, let's try refine it again, but tighter
	result_icp = refine_registration(source, target, voxel_size, transform, voxel_size * 0.1)
	print(result_icp)
	transform = result_icp.transformation;
	
	# And one final time
	result_icp = refine_registration(source, target, voxel_size, transform, voxel_size * 0.01)
	print(result_icp)
	transform = result_icp.transformation;
	
	return transform, result_icp
