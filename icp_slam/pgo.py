import io
import os
import sys
import copy
import math

import rosbag
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import open3d.cpu.pybind as o3d
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

sys.path.append(Path(os.path.abspath(__file__)).parent)

from icp import quaternion_to_euler_angle_vectorized, calculate_transform_between_scans
from scan import RawLidarScan, EuclideanScan, Scan, Odometry



def fetch_scans_from_bag(bag_path: str):
    bag = rosbag.Bag(bag_path, 'r')
    scans = [] 
    odom = None 

    for i, msg in tqdm(enumerate(bag.read_messages(topics=['/odom', "/scan"]))):

        if msg.topic == '/odom':
            x, y, z = quaternion_to_euler_angle_vectorized(msg.message.pose.pose.orientation.w, 
                                                            msg.message.pose.pose.orientation.x, 
                                                            msg.message.pose.pose.orientation.y, 
                                                            msg.message.pose.pose.orientation.z)


            odom_loc = tuple((msg.message.pose.pose.position.x, msg.message.pose.pose.position.y))
            odom_angle = z
            odom = Odometry(loc=odom_loc, angle=odom_angle)

            assert x == 0 and y == 0

        elif msg.topic == '/scan':
            if odom is None:
                continue
            current_scan = Scan(RawLidarScan(msg), odom)
            scans.append(current_scan)

    return scans


def fetch_point_clouds(scans: List[Scan]):
    pcds = []
    prev_elevation = 0
    for scan in scans:
        mask = np.linalg.norm(scan.euclidean_scan.points, axis=1) < 20
            
        P_s = np.array(scan.global_scan)[mask]
        arr_s = np.zeros((P_s.shape[0], 3))
        arr_s[:, :2] = P_s
        arr_s[:, 2] = prev_elevation
        prev_elevation += 0.01
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr_s))
        pc.estimate_normals()
        pcds.append(pc)

    return pcds


def pairwise_registration(source: o3d.geometry.PointCloud, 
                          target: o3d.geometry.PointCloud, 
                          max_correspondence_distance_coarse: float, 
                          max_correspondence_distance_fine: float):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
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


def pairwise_registration_custom_icp(source: Scan, 
                                    target: Scan,
                                    max_correspondence_distance_fine: float):
    T, _ = calculate_transform_between_scans(source, target)

    transformation_icp = np.identity(4)
    transformation_icp[:2, :2] = T[:2, :2]
    transformation_icp[3, :2] = T[2, :2]
    transformation_icp[3, 3] = T[2, 2]

    source_pc, target_pc = fetch_point_clouds([source, target])
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_pc, target_pc, max_correspondence_distance_fine,
        transformation_icp)
    return transformation_icp, information_icp


def generate_pose_graph_custom_icp(scans, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(scans)
    for source_id in tqdm(range(n_pcds)):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration_custom_icp(
                scans[source_id], scans[target_id], max_correspondence_distance_fine)

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
            # else:  # loop closure case
            #     pose_graph.edges.append(
            #         o3d.pipelines.registration.PoseGraphEdge(source_id,
            #                                                  target_id,
            #                                                  transformation_icp,
            #                                                  information_icp,
            #                                                  uncertain=True))
    return pose_graph

def generate_pose_graph(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in tqdm(range(n_pcds)):
        for target_id in range(source_id + 1, n_pcds):
            if target_id != source_id + 1:
                continue
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], 
                max_correspondence_distance_coarse, max_correspondence_distance_fine)

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


def optimize_pose_graph(pose_graph, max_correspondence_distance_fine):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    return pose_graph


def draw_robot(odom: Odometry, color: str = 'black'):
      plt.scatter(*odom.loc, c=color, marker="s", s=30)
      s1 = math.tan(odom.angle)
      x = np.array([odom.loc[0], odom.loc[0] + 0.25 * math.cos(odom.angle)])
      y = np.array([odom.loc[1], odom.loc[1] + 0.25 * math.sin(odom.angle)])
      plt.plot(x, y, c="blue")


def generate_map_plot(pcds, pose_graph, out_dir='/u/shravanr/articulated_slam/out', name='map.png'):
    plt.clf()
    optimized_poses = copy.deepcopy(pcds)
    start_loc = None
    end_loc = None

    for point_id in range(len(pcds)):
        optimized_poses[point_id].transform(pose_graph.nodes[point_id].pose)
        # print(pose_graph.nodes[point_id].pose)
        
        loc = np.dot(pose_graph.nodes[point_id].pose, [*scans[point_id].odom.loc, 1, 1])[:2]
        
        if point_id == 0:
          start_loc = loc
        if point_id == len(pcds) - 1:
          end_loc = loc

        theta_z = np.arctan2(pose_graph.nodes[point_id].pose[1, 0], 
                             pose_graph.nodes[point_id].pose[0, 0])

        theta = scans[point_id].odom.angle + theta_z

        color = 'black'
        if point_id == 0:
            color = 'green'
        if point_id == len(pcds) - 1:
            color = 'red'
        draw_robot(Odometry(loc=tuple(loc), angle=theta), color=color)

    pc_map = o3d.geometry.PointCloud()
    for pose in optimized_poses:
        pc_map += pose

    
    print(f'Relative Transformation: {end_loc - start_loc}')

    points = np.asarray(pc_map.points)
    plt.scatter(*points.T[:2], c='r', s=0.1)
    plt.savefig(Path(out_dir) / name)
    plt.clf()


def draw_original_map(scans, out_dir='/u/shravanr/articulated_slam/out', name='original_map.png'):
    plt.clf()
    for i, scan in enumerate(scans):
        plt.scatter(*np.array(scan.global_scan).T, c='r')
  
    for i, scan in enumerate(scans):        
        color = 'black'
        if i == 0:
            color = 'green'
        if i == len(scans) - 1:
            color = 'red'
        draw_robot(scan.odom, color=color)

    plt.savefig(Path(out_dir) / name)
    plt.clf()

if __name__ == '__main__':
    voxel_size = 0.02
    
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    FRAME_SAMPLE_WINDOW = 10
    map_out_dir = Path(os.path.abspath(__file__)).parent
    map_out_dir.mkdir(exist_ok=True)

    bag_path = '/u/shravanr/articulated_slam/data/2024-01-30-15-08-34.bag'
    scans = fetch_scans_from_bag(bag_path)
    scans = scans[::FRAME_SAMPLE_WINDOW]
    pcds = fetch_point_clouds(scans)
    
    pose_graph = generate_pose_graph(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine)
    # pose_graph = generate_pose_graph_custom_icp(scans, max_correspondence_distance_fine)
    pose_graph = optimize_pose_graph(pose_graph, max_correspondence_distance_fine)

    draw_original_map(scans)
    generate_map_plot(pcds, pose_graph)

    


