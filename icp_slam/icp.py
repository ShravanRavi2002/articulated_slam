import rosbag
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.neighbors import NearestNeighbors

MAX_ITERATIONS = 100
EPSILON = 1e-9
SCAN_SKIP = 25

def nearest_neighbor(prior_points: np.array, 
                     posterior_points: np.array):

  assert prior_points.shape[1] == 2 and posterior_points.shape[1] == 2

  neigh = NearestNeighbors(n_neighbors=1)
  neigh.fit(posterior_points)
  distances, indices = neigh.kneighbors(prior_points, return_distance=True)
  return distances.ravel(), indices.ravel()

def get_optimal_projection(A: np.array, 
                           B: np.array):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def calculate_transform_between_scans(prior_scan: Scan,
                                      posterior_scan: Scan,
                                      new_map: Optional[Map]=None):
  """
  Calculates a transform between two scans. Returns a transform
  for the location and the angle
  """

  prior_points = np.array(prior_scan.euclidean_scan.points)
  posterior_points = np.array(posterior_scan.euclidean_scan.points)

  # weighting points
  prior_weights = np.reciprocal(np.linalg.norm(prior_points, axis=1))
  posterior_weights = np.reciprocal(np.linalg.norm(posterior_points, axis=1))
  prior_points = prior_points[prior_weights > 0.15, :]
  posterior_points = posterior_points[posterior_weights > 0.15, :]

  m = prior_points.shape[1]

  src = np.ones((m+1, prior_points.shape[0]))
  dst = np.ones((m+1, posterior_points.shape[0]))
  src[:m, :] = np.copy(prior_points.T)
  dst[:m, :] = np.copy(posterior_points.T)

  prev_error = 0
  
  for i in range(MAX_ITERATIONS):
    # first, we get the closest points in the posterior scan to all 
    # the points in the prior scan
    distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
    assert len(prior_points) == len(distances)

    # next, we calculate the best transform
    T, _, _ = get_optimal_projection(src[:m, :].T, dst[:m, indices].T)

    # update the source
    src = np.dot(T, src)

    if new_map:
      new_map.add_scan_points(scan_points=src[:m, :].T, color="orange")

    # calculate the avg error between the point clouds
    mean_error = np.mean(distances)
    if np.abs(prev_error - mean_error) < EPSILON:
      # print(f"Completed icp in {i} iterations")
      break
    prev_error = mean_error

  T, _, _ = get_optimal_projection(prior_points, src[:m,:].T)

  return T, distances

def apply_transforms_to_scan(scan: Scan,
                             transform_chain: List[np.array]):

  points = np.array(scan.get_euclidean_scan_robot_frame())
  points = np.hstack((points, np.ones((points.shape[0], 1))))

  for transform in transform_chain:
    points = np.dot(transform, points.T).T
    # points = np.matmul(transform, points.T).T
  return points[:, :2]

def reconstruct_map_from_odom(bag_path: str):
  """
  Constructing a map based on odometry readings. This function is most useful
  for when bag files are generated from the simulation. We expect to see a 
  perfect map in this instance.
  """

  bag = rosbag.Bag(bag_path, 'r')
  new_map = Map()

  odom_idx = 0
  scan_idx = 0

  initial_robot_loc = None # assuming we do not start with an odom of 0,0
  current_robot_loc = None

  for i, msg in tqdm(enumerate(bag.read_messages(topics=['/odom', "/scan"]))):
    if msg.topic == "/odom":

      x, y, z = quaternion_to_euler_angle_vectorized(msg.message.pose.pose.orientation.w, 
                                                     msg.message.pose.pose.orientation.x, 
                                                     msg.message.pose.pose.orientation.y, 
                                                     msg.message.pose.pose.orientation.z)

      current_robot_loc = (msg.message.pose.pose.position.x, msg.message.pose.pose.position.y)
      current_robot_angle = z

      assert x == 0 and y == 0

      new_map.add_robot(robot_loc=current_robot_loc,
                        start=(odom_idx == 0))

      odom_idx += 1
      
    elif msg.topic == "/scan":
      if not current_robot_loc:
        continue

      scan = Scan(angle_min=msg.message.angle_min,
                  angle_max=msg.message.angle_max,
                  angle_increment=msg.message.angle_increment,
                  ranges=msg.message.ranges)

      if scan_idx % SCAN_SKIP == 0:
        new_map.add_scan(scan, 
                         robot_loc=current_robot_loc,
                         robot_angle=current_robot_angle)

      scan_idx += 1

  new_map.save_map("map_odom.png")

def reconstruct_map_from_lidar(bag_path: str):
  """
  Reconstructing the map based on LiDAR scans. This function is
  used when we don't have any previous information about the
  odometry readings. We use the ICP algorithm to solve this.
  """
  
  bag = rosbag.Bag(bag_path, 'r')
  new_map = Map()

  scan_idx = 0
  visualization_incr = 0

  prev_scan = None

  tranform_chain = [] # maintains a list of all previous tranformations from one scan to the next

  robot_loc = np.array([0, 0, 1])

  for i, msg in tqdm(enumerate(bag.read_messages(topics=["/scan"]))):

    if i > 700:
      break

    if scan_idx % SCAN_SKIP == 0:
      scan = Scan(angle_min=msg.message.angle_min,
                  angle_max=msg.message.angle_max,
                  angle_increment=msg.message.angle_increment,
                  ranges=msg.message.ranges)

      if prev_scan == None:
        # the first scan we consider, we add it centered at the origin
        new_map.add_scan(scan,
                         robot_loc=(0, 0),
                         robot_angle=0,
                         color="red")

        new_map.add_robot((0, 0), start=True)
      else:
        # for all other scans, we have to estimate the robot location and
        # angle using ICP
        T, _ = calculate_transform_between_scans(scan, prev_scan, new_map)
        tranform_chain.append(T)

        # apply transforms to current scan
        transformed_points = apply_transforms_to_scan(scan, tranform_chain)

        # add the previous scan
        new_map.add_scan(prev_scan,
                         robot_loc=(0, 0),
                         robot_angle=0,
                         color="red")

        # add the current scan with transform
        new_map.add_scan(scan,
                         robot_loc=(0, 0),
                         robot_angle=0,
                         color="blue")

        # add the transformed scan to the map
        # if scan_idx % 25 == 0:
        #   new_map.add_scan_points(scan_points=transformed_points, color="green")

        # add the scan to the map using robot location and angle
        robot_loc = np.dot(T, robot_loc.T)
        new_map.add_robot((robot_loc[0], robot_loc[1]))
        assert robot_loc[2] == 1

        visualization_incr += 1

        new_map.save_map(f"map_lidar_{scan_idx}.png", clear=True)

    prev_scan = scan

    scan_idx += 1

  # new_map.save_map(f"map_lidar.png")

if __name__ == "__main__":

  bag_path = "/u/ahjoshi9/cs378_terry_cruise/bags/2023-12-06-11-42-56.bag"

  reconstruct_map_from_lidar(bag_path=bag_path)

  