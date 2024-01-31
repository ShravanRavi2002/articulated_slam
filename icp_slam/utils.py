import math
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

MAX_POINT_DISTANCE = 10
POINT_SKIP = 1

def quaternion_to_euler_angle_vectorized(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z

def l2_norm(x, y):
  return math.sqrt(x**2 + y**2)

class Scan:
  """
  Maintains an indiviudal LiDAR scan
  """
  def __init__(
    self,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    ranges: List[float]
  ):
    self.angle_min = angle_min
    self.angle_max = angle_max
    self.angle_increment = angle_increment
    self.ranges = ranges

  def get_euclidean_scan_robot_frame(self):
    """
    Returns the euclidean scan with reference to the robot frame
    """
    euclidean_scan_robot = []
    for i, angle in enumerate(np.arange(self.angle_min, self.angle_max, self.angle_increment)):
        euclidean_scan_robot.append((self.ranges[i] * math.cos(angle), self.ranges[i] * math.sin(angle)))
    euclidean_scan_robot_filtered = [(x, y) for x, y in euclidean_scan_robot if l2_norm(x, y) < MAX_POINT_DISTANCE]
    euclidean_scan_robot_filtered = euclidean_scan_robot_filtered[::POINT_SKIP]
    return euclidean_scan_robot_filtered

  def get_euclidean_scan_odom_frame(self,
                                    robot_loc=List[float],
                                    robot_angle=float):
    """
    Returns the euclidean scan with reference to the odom frame
    """
    euclidean_scan_robot = self.get_euclidean_scan_robot_frame()
    transformation = np.array([
                        [np.cos(robot_angle), -np.sin(robot_angle), robot_loc[0]],
                        [np.sin(robot_angle), np.cos(robot_angle), robot_loc[1]],
                        [0, 0, 1]])

    euclidean_scan_odom = []
    for x, y in euclidean_scan_robot:
      p = np.array([x, y, 1])
      point = np.matmul(transformation, p.T)
      euclidean_scan_odom.append(tuple(point[:2]))
    return euclidean_scan_odom

class Map:
  """
  Maintains a set of points that form a map. This class also contains 
  additional visualization methods. Note that all values handled in
  this class should be in the MAP FRAME.
  """
  def __init__(
    self,
    xlim: Optional[Tuple]=None,
    ylim: Optional[Tuple]=None
  ):
    self.points = []
    
    if xlim and ylim:
      plt.xlim(*xlim)
      plt.ylim(*ylim)

  def add_scan(self, 
               scan: Scan,
               robot_loc: Tuple[float],
               robot_angle: float,
               color: str="purple"):
    assert len(robot_loc) == 2

    scan_points = scan.get_euclidean_scan_odom_frame(robot_loc, robot_angle)
    self.points.extend(scan_points)

    x_values = [point[0] for point in scan_points]
    y_values = [point[1] for point in scan_points]
    plt.scatter(x_values, y_values, c=color, marker="s", s=2)

  def add_scan_points(self,
                      scan_points: np.array,
                      color: str="purple"):
    self.points.extend(scan_points.tolist())

    x_values = [point[0] for point in scan_points]
    y_values = [point[1] for point in scan_points]
    plt.scatter(x_values, y_values, c=color, marker="s", s=2)

  def add_robot(self,
                robot_loc: Tuple[float],
                start: bool=False):
    assert len(robot_loc) == 2

    # TODO: get orientation 
    color = "red" if start else "black"
    size = 30 if start else 4
    plt.scatter(robot_loc[0], robot_loc[1], c=color, marker="o", s=size)
    
  def save_map(self,
               image_file_path: str,
               clear: bool=True):
    plt.savefig(image_file_path)
    if clear:
      plt.clf()
    
