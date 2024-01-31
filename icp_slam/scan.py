import io
import math
import rosbag
import numpy as np
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

class Frame(Enum):
    LOCAL='local'
    GLOBAL='global'

@dataclass
class RawLidarScan:
    angle_min: float
    angle_max: float
    angle_increment: float
    ranges: List[float]

    def __init__(self, msg: rosbag.bag.BagMessage):
        self.angle_min = msg.message.angle_min
        self.angle_max = msg.message.angle_max
        self.angle_increment = msg.message.angle_increment
        self.ranges = msg.message.ranges

@dataclass
class EuclideanScan:
    points: List[Tuple[float, float]]

@dataclass 
class Odometry:
    loc: Tuple[float, float] # (x, y)
    angle: float # radians
    
def get_transformation_from_odom(odom: Odometry):
        return np.array([[np.cos(odom.angle), -np.sin(odom.angle), odom.loc[0]],
                                    [np.sin(odom.angle), np.cos(odom.angle), odom.loc[1]],
                                    [0, 0, 1]])

class Scan:

    def __init__(self, 
                 raw_scan: RawLidarScan,
                 odom: Odometry):

        self.odom = odom
        self.raw_scan = raw_scan
        self.euclidean_scan = self._raw_to_euclidean(raw_scan)
        self.global_scan = self.convert_to_frame(odom)
    
    def _raw_to_euclidean(self, raw_scan: RawLidarScan):
        euclidean_scan = []
        for i, angle in enumerate(np.arange(raw_scan.angle_min, raw_scan.angle_max, raw_scan.angle_increment)):
            euclidean_scan.append((raw_scan.ranges[i] * math.cos(angle), raw_scan.ranges[i] * math.sin(angle)))
        return EuclideanScan(points=euclidean_scan)

    def _euclidean_to_raw(self, euclidean_scan: EuclideanScan):
        pass

    def visualize(self, 
                  clip: bool=False, 
                  frame: Frame=Frame.LOCAL,
                  in_memory: bool=False, 
                  buffer: Optional[io.BytesIO]=None, 
                  path: Optional[str]=None):

        # assert (in_memory and buffer != None) or (not in_memory and path != None)

        # plt.scatter(*self.odom.loc, c="black", marker="s", s=30)
        s1 = math.tan(self.odom.angle)
        x = np.array([self.odom.loc[0], self.odom.loc[0] + math.cos(self.odom.angle)])
        y = np.array([self.odom.loc[1], self.odom.loc[1] + math.sin(self.odom.angle)])
        plt.plot(x, y, c="blue")

        points = self.euclidean_scan if frame == Frame.LOCAL else self.global_scan

        for i, (x, y) in enumerate(points):
            if clip:
                local_x, local_y = self.euclidean_scan.points[i]
                # if math.sqrt(local_x**2 + local_y**2) <= 30:
                    # plt.scatter(x, y, c="red", s=0.5)
            # else:
                # plt.scatter(x, y, c="red", s=0.5)

        if in_memory:
            plt.savefig(buffer, format='png')
        else:
            plt.savefig(path)
            
    def convert_to_frame(self,
                          odom: Odometry):

        transformation = np.array([[np.cos(odom.angle), -np.sin(odom.angle), odom.loc[0]],
                                    [np.sin(odom.angle), np.cos(odom.angle), odom.loc[1]],
                                    [0, 0, 1]])
        
        points = []
        for x, y in self.euclidean_scan.points:
            p = np.array([x, y, 1])
            point = np.matmul(transformation, p.T)
            points.append(tuple(point[:2]))

        return points
            