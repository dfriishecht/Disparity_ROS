#!/usr/bin/env python
import sys
import os
import rospy
import argparse
import glob
import numpy as np
import torch

HOME_DIR = os.path.expanduser('~')
DEVICE = 'cuda'

os.chdir(HOME_DIR + '/RAFT-Stereo')
sys.path.append(HOME_DIR + '/RAFT-Stereo')
sys.path.append('core')

from tqdm import tqdm
import time
from pathlib import Path
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from raftstereo.msg import depth

import submodules.disparity as disparity

class StereoSync:
    def __init__(self):
        self.left_image_sub = Subscriber('/theia/left/image_rect_color', Image, queue_size=2) # Replace with correct image topic if needed
        self.right_image_sub = Subscriber('/theia/right/image_rect_color', Image, queue_size=2) #
    
        self.pub = rospy.Publisher('/disparity_map', depth, queue_size=2)

        self.stereo_image_ = ApproximateTimeSynchronizer([self.left_image_sub, self.right_image_sub], queue_size=2, slop=0.1)
        self.stereo_image_.registerCallback(self.stereo_callback)

        rospy.set_param('/disp_method', 'RAFT')
        self.method = rospy.get_param('/disp_method') # Use to specify disparity map generation. Options are 'RAFT' or 'SGBM'
        self.model = self.init_()

    def init_(self):
        self.method = rospy.get_param('/disp_method')
        model = disparity.disparity(method=self.method)
        print("node initialized...")
        return model
    
    def stereo_callback(self, image_L, image_R):

        print("stereo pair recieved ...")
        img_L = np.ndarray(shape=(image_L.height, image_L.width, 3), dtype=np.uint8, buffer=image_L.data)
        img_R = np.ndarray(shape=(image_L.height, image_L.width, 3), dtype=np.uint8, buffer=image_R.data)

        disp_map = self.model.generate_folder(self.method, img_L, img_R)
        disp_msg = depth()
        disp_msg.imageData = disp_map.astype('float32').ravel()
        self.pub.publish(disp_msg)

        print("disparity complete")
        time.sleep(1)
        self.model = self.init_()

def init():
    b = StereoSync()
    rospy.init_node('disparity', anonymous=False)
    rospy.spin()

if __name__ == '__main__':
    init()
