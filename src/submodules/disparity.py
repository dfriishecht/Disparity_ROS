import sys
import os
HOME_DIR = os.path.expanduser('~')
DEVICE = 'cuda'

os.chdir(HOME_DIR + '/RAFT-Stereo')
sys.path.append(HOME_DIR + '/RAFT-Stereo')
sys.path.append('core')


import cv2
import numpy as np
import glob 
from tqdm import tqdm
from pathlib import Path
import re
import torch
import argparse
from utils.utils import InputPadder
from raft_stereo import RAFTStereo
from matplotlib import pyplot as plt

class disparity:
    def __init__(self, method, DEVICE='cuda'):
        self.method = method
        print(f"Current Method: {self.method}")
        if self.method=='RAFT': 
            parser = argparse.ArgumentParser()
            parser.add_argument('--restore_ckpt', help="restore checkpoint",
                            default= HOME_DIR + '/RAFT-Stereo/models/raftstereo-eth3d.pth')
            parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
            parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                                 default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
            parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                                 default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
            parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
            parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
            parser.add_argument('--valid_iters', type=int, default=32,
                                help='number of flow-field updates during forward pass')

            # Architecture choices
            parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                                help="hidden state and context dimensions")
            parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg_cuda",
                                help="correlation volume implementation")
            parser.add_argument('--shared_backbone', action='store_true',
                                help="use a single backbone for the context and feature encoders")
            parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
            parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
            parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
            parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'],
                                help="normalization of context encoder")
            parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
            parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
            
            self.args = parser.parse_args()
            self.DEVICE = DEVICE
            self.model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
            self.model.load_state_dict(torch.load(self.args.restore_ckpt, map_location=self.DEVICE))

            self.model = self.model.module
            self.model.to(self.DEVICE)
            self.model.eval()

    def generate(self, image1, image2):
        """Generates disparity map from rectified stereo pairs. 

        Args:
            image1: numpy RGB image of uint8 type obtained by cv2.imread().
            image2: numpy RGB image of uint8 type obtained by cv2.imread().
            camera_param_right: dictionary of right camera intrinsics.

        Returns:
            Returns numpy disparity map.

        """
    
        if self.method=='SGBM':
            disparity_map = self.SGBM(image1, image2)
            return disparity_map
        elif self.method=='RAFT':
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            disparity_map = self.RAFT_stereo(image1[None].to(self.DEVICE), image2[None].to(self.DEVICE))
        return disparity_map

    def generate_folder(self, method, img_L, img_R):
        self.method = method
        output_directory = Path(os.path.join(HOME_DIR, 'output', 'DM_'+self.method))
        # Check if disparity maps are already in the output folder
        if output_directory.exists():
            num_files = len([file for file in output_directory.iterdir() if file.is_file() if file.suffix == '.npy'])
            expected_num_files = len(glob.glob(os.path.join(HOME_DIR, 'output', 'LEFT_RECT/*')))
            if num_files == expected_num_files:
                print("Disparity maps already exist in output folder")
                return

        output_directory.mkdir(parents=True, exist_ok=True)

        disparity_map = self.generate(img_L, img_R)
        np.save(output_directory / "np_disparity.npy", disparity_map)
        plt.imsave(output_directory / "disparity.png", disparity_map)
        plt.imsave(output_directory / "left_image.png", img_L)
        return disparity_map

    def SGBM(self, image1, image2):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        win_size = 3
        min_disp = 0
        max_disp = 256 
        num_disp = max_disp - min_disp # Needs to be divisible by 16
        #Create Block matching object. 
        stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
                                    numDisparities = num_disp,
                                    blockSize = 3,
                                    #uniquenessRatio = 25,
                                    #speckleWindowSize = 50,
                                    #speckleRange = 1,
                                    #disp12MaxDiff = 32,
                                    P1 = 8*3*win_size**2,
                                    P2 = 32*3*win_size**2)
                                 
        disparity_map = stereo.compute(image1, image2)
        disparity_map = np.rint(disparity_map/16).astype(int)
        #print(np.unique(disparity_map))
        return disparity_map

    def RAFT_stereo(self, image1, image2):
        with torch.no_grad():


            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, disparity_map = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
            disparity_map = padder.unpad(disparity_map).squeeze()
            self.model.cpu()
            torch.cuda.empty_cache()
        return -disparity_map.cpu().numpy().squeeze()