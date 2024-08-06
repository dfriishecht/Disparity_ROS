# Disparity_ROS
ROS1 for generating a disparity map from a stereo image pair. The node supports either Raft Stereo or SGBM as the disparity methods.

By default, the node will use Raft Stereo, as it delivers higher quality at a small performance penalty relative to SGBM.

## Setup Conda Environment
To set up the Conda environment, run the following command:
```bash
conda env create -f conda_environment.yaml
```

## Running the Node

1. Activate conda environment:
```
conda activate disparity_node
```

2. Run the Node:
```
rosrun disparity disparity_node.py
```

If you want to change the disparity method without restarting the node, in a seperate terminal use:
```
rosparam set /disp_method 'RAFT'
```
The above command will set the mode to Raft Stereo. If you wish to use SGBM, type `SGBM` instead.
