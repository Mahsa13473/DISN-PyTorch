# DISN in PyTorch
PyTorch implementaion of Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction.   
Original Paper: [Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction](https://arxiv.org/abs/1905.10711)

## System Requirements
- Python 2.7
- PyTorch 1.0
- OpenCV 3.3.0

## Preprocessing:
- use [Vega FEM Library](http://barbic.usc.edu/vega/) for SDF computation
- Download ShapeNetCore.v1 from [here](https://www.shapenet.org/account/) 
- Download ShapeNet rendered images from [here](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)

```
# get obj directory that we have their renderings
# This code will generate obj_path.txt that contains the directories
python get_dir.py

# generate new obj file that can be used for compute SDF
# tThis code will only save f and v values
python generateObj.py

python get_dir1.py

# compute SDF value with VEGA and save output as text files
# ../../utilities/computeDistanceField/computeDistanceField model_normalized.obj 256 256 256 -s 1 -m 1 -g 0.01 -o model_output.txt

# sample 2048 point from SDF gridpoint and save in npy files as a tuple([x, y, z])] = value
python sampling.py

# generate camera_parameters (Intrinsic and RT matrix)
# python get_r2n2_cameras.py $metadata_file $camera_out_file

# split train and test set
python split.py
```

## Installation
You can use [MeshLab](https://www.meshlab.net/#download) to visualize reconstructed 3D model in OFF format

![plot](https://github.com/Mahsa13473/DISN-PyTorch/blob/main/result.png)

