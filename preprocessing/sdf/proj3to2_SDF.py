import os
from projection_cpu import project2d
import numpy as np
import torch

from collections import OrderedDict

resolution = 64


cam_dir = '/local-scratch/mma/DISN/ShapeNetRendering/03001627'
in_dir =  '/local-scratch/mma/DISN/ShapeNetOut64/03001627'


path1 = []
path2 = []

for path, subdirs, files in os.walk(in_dir):
    for name in files:
        if name.endswith('SDF.npy'):
            path_split = path.split(os.sep)
            path2.append(os.path.join(cam_dir, path_split[6], 'rendering', 'rendering_metadata.txt'))
            path1.append(os.path.join(path, 'SDF.npy'))

# print(path1)
print(len(path1))
print(len(path2))


for i in range(len(path1)):
    point_path = path1[i]
    meta_path = path2[i]
    print(point_path)


    read_dict = np.load(point_path, allow_pickle=True)
    SDF_dict = read_dict.item().get('SDF')
    b_min = read_dict.item().get('bmin')
    b_max = read_dict.item().get('bmax')

    points = sorted(SDF_dict.keys()) #sort to have order for your dictionary
    #p = points[idx]
    pp = []
    pp1 = []
    sdf1 = []

    for idx in range(2048):
        p = points[idx]
        point = torch.FloatTensor([[p[0]],[p[1]],[p[2]]])

        sdf = SDF_dict[p]
        sdf = torch.FloatTensor([sdf])
        sdf = sdf.unsqueeze(0)


        gridx = (b_max[0]-b_min[0])/resolution
        gridy = (b_max[1]-b_min[1])/resolution
        gridz = (b_max[2]-b_min[2])/resolution

        px = point[0]*gridx+b_min[0]
        py = point[1]*gridy+b_min[1]
        pz = point[2]*gridz+b_min[2]

        point1 = torch.FloatTensor([[px],[py],[pz]])

        point = point.unsqueeze(0)
        point1 = point1.unsqueeze(0)

        pp.append(point)
        pp1.append(point1)
        sdf1.append(sdf)


    point = torch.cat(pp, dim = 0)
    point1 = torch.cat(pp1, dim = 0)
    sdf = torch.cat(sdf1, dim = 0)



    with open(meta_path, 'rb') as f:
        lines = f.read().splitlines()

    for index in range(24):
        print(index)
        camera_param = lines[index]
        camera_param =camera_param.split()
        c = [float(i) for i in camera_param]
        camera_param = torch.FloatTensor(c)

        camera_param = camera_param.unsqueeze(0)



        project_point = project2d(point1, camera_param)
        project_point = project_point.int() #(2048, 2, 1)


        data = OrderedDict()

        p = torch.empty(2048, 3, 1).float()
        pp = torch.empty(2048, 2, 1)
        s = torch.empty(2048, 1)

        for j in range(2048):

            p[j, :, :] = point[j]
            pp[j, :, :] = project_point[j]
            s[j, :] = sdf[j]


        data = {'point':p, 'proj_point':pp, 'sdf': s}

        out_path = os.path.join(point_path[:-8], str("{:02d}".format(index)+'.npy'))
        #print(out_path)

        np.save(out_path, data)
