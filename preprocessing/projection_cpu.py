import torch
import torch.nn as nn
import numpy as np
import time
import math
'''
from config import Struct, load_config, compose_config_str
config_dict = load_config(file_path='./config_sdfnet.yaml')
configs = Struct(**config_dict)
batch_size = configs.batch_size
'''


device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ="cpu"

def project2d(point, camera_param):
    batch_size = point.size(0)
    #print(batch_size)
    project_point = torch.empty(batch_size, 2, 1).to(device)
    cam_mat, cam_pos = camera_info(camera_param)
    for i in range(batch_size):
        if i%64 == 0:
            print(i)
        project_point[i,:,:] = project(i, point, cam_mat, cam_pos)

    return project_point

def camera_info(camera_param):

    param = camera_param[0]

    pi = math.pi

    theta = param[0]*pi/180
    phi = param[1]*pi/180

    camY = param[3] * torch.sin(phi)
    temp = param[3] * torch.cos(phi)

    camX = temp * torch.cos(theta)
    camZ = temp * torch.sin(theta)

    cam_pos = torch.stack((camX, camY, camZ), 0)

    axisZ = cam_pos.float()

    axisY = torch.tensor([0,1,0]).float().to(device)
    axisX = torch.cross(axisY, axisZ)
    axisY = torch.cross(axisZ, axisX)

    cam_mat = torch.stack((unit(axisX), unit(axisY), unit(axisZ)), 0)

    return cam_mat.float(), cam_pos.float()

def unit(v):
    # norm = np.linalg.norm(v)
    norm = torch.norm(v)
    if norm == 0:
        return v
    return v / norm



def project(i, point, cam_mat, cam_pos):
    point = torch.transpose(point[i], 0, 1)

    point = point * 0.57

    pt_trans = torch.mm(point-cam_pos, torch.t(cam_mat))

    # X,Y,Z = pt_trans.T
    X,Y,Z = torch.t(pt_trans)

    X = torch.unsqueeze(X, 0)
    Y = torch.unsqueeze(Y, 0)
    Z = torch.unsqueeze(Z, 0)

    F = 248
    h = (-Y)/(-Z)*F + 224/2.0
    w = X/(-Z)*F + 224/2.0

    # h = np.clip(h, 0, 223)
    # w = np.clip(w, 0, 223)
    h = torch.clamp(h, 0, 223)
    w = torch.clamp(w, 0, 223)

    # h = np.round(h).astype(int)
    # w = np.round(w).astype(int)
    h = torch.round(h).int()
    w = torch.round(w).int()

    # assert (w >= 0 and w < 224), "w is out of range!"
    # assert (h >= 0 and h < 224), "h is out of range!"

    # h = torch.from_numpy(h)
    # w = torch.from_numpy(w)
    out = torch.cat((w,h))

    return out




if __name__ == '__main__':
    print("HI")

    batch_size = 5 # do it for every image once!
    torch.manual_seed(123)

    point1 = torch.randn(batch_size, 3, 1).to(device)
    #point = torch.FloatTensor([[[-0.4230765],[-0.0604395],[-0.080586]],[[-0.4230765],[-0.0604395],[-0.080586]]])
    print(point1.shape)

    # camera_param = torch.FloatTensor([[293.5551607, 26.85558558, 0., 0.8375818, 25.], [137.30681486, 28.90141833, 0., 0.73950087, 25.]])
    camera_param = torch.FloatTensor([[293.5551607, 26.85558558, 0., 0.8375818, 25.]]).to(device)

    project_point = project2d(point1, camera_param)
    project_point = project_point.int()
    print(project_point)
    print(project_point.shape)
