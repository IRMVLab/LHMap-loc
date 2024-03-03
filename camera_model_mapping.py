# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import numpy as np
import torch


class CameraModel:

    def __init__(self, focal_length=None, principal_point=None):
        self.focal_length = focal_length
        self.principal_point = principal_point

    def project_pytorch(self, xyz: torch.Tensor, pcl: torch.Tensor, image_size, reflectance=None):
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                raise TypeError("Wrong Coordinates")

        if pcl.shape[0] == 3:
            pcl = torch.cat([pcl, torch.ones(1, pcl.shape[1], device=pcl.device)])
        else:
            if not torch.all(pcl[3, :] == 1.):
                pcl[3, :] = 1.
                raise TypeError("Wrong Coordinates")
    
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :] # lidar_coor -> camera_coor
        pclw = pcl[order, :]
        indexes = xyzw[2, :] >= 0 
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]
        pclw = pclw[:, indexes]
        uv = torch.zeros((4, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        uv[2, :] = self.focal_length[0] * pclw[0, :] / pclw[2, :] + self.principal_point[0]
        uv[3, :] = self.focal_length[1] * pclw[1, :] / pclw[2, :] + self.principal_point[1]
        indexes = (uv[0, :] >= 0.1) & (uv[2, :] >= 0.1)
        indexes = indexes & (uv[1, :] >= 0.1) & (uv[3, :] >= 0.1)
        indexes = indexes & (uv[0,:] < image_size[1]) & (uv[2,:] < image_size[1])
        indexes = indexes & (uv[1,:] < image_size[0]) & (uv[3,:] < image_size[0])
        if reflectance is None:
            uv = uv[:2, indexes], uv[2:, indexes], xyzw[2, indexes], pclw[2, indexes], pclw[0, indexes], pclw[1, indexes], None
        else:
            uv = uv[:2, indexes], uv[2:, indexes], xyzw[2, indexes], pclw[2, indexes], pclw[0, indexes], pclw[1, indexes], reflectance[:, indexes]

        return uv
