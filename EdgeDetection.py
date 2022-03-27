##############
#computer the edge points according to the supplyment
#1. knn points 2. knn center 3. distance between p and knn center 4.judge
##############
import torch
import h5py
import numpy as np
from knn_cuda import KNN
from matplotlib import pyplot as plt

def polt(origin_pc, edge_pc):
    
    elev = 90  # which height to view the surface
    azim = 180-(-0 + 0)  # angle of rotation
    cmap=['Blues']
    size = 5
    xlim=(-0.32, 0.32)
    ylim=(-0.32, 0.32)
    zlim=(-0.32, 0.32)

    pcd = origin_pc
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.view_init(elev, azim)
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=-pcd[:, 0], s=size, cmap=cmap[0], vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    pcd = edge_pc    
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.view_init(elev, azim)
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=-pcd[:, 0], s=size, cmap=cmap[0], vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.show()



def EdgeDetection(x, k=150, beta=1.8):
    knn_cuda = KNN(k=k, transpose_mode=False)
    
    x = torch.from_numpy(x).unsqueeze(0).cuda()
    x = x.transpose(1,2).contiguous()
    
    batch_size, _, num_points = x.size()
    _, idx = knn_cuda(x, x)  # bs k np
    assert idx.shape[1] == k
    idx = idx.transpose(1,2).contiguous()
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    num_dims = x.size(1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # (b,n,k,c)
    center = torch.mean(feature, dim=2, keepdim=False) #(b,n,3)
    distance = torch.norm((x-center), dim=-1) #(b,n)
    x_k = x.unsqueeze(2).repeat(1, 1, k, 1)
    x_knn_distance = torch.norm((x_k-feature), dim=-1)
    min_distance, idx = torch.kthvalue(x_knn_distance,2,dim=-1)
    idx = distance > (beta * min_distance)
    edge_points = x[idx,:]
    return edge_points


file_path = '/home/zzy/Desktop/Completion/MVP_Train_CP.h5'
input_file = h5py.File(file_path, 'r')
input_data = np.array(input_file['incomplete_pcds'][()])
gt_data = np.array(input_file['complete_pcds'][()])
labels = np.array(input_file['labels'][()])


x = gt_data[17000//26,:,:] # N,3
edge_points = EdgeDetection(x)
edge_points = edge_points.cpu().numpy() # M,3
polt(x, edge_points)