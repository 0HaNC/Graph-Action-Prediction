"""
Created on May 31 , 2019

@author: Wang Ruiqi

edge feature: the combined bbox of two nodes

Description of the file.
node feature and edge feature both add to the graph
return scene feature


version3.0

"""

import os
import sys
import time
import pickle
import argparse

import math
import numpy as np
from PIL import Image
from collections import defaultdict

from torch.utils.data import Dataset

import torch
import torch.nn as nn
from torchvision import models
#import torchvision.transforms as transforms
from torch.autograd import Variable
'''Lin commented on Sept. 1st
import pretrainedmodels
import pretrainedmodels.utils as utils
'''
import h5py
import pandas as pd
'''Lin commented on Sept. 1st
import config
'''



def get_feature(feat_path):

    
    feat_category = feat_path.split('/')[-3]


    '''3DResNext101 avg feature'''
    if  feat_category=='ap_sample20' or feat_category=='ap_sample30':
        file = open(feat_path[:-4]+'.pkl','rb')
        feat_info = pickle.load(file)
        adj_matrixes = feat_info['adj_mat']
        nodes_features = feat_info['node_features']
        edges_features = feat_info['edge_features']
        scene_features = feat_info['tem_features']
        labels = feat_info['graph_label']

        return adj_matrixes,nodes_features,edges_features,scene_features,labels
    return feature


    
class dataset(Dataset):
    features = None

    def __init__(self, datalist, video_feat_path, opt):
        self.data = datalist
           
        feats_dir = []

        # video_lines
        # Lin changed 'rb' to 'r' on Sept. 1st
        lines = open(datalist, 'r')
        for line in lines:
            line = line.strip()
            video_name, video_label, video_length = line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2])
            video_frame_path = video_feat_path + video_name
            
            feats_dir.append(video_frame_path) 
            
            
            #imgFeat_list_.append(img_feat_path)
        self.feats_dirs = feats_dir
        


    def __getitem__(self, index):
        #line = self.data[index].strip()
        featInd = self.feats_dirs[index]
        
        adj_matrixes,nodes_features,edges_features,scene_features,labels = get_feature(featInd)
        
        #adj_matrixes = np.array(adj_matrixes, dtype=np.int32)
        #nodes_features = np.array(nodes_features, dtype=np.float32)
        #edges_features = np.array(edges_features, dtype=np.float32)
        #scene_features = np.array(scene_features, dtype=np.float32)
        #labels = np.array(labels, dtype=np.int32)
        #tem_features =np.array(tem_features, dtype=np.float32)
      
        #return imgFeats, res3dFeats, adj_matrixes, annotations,labels
        ##return nodes_feature, edge_feature, adj_matrixes,labels

        return edges_features, nodes_features, adj_matrixes, labels, scene_features

    def __len__(self):
        return len(self.feats_dirs)

