"""
Created on Jun 19, 2019

@author: Wang Ruiqi

Description of the file.
adding vlad to represent temporal variation


version4.0

"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gc
import random
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import math
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import itertools
import copy


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Semantic(nn.Module):
    
    #Semantic Transformation Mmodule  for CAD120
    #Implementation based on https://arxiv.org/abs/1811.10696
    
    def __init__(self, opt):
        super(Semantic, self).__init__()
        self.loss=0.
     
        self.state_dim = opt.state_dim
        self.seq_size = opt.seq_size
        self.projection_layer1 = nn.Linear(self.state_dim, 300)
        self.projection_layer2 = nn.Linear(self.state_dim, 300)
        self.projection_layer3 = nn.Linear(self.state_dim, 300)
        self.embedding = nn.Sequential(
            nn.Linear(900, 500),
            nn.ReLU(inplace=True)
        )
        self.vis_loss = nn.MSELoss()
    def forward(self, nodes_feats, edges_feats, adj_mat, true_lengths):
        # Lin added 'torch.set_grad_enabled' on Sept. 3rd
        with torch.set_grad_enabled(False):
            #vis_trans = []
            
            losses =[]
            
            batseq, node_num, node_feat_dim =nodes_feats.size()
            _, edge_num, edge_feat_dim =edges_feats.size()
            nodes_feats = nodes_feats.view(-1, self.seq_size, node_num, node_feat_dim)
            edges_feats = edges_feats.view(-1, self.seq_size, edge_num, edge_feat_dim)
            adj_mat = adj_mat.view(-1, self.seq_size, node_num, edge_num)
            batch, _, _, _ = nodes_feats.size()

            for batch_ind in range(batch):
                
                true_length = true_lengths[batch_ind]
                # Lin added on Sept. 3rd
                true_length = int(true_length.cpu().numpy())
                for seq_i in range(true_length):
                    vis_tran = []
                    relation_num = torch.sum(torch.eq(adj_mat[batch_ind,seq_i, :, :], 1))
                    n_node = len(nodes_feats[batch_ind, seq_i, :, :])

                    if relation_num == 0:
                        continue
                    #print adj_mat[batch_ind, seq_i, :, :]
                    # modify
                    index1, index2 = np.argwhere((adj_mat[batch_ind, seq_i, :, :] == 1).cpu())
                   
                    node_feat = nodes_feats[batch_ind, seq_i, :, :]
                    edge_feat = edges_feats[batch_ind, seq_i, :, :]
                  
                   
                    for i in range(relation_num):
                     
                        obj1_feat = node_feat[index1[i], :]
                        f_i = self.projection_layer1(obj1_feat)
                        rela_feat = edge_feat[index2[i], :]
                        f_ij = self.projection_layer2(rela_feat)
                  
                        obj2_feat = node_feat[index2[i]-(index1[i]*n_node), :]

                        f_j =self.projection_layer3(obj2_feat)
                        f_i = torch.unsqueeze(f_i,0)
                        f_ij = torch.unsqueeze(f_ij,0)
                        f_j = torch.unsqueeze(f_j,0)
                        #tmp1 = torch.cat((f_i,f_ij),1)
                        #f_ij_cap = self.embedding(torch.cat((tmp1,f_j),1))
                        #f_ij_cap = torch.squeeze(f_ij_cap)
                        #vis_tran.append(f_ij_cap)
                        #print self.vis_loss(f_j,(f_ij+f_i)).data[0]
                        # Lin changed '.data[0]' to '.item()'
                        losses.append(self.vis_loss(f_j,(f_ij+f_i)).item())
                        #if self.vis_loss(f_j,(f_ij+f_i)).data[0]>0:
                        #    self.loss += self.vis_loss(f_j,(f_ij+f_i)).data[0]
                      
                    #print loss
                    #print vis_tran.size()
                #vis_tran = torch.stack(vis_tran)
                #print vis_tran.size()
                #vis_trans.append(vis_tran)
            #losses = np.array(losses)
            #print losses
            #num = len(np.argwhere(losses > 0.))
            ##loss = losses.sum()/num
            losses = np.array(losses)
            num = len(np.argwhere(losses > 0.))
            loss = losses.sum()/num
            ##self.loss  = self.loss /batch
            #print self.loss 
            ##tor_loss = torch.from_numpy(loss)
            #print loss.dtype
            for x in locals().keys():
                del locals()[x]
            gc.collect()
        return loss



class TemporalRelationGraph(torch.nn.Module):
    def __init__(self, opt):
        super(TemporalRelationGraph, self).__init__()
        
        self.featdir = opt.featdir
        self.seq_size = opt.seq_size
        self.ratio_split_num = opt.ratio_split_num
        self.belta = opt.belta
        self.subsample_num = opt.subsample_num # how many relations selected to sum up
        self.state_dim = opt.state_dim
        self.num_bottleneck = opt.num_bottleneck
        # Lin change 0->1 on 10.7
        self.scales = [i for i in range(opt.num_frames, 0, -1)] # generate the multiple frame relations
        self.K_clusters = opt.K_clusters
        self.seq_size = opt.seq_size
        self.num_class = opt.classNum
        self.num_frames = opt.num_frames
        self.num_bottleneck = opt.num_bottleneck
        self.d_pos = opt.d_pos
        self.cuda = opt.cuda
        self.hidden_size = opt.hidden_size
        self.rnn_num_layer = opt.rnn_num_layer
        self.rnn = nn.LSTM(self.num_bottleneck, self.hidden_size, self.rnn_num_layer, batch_first=True)
        self.classifier = nn.Sequential(
                        nn.Linear(self.hidden_size, self.state_dim),
                        nn.Tanh(),
                        nn.Linear(self.state_dim,self.num_class)
                        )
          # for action prediction, each step need select frames
        self.relations_scales = []
    
        self.subsample_scales = []
        # Lin added on Sept. 9
        if len(self.scales) == 0:
            self.scales = [1]
        #if len(self.scales) == 1:
        #    self.scales = [2,1]
        for ratio in range(1,self.ratio_split_num+1):
            # 0~4, 0~5, ..., 0~14
            relations_scales_tmp = []
            subsample_scales_tmp = []
            seq = int(ratio*2) # calculate seq: for BIT, sequence length is 2 times than the number of total observation ratios
            for scale in self.scales:
                
                #relations_scale = self.return_relationset(self.num_frames+seq, scale)
                relations_scale = self.return_relationset(seq, min(scale,seq))
               
                relations_scales_tmp.append(relations_scale)
                subsample_scales_tmp.append(min(self.subsample_num, len(relations_scale)))
            self.relations_scales.append(relations_scales_tmp)
            self.subsample_scales.append(subsample_scales_tmp)# how many samples of relation to select in each forward pass

        
        self.attention =  nn.Sequential(
                        nn.Linear(((self.K_clusters+1)*self.state_dim), 1),
                        #nn.Linear(self.state_dim, 1),
                        nn.Sigmoid()
                        )
        
        self.fc_fusion_scales = nn.ModuleList() 
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Linear(scale * ((self.K_clusters+1)*self.state_dim), self.num_bottleneck),
                        #nn.Linear(scale * self.state_dim, self.num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(self.num_bottleneck,self.num_bottleneck),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.num_bottleneck, self.num_class)
                        )
            self.fc_fusion_scales += [fc_fusion]
            #self.attention_scales += [attention]
            #self.classifier_scales += [classifier]
    
    def feat_mode(self):
        self.fc_fusion_scales_forfeat = copy.deepcopy(self.fc_fusion_scales)
        for i in range(len(self.scales)):
            self.fc_fusion_scales_forfeat[i] = self.fc_fusion_scales[i][:-1]

    def forward(self, graph_out, init_input, is_test, groundtruth=None):
        batch,_,_ = graph_out.size()
        # grapg_out.size() = (batch, seq_size, state_dim)
        # graph representation: g_l
        batch_variation = self.temporal_variation(init_input) 
        # object feature: O_l
        # shape=(batch, seq_size, n_cluster, state_dim)
        batch_variation = batch_variation.view(batch, self.seq_size, -1)
        
        # shape=(batch, seq_size, n_cluster*state_dim)
        if self.cuda:
            batch_variation = batch_variation.cuda()  
        new_input = torch.cat((batch_variation, graph_out), dim=2)
        # size=(batch, seq_size, (n_cluster+1)*state_dim)
        
        if self.cuda: 
            #result = torch.zeros((batch,self.seq_size,self.num_class)).cuda() # for seq=ratio_num
            result = torch.zeros((batch,self.ratio_split_num,self.num_class)).cuda() #for seq != ratio_num
        else:
            #result = torch.zeros((batch,self.seq_size,self.num_class))
            result = torch.zeros((batch,self.ratio_split_num,self.num_class))

        ##new_input_con = temporalvar[:, :1, :].repeat(1, self.num_frames, 1)
        '''
        # Padding
        # size=(batch, num_frames, 1)
        pos_input = get_sinusoid_encoding_table(self.seq_size+self.num_frames, self.d_pos).repeat(new_input_con.size(0), 1, 1)
        # position information feature: pho_l
        # tensor size=(batch, num_frames+seq_size, d_pos)=(batch, 14, 256)
        if self.cuda:
            pos_input = pos_input.cuda()
        else:
            pos_input = pos_input
        '''
        ##new_input = torch.cat((new_input_con, temporalvar), dim=1)
        # (batch, num_frames+seq_size, (n_cluster+1)*state_dim)
        ##new_input = torch.cat((new_input, pos_input), dim=2)
        # (batch, num_frames+seq_size, (n_cluster+1)*state_dim + d_pos)
        ##new_input = new_input+pos_input
        rnn_input = []
        for ratio in range(1, self.ratio_split_num+1):
            seq_ind = int(ratio*2)
            #*****adjust scales with the increasing input******#
            if is_test:
                new_scales = self.adjust_scale_test(seq_ind,self.num_frames,self.scales)
            else:
                new_scales = self.adjust_scale_train(seq_ind,self.num_frames,self.scales)

                
            if self.cuda:
                act_all = torch.zeros((batch,self.num_class)).cuda()
            else:
                act_all = torch.zeros((batch,self.num_class))

            ##new_input__ = new_input[:, :(seq_ind+self.num_frames), :]
            new_input__ = new_input[:, :seq_ind, :]

            ##act_all = new_input__[:, self.relations_scales[seq_ind-1][0][0], :]
        
            ##act_all = act_all.view(act_all.size(0), self.scales[0] * new_input.size(-1)) # [batch,  num_frames+seq_size * (n_cluster+1)*state_dim + d_pos)]
   
            ##act_all = self.fc_fusion_scales[0](act_all)  #[batch, num_class]  
      
       
            ##for scaleID in range(len(self.scales)):# for scale 1
            for scaleID in range(len(new_scales)):# for scale 1

                scale_index = self.scales.index(new_scales[scaleID])
                idx_relations_randomsample = self.select_idx(self.relations_scales[ratio-1][scale_index],new_scales[scaleID],seq_ind)
                

                '''
                # iterate over the scales
                # Lin added if on Sept. 10
                if is_test:#for test, select equal distance frames
                    idx_relations_randomsample = self.select_idx(self.relations_scales[seq_ind-1][scaleID], self.scales[scaleID], self.num_frames+seq_ind-1)
                    #idx_relations_randomsample = np.random.choice(len(self.relations_scales[seq_ind-1][scaleID]), self.subsample_scales[seq_ind-1][scaleID], replace=False)
                else:
                    #idx_relations_randomsample = self.select_idx(self.relations_scales[seq_ind-1][scaleID],self.scales[scaleID],self.num_frames+seq_ind-1)
                    idx_relations_randomsample = np.random.choice(len(self.relations_scales[seq_ind-1][scaleID]), self.subsample_scales[seq_ind-1][scaleID], replace=False)   
                
                ##if self.cuda:
                ##    act_all_scale = torch.zeros((batch,self.num_bottleneck)).double().cuda()
                ##else:
                ##    act_all_scale = torch.zeros((batch,self.num_bottleneck)).double()

                # random sample subsample_scales[seq_ind-1][scaleID] frames, in relations_scales[seq_ind-1][scaleID]
                
                print(self.scales[scaleID], self.relations_scales[idx_relations_randomsample[0]])
                continue
                '''
                for idx in idx_relations_randomsample:
                    ##act_relation = new_input__[:, self.relations_scales[seq_ind-1][scaleID][idx], :]
                    act_relation = new_input__[:, self.relations_scales[ratio-1][scale_index][idx], :]
                    #act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.state_dim)
                    ##atten = self.attention(act_relation) 
                    ##act_relation = atten*act_relation
                    ##act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * new_input.size(-1))
                    act_relation = act_relation.view(act_relation.size(0), -1)
                    #act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] *  self.state_dim)
                    '''
                    ##save the feature ##
                    if self.featdir and seq_ind==5:
                        out = self.fc_fusion_scales_forfeat[scaleID](act_relation).cpu().numpy()
                        for b in range(batch):
                            if groundtruth[b,seq_ind].item() not in {7,6,5,10,11}:
                                continue
                            #print(groundtruth[b,seq_ind])
                            save_dir = os.path.join(self.featdir, 'scale{}'.format(self.scales[scaleID]), str(groundtruth[b,0].item()))
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            ind = len(os.listdir(save_dir))
                            np.save(os.path.join(save_dir, str(ind)), out[b])
                    '''
                    ##act_relation = self.fc_fusion_scales[scaleID](act_relation)  #[batch,num_class]
                    if is_test:
                        #print (act_relation.size())
                        act_relation = self.fc_fusion_scales[scale_index](act_relation)  #[batch,num_class]
                    else:                   
                        act_relation = self.fc_fusion_scales[scale_index](act_relation)  #[batch,num_class]
                    
                    act_all += act_relation
                    ##act_all_scale+=act_relation
                
                ##act_all_scale = act_all_scale/len(idx_relations_randomsample)
     
                ##act_all_seq += act_all_scale
            
            ##act_all_seq = act_all_seq/len(self.scales)
            #result[:, seq_ind-1, :] = act_all #for seq=10
            result[:, ratio-1, :] = act_all #for BIT, seq=20,ratio_num=10

            ##rnn_input.append(act_all_seq)
        ##rnn_input=torch.stack(rnn_input,dim=1)
        ##output,_ = self.rnn(rnn_input)
        ##output = self.classifier(output)
        ##output = output.view(batch*self.seq_size,-1)


        '''select 20 frames by directly add action probability score'''
        '''
        index = np.array(np.linspace(0, self.seq_size-1+1e-3, self.ratio_split_num), dtype=np.int)
        for ratio in range(self.ratio_split_num-1):
            final_result[:,ratio,:] = torch.add(result[:,index[ratio],:], 1, result[:,index[ratio]+1,:])
        final_result[:,self.ratio_split_num-1,:]=torch.add(result[:,(index[self.ratio_split_num-1]-1),:],1,result[:,index[self.ratio_split_num-1],:])
        '''
        #return result.view(batch*self.seq_size, -1)##output#result.view(batch*self.seq_size,-1)
        return result.view(batch*self.ratio_split_num, -1)##output#result.view(batch*self.seq_size,-1)

    def temporal_variation(self, init_input):
        #using VLAD to represent the temporal variation of each frame
        #batch,_,_ = init_input.size()  
        # Fix n_cluster to '3'
        n_cluster = self.K_clusters
        cluster_inputs = init_input.view(-1, self.seq_size, init_input.size(1), init_input.size(-1))
        cluster_input = cluster_inputs.view(-1, self.seq_size*init_input.size(1), init_input.size(-1))
        # size = (batch, seq_size*n_nodes, state_dim)
        
        object_centers = []
        temporal_variation = []
        batch_variation = torch.zeros((cluster_inputs.size(0), self.seq_size, n_cluster, init_input.size(2)))
        for batch_ind in range(cluster_input.size(0)):
            # to cpu, again??
            cluster_input_ = cluster_input[batch_ind].cpu().detach().numpy()
            initial_centers = kmeans_plusplus_initializer(cluster_input_, n_cluster).initialize()
            kmeans_instance = kmeans(cluster_input_, initial_centers)
            kmeans_instance.process()
            #clusters = kmeans_instance.get_clusters()
            # 'centers' is a list of ndarr
            centers = kmeans_instance.get_centers()
            # Lin commented on Sept. 3rd
            #n_cluster =len(centers)
            #assert len(centers)==n_cluster, '{} centers'.format(len(centers))
            ex = torch.zeros((cluster_input.size(1), len(centers)))
            for i in range(len(centers)):
                centers_i = np.tile(np.array(centers[i]), cluster_input.size(1)).reshape(cluster_input.size(1), -1)
               # print np.array(centers[i]).reshape(1,cluster_input.size(2)).shape
                ex[:, i] = -self.belta*pow(np.linalg.norm((cluster_input_ - centers_i), ord=2), 2)   #||x-d_i||_2^2
            #print clusters
            #object_centers.append(centers)
            w = F.softmax(ex, dim=1)  #the normalized weight of descriptor x with respect to codeword d_i
            w = w.reshape(self.seq_size, -1, len(centers))  #(seq,n_node,n_cluster)
            x = cluster_input_.reshape(self.seq_size, -1, cluster_input.size(-1)) #(seq,n_node,nodde_state_dim)
        
            S_seq = []
            # Video O_l
            for seq_ind in range(self.seq_size):
                S = []
                # Object Feature: O_l
                for i in range(len(centers)): 
                    center_i = np.tile(np.array(centers[i]), cluster_inputs.size(2)).reshape(cluster_inputs.size(2), -1) #(n_node,node_state_dim)
                    w_i =  np.tile(w[seq_ind,:,i].reshape(-1,1), cluster_inputs.size(-1))
                    # wi: (n_nodes, state_dim), weight corresponding to cluster_i
                    center_varaiation = np.sum(np.multiply(w_i, (x[seq_ind,:,:]-center_i)), axis=0)
                    S.append(center_varaiation)
                S_seq.append(np.array(S))
            batch_variation[batch_ind, :, :len(centers), :] = torch.from_numpy(np.array(S_seq))
            #               (batch,   seq, n_cluster, state_dim)
        return batch_variation

    def temporal_variation_test(self, init_input):
        #using VLAD to represent the temporal variation of each frame
        #batch,_,_ = init_input.size()  
        # Fix n_cluster to '3'
        n_cluster = self.K_clusters
        cluster_inputs = init_input.view(-1, self.seq_size, init_input.size(1), init_input.size(-1))
        cluster_input = cluster_inputs.view(-1, self.seq_size*init_input.size(1), init_input.size(-1))
        # size = (batch, seq_size*n_nodes, state_dim)

        object_centers = []
        temporal_variation = []
        batch_variation = torch.zeros((cluster_inputs.size(0), self.seq_size, n_cluster, init_input.size(2)))
        for batch_ind in range(cluster_input.size(0)):
            # to cpu, again??
            cluster_input_ = cluster_input[batch_ind].cpu().detach().numpy()
            initial_centers = kmeans_plusplus_initializer(cluster_input_, n_cluster).initialize()
            kmeans_instance = kmeans(cluster_input_, initial_centers)
            kmeans_instance.process()
            #clusters = kmeans_instance.get_clusters()
            # 'centers' is a list of ndarr
            centers = kmeans_instance.get_centers()
            # Lin commented on Sept. 3rd
            #n_cluster =len(centers)
            #assert len(centers)==n_cluster, '{} centers'.format(len(centers))
            ex = torch.zeros((cluster_input.size(1), len(centers)))
            for i in range(len(centers)):
                centers_i = np.tile(np.array(centers[i]), cluster_input.size(1)).reshape(cluster_input.size(1), -1)
               # print np.array(centers[i]).reshape(1,cluster_input.size(2)).shape
                ex[:, i] = -self.belta*pow(np.linalg.norm((cluster_input_ - centers_i), ord=2), 2)   #||x-d_i||_2^2
            #print clusters
            #object_centers.append(centers)
            w = F.softmax(ex, dim=1)  #the normalized weight of descriptor x with respect to codeword d_i
            w = w.reshape(self.seq_size, -1, len(centers))  #(seq,n_node,n_cluster)
            x = cluster_input_.reshape(self.seq_size, -1, cluster_input.size(-1)) #(seq,n_node,nodde_state_dim)
        
            S_seq = []
            # Video O_l
            for seq_ind in range(self.seq_size):
                S = []
                # Object Feature: O_l
                for i in range(len(centers)): 
                    center_i = np.tile(np.array(centers[i]), cluster_inputs.size(2)).reshape(cluster_inputs.size(2), -1) #(n_node,node_state_dim)
                    w_i =  np.tile(w[seq_ind,:,i].reshape(-1,1), cluster_inputs.size(-1))
                    # wi: (n_nodes, state_dim), weight corresponding to cluster_i
                    center_varaiation = np.sum(np.multiply(w_i, (x[seq_ind,:,:]-center_i)), axis=0)
                    S.append(center_varaiation)
                S_seq.append(np.array(S))
            batch_variation[batch_ind, :, :len(centers), :] = torch.from_numpy(np.array(S_seq))
            #               (batch,   seq, n_cluster, state_dim)
        return batch_variation

    def return_relationset(self, num_frames, num_frames_relation):  #num_frames_relation is scale
        '''select video frame in order'''
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

    def select_idx(self, relations_scales, scale, num_frames):
        def is_Arithmetic_progression(arr,distance):
            arr_dis= np.diff(arr)
            if (arr_dis == distance).all():
                return True
            else:
                return False
        idx = list()
        idx_count=0
        division = math.floor(num_frames/scale)
        for relations_scale in relations_scales:

            if is_Arithmetic_progression(relations_scale,division):
                idx.append(idx_count)
            idx_count += 1
        return idx


    def return_relationset_random(self, num_frames, num_frames_relation):  
        '''select video frame in order'''
        import itertools
        permutations=list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
        new_permutations=list()
        #print type(permutations[0])
        for i in range(len(permutations)):
            
            tmp = list(permutations[i])
            random.shuffle(tmp)
            tmp = tuple(tmp)
            new_permutations.append(tmp)
      
        return new_permutations

    def adjust_scale_train(self,observe_ratio,scale_max,scales):
        '''adjust scale selection with the increasing video length '''
        if observe_ratio<scale_max:
            return scales[-min(5,observe_ratio):]
        else:
            scale_min_num = min(5,observe_ratio)
            new_scales = scales[:scale_min_num+1]
            return new_scales

    def adjust_scale_test(self,num_frames,scale_max,scales):
        '''adjust scale selection with the increasing video length '''
        if num_frames<scale_max:
            return scales[-num_frames:]
        else:
            scale_min_num = min(5,num_frames)
            new_scales = scales[:scale_min_num+1]
            return new_scales



class Residual_Block(nn.Module):
    def __init__(self, state_dim,node_state_dim,edge_state_dim):
        super(Residual_Block, self).__init__()
        self.node_state_dim = node_state_dim
        self.edge_feat_dim = edge_state_dim
        self.state_dim = state_dim
        #self.linear_node = nn.Linear(self.node_state_dim, self.state_dim )
        #self.linear_edge = nn.Linear(self.edge_feat_dim, self.state_dim )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, graph_out, init_input,edge_state):
       
        #node_states = self.linear_node(init_input)
        #edge_states = self.linear_edge(edge_state)
        feat_input = torch.cat((init_input,edge_state),1)
        residual = torch.mean(feat_input, dim=1)

        out = graph_out+residual
        out = self.relu(out)
        return out          

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()  #TODO maybe it should add BN layer if feature change in the future
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self, state_edge, state_cur, A):
        #print A.size()
        a_cur = torch.bmm(A, state_edge)  #[n_node,state_dim]
        #print ("a_cur:", a_cur.size())
        a = torch.cat((a_cur, state_cur), 2)
        #print ("a:", a.size())
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_cur, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat  #  [batch_size, n_node, state_dim]

        return output
'''
class RelationLoss(nn.Module):
    def __init__(self, obj1_feat, obj2_feat, edge_feat):
        super(RelationLoss, self).__init__()
        print '1'
        self.obj1_feat = obj1_feat
        self.obj2_feat = obj2_feat
        self.edge_feat = edge_feat
    def forward(self, pred, truth):
        return  torch.mean(torch.mean((pred-truth)**2,1),0)
'''
class SpatialRelation(nn.Module):
    """
    modified from GGNN, add external edge states
    Mode: Graph-level output
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(SpatialRelation, self).__init__()

        #assert (opt.state_dim >= opt.annotation_dim,  \
        #        'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.node_feat_dim = opt.node_feat_dim
        self.edge_feat_dim = opt.edge_feat_dim
        #self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        #for i in range(self.n_edge_types):
            # edge embedding--undirected edge

        #    link_fc = nn.Linear(self.edge_feat_dim, self.state_dim)
        #    self.add_module("link_{}".format(i), link_fc)

        #self.link_fcs = AttrProxy(self, "link_")
        self.link_fc = nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)
        self.residual_block = Residual_Block(self.state_dim, self.node_feat_dim, self.edge_feat_dim)

        # Output Model  Using soft attention mechanism to decide which nodes are relevant to the task
        self.attention = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim,1)
        )

        self.out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            #nn.Linear(self.state_dim, 1)
        )

        self.result = nn.Tanh()

        self.linear_node = nn.Linear(self.node_feat_dim, self.state_dim )
        self.linear_edge = nn.Linear(self.edge_feat_dim, self.state_dim )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, prop_state, edge_states, A):  # prop_State:[batch*seq_size,n_nodes,state_dim]

        init_node_states = prop_state
        init_edge_states = edge_states
       
        for i_step in range(self.n_steps):
            #edge_states = []
            #print ("initial prop_state size ", prop_state.size())
            #for i in range(len(edge_states)):
            #    edge_states[i,:]=self.link_fcs(edge_states)

            #edge_states = torch.stack(edge_states).transpose(0, 1).contiguous() #[batch*seq_size,n_edge_types,n_nodes,state_dim]
            #edge_states = edge_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            # comment on 11.1
            #edge_states = self.link_fc(edge_states)
            # replace on 11.1
            edge_states = self.link_fc(edge_states)
            prop_state = self.propogator(edge_states, prop_state, A)

        #join_state = torch.cat((prop_state, annotation), 2)
        atten = self.attention(prop_state)  # [batch_size, n_node, 1]
        out = self.out(prop_state)
        A = torch.transpose(atten, 2, 1)  # [batch,1,n_node]
        A = F.softmax(A,dim = 2) #Attention on each node
        mul = torch.bmm(A,out)
        w_sum = torch.squeeze(mul)
        #mul = atten * out
        #output = output.sum(2)
        #w_sum = torch.sum(mul, dim=1)
        #w_sum = w_sum.view(-1, self.state_dim)
        graph_out = self.result(w_sum)
        res = self.residual_block(graph_out, init_node_states, init_edge_states)

        return res, out, edge_states
    '''
    def residual_block(self,graph_out,init_node,init_edge):
        node_states = self.linear_node(init_input)
        edge_states = self.linear_edge(edge_state)
        feat_input = torch.cat((node_states,edge_states),1)
        residual = torch.mean(feat_input, dim=1)
        #print residual.size()
        #print graph_out.size()
        out = graph_out+residual
        out = self.relu(out)
        return out   
    '''

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.seq_size = opt.seq_size
        self.hidden_size = opt.hidden_size
        self.n_class = opt.classNum
        self.n_rclass = opt.rclassNum
        self.n_node_class = opt.nodeclassNum
        self.rnn_num_layer = opt.rnn_num_layer
        self.state_dim = opt.state_dim
        self.num_bottleneck = opt.num_bottleneck
        self.node_feat_dim = opt.node_feat_dim
        self.edge_feat_dim = opt.edge_feat_dim
        self.num_layers = opt.rnn_num_layer

        self.graph_net = SpatialRelation(opt)
        self.spatial_relation = Semantic(opt)
        #self.temporal_relation = TemporalRelation_naive(opt)
        #self.temporal_relation = TemporalRelation(opt)
        self.temporal_relation = TemporalRelationGraph(opt)
        self.residual_block = Residual_Block(self.state_dim, self.node_feat_dim, self.edge_feat_dim)

        #self.avg_pooling = nn.AvgPool1d(3,stride=1)
        self.linear_node = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.state_dim),
            #nn.ReLU()
        )
        self.linear_edge = nn.Sequential(
            nn.Linear(self.edge_feat_dim, self.state_dim),
            nn.ReLU(inplace=True)
        )
       
        
        self.classifier =nn.Sequential (
            nn.Linear(self.hidden_size, self.state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.state_dim, self.n_class),

        )
        
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def feat_mode(self):
        self.temporal_relation.feat_mode()

    def forward(self, init_input, edge_states, A, true_lengths, is_test, groundtruth=None): # init_input:[batch*seq,n_nodes,state_dim]
        batch, _, _ = init_input.size()
      
        init_input_ = self.linear_node(init_input)
        edge_states_ = self.linear_edge(edge_states)
        #print "Node and edge have been tranfered to the same size"
        graph_output, nodes_out, edges_out = self.graph_net(init_input_, edge_states_, A)
        # Lin: graph_output.shape = (batch*seq, state_dim)
        #print "Spatial relation reasoning finished."
        graph_output = graph_output.contiguous().view(-1, self.seq_size, self.state_dim)  # [batch,seq,n_nodes]
        vis_loss = self.spatial_relation(nodes_out, edges_out, A, true_lengths)
        #print "Visual semantic loss calculation finished."
        graph_outputs = graph_output.view(batch, -1)  # if residual before lstm it should be done
        #output = self.residual_block(graph_output,init_input,edge_states)
        output = graph_outputs.view(-1, self.seq_size, self.state_dim)
        # Tempo relation Net reads 'graph_output' and 'node_feat_input(dim-256)'
        output = self.temporal_relation(output, init_input_, is_test, groundtruth)
        '''
        #avg pooling
        mask = torch.ones(output.size()).double().cuda()
        #print mask
        output_ = torch.cumsum(output,dim=1)
        mask = torch.cumsum(mask,dim=1)
        #print mask
        output = output_ / mask
        '''
        '''
        bs_seq,_ = output.size()
        
        pre_out = list()
        ##relation_pre_out = list()
        for i in range(bs_seq):
            pre = self.classifier(output[i])
            ##re_pre = self.classifier_r(relation_state[i])
            pre_out.append(pre)
            ##relation_pre_out.append(re_pre)
        pre_out = torch.stack(pre_out)
        ##relation_pre_out = torch.stack(relation_pre_out)
        #print self.classfierNode(nodes_out).size()
        '''
        return output, vis_loss##, relation_pre_out#, self.classfierNode(nodes_out)



'''
if __name__ == "__main__":
    batch_size = 24
    seq_size = 10
    num_frames = 5
    num_class = 10
    img_feature_dim = 256
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of rnn')
    parser.add_argument('--seq_size', type=int, default=10, help='sequence length of rnn')
    parser.add_argument('--rnn_num_layer', type=int, default=1, help='layer number of rnn')
    parser.add_argument('--classNum', type=int, default=13, help='number of classes')
    parser.add_argument('--rclassNum', type=int, default=10, help='number of relation classes')
    parser.add_argument('--nodeclassNum', type=int, default=12, help='number of classes')
    #parser.add_argument('--n_nodes', type=int, default=3, help='number of nodes in graph')
    parser.add_argument('--n_type_node', type=int, default=6, help='#objects+human')
    parser.add_argument('--node_feat_dim', type=int, default=250, help='node state size')
    parser.add_argument('--edge_feat_dim', type=int, default=250, help='edge state size')
    parser.add_argument('--state_dim', type=int, default=256, help='dim of annotation')
    parser.add_argument('--num_bottleneck', type=int, default=56, help='dim of temporal reasoning module')
    parser.add_argument('--num_frames', type=int, default=4, help='number of sampled frames in each segment ')
    parser.add_argument('--n_steps', type=int, default=1, help='propogation steps number of GGNN')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    opt = parser.parse_args()
    opt.cuda = True
    input_var = Variable(torch.randn(opt.batch_size,opt.seq_size, opt.state_dim))
    node_input = Variable(torch.randn(opt.batch_size*opt.seq_size,opt.n_type_node,opt.node_feat_dim))
    edge_input = Variable(torch.randn(opt.batch_size*opt.seq_size,  opt.n_type_node*opt.n_type_node,opt.edge_feat_dim))
    adj_mat = Variable(torch.randn(opt.batch_size*opt.seq_size, opt.n_type_node, opt.n_type_node*opt.n_type_node))
    true_lengths =  Variable(torch.randn(opt.batch_size,1))
    model = Model(opt)

    output,vis_loss = model(node_input,edge_input,adj_mat,true_lengths)
    print(output)

'''
