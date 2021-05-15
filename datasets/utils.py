"""
Created on Jan 08, 2019

@author: Ruiqi Wang

Description of the file.
version 1.0


"""

import os
import shutil

import numpy as np
import torch

def collate_fn_cad(batch):  # copy feature at the end
    seq_size=10  #max is 23
    edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id = batch[0]
    #print node_labels.shape

    max_node_num = np.max(np.array([[adj_mat.shape[1]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_edge_num = np.max(np.array([[adj_mat.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))

    ## max_edge_num = np.max(np.array([[edge_features.shape[0]] for (edge_features, node_features, adj_mat, node_labels, graph_label, sequence_id) in batch]))
    max_subactivity_num = np.max(np.array([[relation_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_activity_num = np.max(np.array([[graph_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[2]
    node_label_dim = node_labels.ndim
    if node_label_dim > 1:
        node_label_len = node_labels.shape[2]
    del edge_features, node_features, adj_mat, node_labels

    #edge_features_batch = np.zeros((len(batch), edge_feature_len, max_node_num, max_node_num))
    edge_features_batch = np.zeros((len(batch), seq_size, max_edge_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), seq_size, max_node_num, node_feature_len))
    adj_mat_batch = np.zeros((len(batch), seq_size, max_node_num, max_edge_num))
    ##adj_mat_batch = np.zeros((len(batch), max_node_num, max_node_num*max_edge_num)) # for GGNN
    relation_label_batch = np.zeros((len(batch), seq_size, max_subactivity_num+1))
    graph_label_batch = np.zeros((len(batch), seq_size, max_activity_num+1))
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num))

    sequence_ids = list()
    subject_ids = list()
    node_nums = list()
    #count = 0
    for i, (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in enumerate(batch):
        true_length = adj_mat.shape[0]
        node_num = node_features.shape[1]
        edge_num = edge_features.shape[1]

        #print true_length
        ##edge_num = node_num*node_num  # for GGNN
        #print edge_num
        #print node_num
        # TODO code need to be shortened
        #if true_length==1:
        #    count = count+1
        #    continue
        
        if true_length< seq_size:
            edge_features_batch[i, :true_length, :edge_num, :] = edge_features
            node_features_batch[i, :true_length, :node_num, :] = node_features
            adj_mat_batch[i, :true_length, :node_num, :edge_num] = adj_mat
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN

            relation_label_batch[i,:true_length, :-1] = relation_label.squeeze()
            graph_label_batch[i,:true_length, :-1] = graph_label.squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, :true_length, :node_num, :] = node_labels
            else:
                node_labels_batch[i, :true_length, :node_num] = node_labels
           
            edge_features_batch[i, true_length:, :edge_num, :] = edge_features[-1]
            node_features_batch[i, true_length:, :node_num, :] = node_features[-1]
            adj_mat_batch[i, true_length:, :node_num, :edge_num] = adj_mat[-1]
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
            
            relation_label_batch[i,true_length:, :-1] = relation_label[-1].squeeze()
            graph_label_batch[i,true_length:, :-1] = graph_label[-1].squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, true_length:, :node_num, :] = node_labels[-1]
            else:
                node_labels_batch[i, true_length:, :node_num] = node_labels[-1]
            
        else:
            slt_seg = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
            for j in range(seq_size):
                edge_features_batch[i, j, :edge_num, :] = edge_features[int(slt_seg[j])-1]
                node_features_batch[i, j, :node_num, :] = node_features[int(slt_seg[j])-1]
                adj_mat_batch[i, j, :node_num, :edge_num] = adj_mat[int(slt_seg[j])-1]
                ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
                relation_label_batch[i, j, :-1] = relation_label[int(slt_seg[j])-1]
                graph_label_batch[i, j, :-1] = graph_label[int(slt_seg[j])-1]
                relation_label_batch[i, j, -1] = seq_size
                graph_label_batch[i, j, -1] = seq_size
                if node_label_dim > 1:
                    node_labels_batch[i, j, :node_num, :] = node_labels[int(slt_seg[j])-1]
                else:
                    node_labels_batch[i, j, :node_num] = node_labels[int(slt_seg[j])-1]

        
        sequence_ids.append(sequence_id)
        subject_ids.append(subject_id)
        node_nums.append(node_num)


    edge_features_batch = torch.DoubleTensor(edge_features_batch)
    node_features_batch = torch.DoubleTensor(node_features_batch)
    adj_mat_batch = torch.DoubleTensor(adj_mat_batch)
    node_labels_batch = torch.DoubleTensor(node_labels_batch)
    relation_label_batch = torch.DoubleTensor(relation_label_batch)
    graph_label_batch = torch.DoubleTensor(graph_label_batch)
    #print "%d video length is 1" % count
    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, relation_label_batch, graph_label_batch, sequence_ids, subject_ids


def collate_fn_cad_abl(batch):  # copy feature at the end
    seq_size=10  #max is 23
    edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id = batch[0]
    #print node_labels.shape

    max_node_num = np.max(np.array([[adj_mat.shape[1]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    #max_edge_num = np.max(np.array([[edge_features.shape[1]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_edge_num = max_node_num*max_node_num #for GGNN

    ## max_edge_num = np.max(np.array([[edge_features.shape[0]] for (edge_features, node_features, adj_mat, node_labels, graph_label, sequence_id) in batch]))
    max_subactivity_num = np.max(np.array([[relation_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_activity_num = np.max(np.array([[graph_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[2]
    node_label_dim = node_labels.ndim
    if node_label_dim > 1:
        node_label_len = node_labels.shape[2]
    del edge_features, node_features, adj_mat, node_labels

    #edge_features_batch = np.zeros((len(batch), edge_feature_len, max_node_num, max_node_num))
    edge_features_batch = np.zeros((len(batch), seq_size, max_edge_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), seq_size, max_node_num, node_feature_len))
    ##adj_mat_batch = np.zeros((len(batch), seq_size, max_node_num, max_edge_num))
    adj_mat_batch = np.zeros((len(batch),seq_size, max_node_num, max_node_num*max_edge_num)) # for GGNN
    relation_label_batch = np.zeros((len(batch), seq_size, max_subactivity_num+1))
    graph_label_batch = np.zeros((len(batch), seq_size, max_activity_num+1))
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num))

    sequence_ids = list()
    subject_ids = list()
    node_nums = list()
    #count = 0
    for i, (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in enumerate(batch):
        true_length = adj_mat.shape[0]
        node_num = node_features.shape[1]
        ##edge_num = edge_features.shape[1]

        #print true_length
        edge_num = node_num*node_num  # for GGNN
        #print edge_num
        #print node_num
        # TODO code need to be shortened
        #if true_length==1:
        #    count = count+1
        #    continue
        
        if true_length< seq_size:
            edge_features_batch[i, :true_length, :edge_num, :] = edge_features
            node_features_batch[i, :true_length, :node_num, :] = node_features
            ##adj_mat_batch[i, :true_length, :node_num, :edge_num] = adj_mat
            adj_mat_batch[i, :true_length, :node_num, :edge_num*node_num] = adj_mat  # for GGNN

            relation_label_batch[i,:true_length, :-1] = relation_label.squeeze()
            graph_label_batch[i,:true_length, :-1] = graph_label.squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, :true_length, :node_num, :] = node_labels
            else:
                node_labels_batch[i, :true_length, :node_num] = node_labels
           
            edge_features_batch[i, true_length:, :edge_num, :] = edge_features[-1]
            node_features_batch[i, true_length:, :node_num, :] = node_features[-1]
            adj_mat_batch[i, true_length:, :node_num, :edge_num*node_num] = adj_mat[-1]  # for GGNN
            
            
            relation_label_batch[i,true_length:, :-1] = relation_label[-1].squeeze()
            graph_label_batch[i,true_length:, :-1] = graph_label[-1].squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, true_length:, :node_num, :] = node_labels[-1]
            else:
                node_labels_batch[i, true_length:, :node_num] = node_labels[-1]
            
        else:
            slt_seg = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
            for j in range(seq_size):
                edge_features_batch[i, j, :edge_num, :] = edge_features[int(slt_seg[j])-1]
                node_features_batch[i, j, :node_num, :] = node_features[int(slt_seg[j])-1]
                ##adj_mat_batch[i, j, :node_num, :edge_num] = adj_mat[int(slt_seg[j])-1]
                adj_mat_batch[i,j, :node_num, :(edge_num*node_num)] = adj_mat[int(slt_seg[j])-1]  # for GGNN
                relation_label_batch[i, j, :-1] = relation_label[int(slt_seg[j])-1]
                graph_label_batch[i, j, :-1] = graph_label[int(slt_seg[j])-1]
                relation_label_batch[i, j, -1] = seq_size
                graph_label_batch[i, j, -1] = seq_size
                if node_label_dim > 1:
                    node_labels_batch[i, j, :node_num, :] = node_labels[int(slt_seg[j])-1]
                else:
                    node_labels_batch[i, j, :node_num] = node_labels[int(slt_seg[j])-1]

        
        sequence_ids.append(sequence_id)
        subject_ids.append(subject_id)
        node_nums.append(node_num)


    edge_features_batch = torch.DoubleTensor(edge_features_batch)
    node_features_batch = torch.DoubleTensor(node_features_batch)
    adj_mat_batch = torch.DoubleTensor(adj_mat_batch)
    node_labels_batch = torch.DoubleTensor(node_labels_batch)
    relation_label_batch = torch.DoubleTensor(relation_label_batch)
    graph_label_batch = torch.DoubleTensor(graph_label_batch)
    #print "%d video length is 1" % count
    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, relation_label_batch, graph_label_batch, sequence_ids, subject_ids    

def collate_fn_cad1(batch):  #copy feature in the middle
    seq_size=10
    edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id = batch[0]
    #print node_labels.shape

    max_node_num = np.max(np.array([[adj_mat.shape[1]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_edge_num = np.max(np.array([[adj_mat.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))

    ## max_edge_num = np.max(np.array([[edge_features.shape[0]] for (edge_features, node_features, adj_mat, node_labels, graph_label, sequence_id) in batch]))
    max_subactivity_num = np.max(np.array([[relation_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_activity_num = np.max(np.array([[graph_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[2]
    node_label_dim = node_labels.ndim
    if node_label_dim > 1:
        node_label_len = node_labels.shape[2]
    del edge_features, node_features, adj_mat, node_labels

    #edge_features_batch = np.zeros((len(batch), edge_feature_len, max_node_num, max_node_num))
    edge_features_batch = np.zeros((len(batch), seq_size, max_edge_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), seq_size, max_node_num, node_feature_len))
    adj_mat_batch = np.zeros((len(batch), seq_size, max_node_num, max_edge_num))
    ##adj_mat_batch = np.zeros((len(batch), max_node_num, max_node_num*max_edge_num)) # for GGNN
    relation_label_batch = np.zeros((len(batch), seq_size, max_subactivity_num+1))
    graph_label_batch = np.zeros((len(batch), seq_size, max_activity_num+1))
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num))

    sequence_ids = list()
    subject_ids = list()
    node_nums = list()
    #count = 0
    for i, (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in enumerate(batch):
        true_length = adj_mat.shape[0]
        node_num = adj_mat.shape[1]
        edge_num = adj_mat.shape[2]

        #print true_length
        ##edge_num = node_num*node_num  # for GGNN
        #print edge_num
        #print node_num
        # TODO code need to be shortened
        #if true_length==1:
        #    count = count+1
        #    continue
        '''
        if true_length< seq_size:
            edge_features_batch[i, :true_length, :edge_num, :] = edge_features
            node_features_batch[i, :true_length, :node_num, :] = node_features
            adj_mat_batch[i, :true_length, :node_num, :edge_num] = adj_mat
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN

            relation_label_batch[i,:true_length, :-1] = relation_label.squeeze()
            graph_label_batch[i,:true_length, :-1] = graph_label.squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, :true_length, :node_num, :] = node_labels
            else:
                node_labels_batch[i, :true_length, :node_num] = node_labels

            edge_features_batch[i, true_length:, :edge_num, :] = edge_features[-1]
            node_features_batch[i, true_length:, :node_num, :] = node_features[-1]
            adj_mat_batch[i, true_length:, :node_num, :edge_num] = adj_mat[-1]
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN

            relation_label_batch[i,true_length:, :-1] = relation_label[-1].squeeze()
            graph_label_batch[i,true_length:, :-1] = graph_label[-1].squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, true_length:, :node_num, :] = node_labels[-1]
            else:
                node_labels_batch[i, true_length:, :node_num] = node_labels[-1]

        else:
            slt_seg = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
            for j in range(seq_size):
                edge_features_batch[i, j, :edge_num, :] = edge_features[int(slt_seg[j])-1]
                node_features_batch[i, j, :node_num, :] = node_features[int(slt_seg[j])-1]
                adj_mat_batch[i, j, :node_num, :edge_num] = adj_mat[int(slt_seg[j])-1]
                ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
                relation_label_batch[i, j, :-1] = relation_label[int(slt_seg[j])-1]
                graph_label_batch[i, j, :-1] = graph_label[int(slt_seg[j])-1]
                relation_label_batch[i, j, -1] = seq_size
                graph_label_batch[i, j, -1] = seq_size
                if node_label_dim > 1:
                    node_labels_batch[i, j, :node_num, :] = node_labels[int(slt_seg[j])-1]
                else:
                    node_labels_batch[i, j, :node_num] = node_labels[int(slt_seg[j])-1]
        '''
        if true_length< seq_size:
           
            index_primal = np.arange(1.0, true_length + 1, 1.0)
            new_index = np.round(index_primal / true_length * 10)
            slt_frm_index = new_index - 1
            slt_frm_index = slt_frm_index.astype(np.int32)
            #print slt_frm_index
            if slt_frm_index[0]!=0:
                #print "ind0=%d"%slt_frm_index[0]
                edge_features_mask = np.ones((1, slt_frm_index[0], edge_num, edge_feature_len ))*edge_features[0]
                node_features_mask = np.ones((1, slt_frm_index[0], node_num, node_feature_len ))*node_features[0]
                adj_mat_mask = np.ones((1, slt_frm_index[0], node_num, edge_num ))*adj_mat[0]
                relation_label_mask = np.ones((1, slt_frm_index[0], max_subactivity_num))*relation_label[0].squeeze()
                graph_label_mask = np.ones((1, slt_frm_index[0], max_activity_num))*graph_label[0].squeeze()
                #print edge_features_mask.shape
                #print node_features_mask.shape
                edge_features_batch[i, :slt_frm_index[0], :edge_num, :] = edge_features_mask
                node_features_batch[i, :slt_frm_index[0], :node_num, :] = node_features_mask
                adj_mat_batch[i, :slt_frm_index[0],:node_num, :edge_num] = adj_mat_mask
                relation_label_batch[i, :slt_frm_index[0], :-1] = relation_label_mask
                graph_label_batch[i, :slt_frm_index[0], :-1] = graph_label_mask
                if node_label_dim > 1:
                    node_labels_mask = np.ones((1, slt_frm_index[0], node_num, node_labels.shape[2])) * node_labels[0]
                    node_labels_batch[i, :slt_frm_index[0], :node_num, :] = node_labels_mask
                else:
                    node_labels_mask = np.ones((1, slt_frm_index[0], node_num))*node_labels[0]
                    node_labels_batch[i, :slt_frm_index[0], :node_num] = node_labels_mask

            for ind in range(len(slt_frm_index) - 1):
                #print "ind=%d"%slt_frm_index[ind]

                if (slt_frm_index[ind+1]-slt_frm_index[ind]-1)==0:
                    continue
                edge_features_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], edge_num, edge_feature_len )) * edge_features[ind]
                node_features_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], node_num, node_feature_len )) * node_features[ind]
                adj_mat_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], node_num, edge_num)) * adj_mat[ind]
                relation_label_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], max_subactivity_num)) * relation_label[
                    ind].squeeze()
                graph_label_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], max_activity_num)) * graph_label[ind].squeeze()
                #print slt_frm_index[ind+1]-slt_frm_index[ind]
                #print edge_features_mask.shape
                #print node_features_mask.shape
                edge_features_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :edge_num, :] = edge_features_mask
                node_features_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :node_num, :] = node_features_mask
                adj_mat_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :node_num, :edge_num] = adj_mat_mask
                relation_label_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :-1] = relation_label_mask
                graph_label_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :-1] = graph_label_mask
                if node_label_dim > 1:
                    node_labels_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], node_num, node_labels.shape[2])) * node_labels[
                        ind]
                    node_labels_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :node_num, :] = node_labels_mask
                else:
                    node_labels_mask = np.ones((1, slt_frm_index[ind+1]-slt_frm_index[ind], node_num)) * node_labels[ind]
                    node_labels_batch[i, slt_frm_index[ind]:slt_frm_index[ind+1], :node_num] = node_labels_mask

            relation_label_batch[i, :, -1] = seq_size
            graph_label_batch[i, :, -1] = seq_size

        else:
            slt_seg = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
            for j in range(seq_size):
                edge_features_batch[i, j, :edge_num, :] = edge_features[int(slt_seg[j])-1]
                node_features_batch[i, j, :node_num, :] = node_features[int(slt_seg[j])-1]
                adj_mat_batch[i, j, :node_num, :edge_num] = adj_mat[int(slt_seg[j])-1]
                ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
                relation_label_batch[i, j, :-1] = relation_label[int(slt_seg[j])-1]
                graph_label_batch[i, j, :-1] = graph_label[int(slt_seg[j])-1]
                relation_label_batch[i, j, -1] = seq_size
                graph_label_batch[i, j, -1] = seq_size
                if node_label_dim > 1:
                    node_labels_batch[i, j, :node_num, :] = node_labels[int(slt_seg[j])-1]
                else:
                    node_labels_batch[i, j, :node_num] = node_labels[int(slt_seg[j])-1]
        
        sequence_ids.append(sequence_id)
        subject_ids.append(subject_id)
        node_nums.append(node_num)


    edge_features_batch = torch.DoubleTensor(edge_features_batch)
    node_features_batch = torch.DoubleTensor(node_features_batch)
    adj_mat_batch = torch.DoubleTensor(adj_mat_batch)
    node_labels_batch = torch.DoubleTensor(node_labels_batch)
    relation_label_batch = torch.DoubleTensor(relation_label_batch)
    graph_label_batch = torch.DoubleTensor(graph_label_batch)
    #print "%d video length is 1" % count
    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, relation_label_batch, graph_label_batch, sequence_ids, subject_ids

def collate_fn_cad2(batch):  # do not copy feature
    seq_size=10
    edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id = batch[0]
    #print node_labels.shape

    max_node_num = np.max(np.array([[adj_mat.shape[1]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_edge_num = np.max(np.array([[adj_mat.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))

    ## max_edge_num = np.max(np.array([[edge_features.shape[0]] for (edge_features, node_features, adj_mat, node_labels, graph_label, sequence_id) in batch]))
    max_subactivity_num = np.max(np.array([[relation_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    max_activity_num = np.max(np.array([[graph_label.shape[2]] for (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in batch]))
    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[2]
    node_label_dim = node_labels.ndim
    if node_label_dim > 1:
        node_label_len = node_labels.shape[2]
    del edge_features, node_features, adj_mat, node_labels

    #edge_features_batch = np.zeros((len(batch), edge_feature_len, max_node_num, max_node_num))
    edge_features_batch = np.zeros((len(batch), seq_size, max_edge_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), seq_size, max_node_num, node_feature_len))
    adj_mat_batch = np.zeros((len(batch), seq_size, max_node_num, max_edge_num))
    ##adj_mat_batch = np.zeros((len(batch), max_node_num, max_node_num*max_edge_num)) # for GGNN
    relation_label_batch = np.zeros((len(batch), seq_size, max_subactivity_num+1))
    graph_label_batch = np.zeros((len(batch), seq_size, max_activity_num+1))
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), seq_size, max_node_num))

    sequence_ids = list()
    subject_ids = list()
    node_nums = list()
    #count = 0
    for i, (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_id, subject_id) in enumerate(batch):
        true_length = adj_mat.shape[0]
        node_num = adj_mat.shape[1]
        edge_num = adj_mat.shape[2]

        #print true_length
        ##edge_num = node_num*node_num  # for GGNN
        #print edge_num
        #print node_num
        # TODO code need to be shortened
        #if true_length==1:
        #    count = count+1
        #    continue
        
        if true_length< seq_size:
            edge_features_batch[i, :true_length, :edge_num, :] = edge_features
            node_features_batch[i, :true_length, :node_num, :] = node_features
            adj_mat_batch[i, :true_length, :node_num, :edge_num] = adj_mat
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN

            relation_label_batch[i,:true_length, :-1] = relation_label.squeeze()
            graph_label_batch[i,:true_length, :-1] = graph_label.squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, :true_length, :node_num, :] = node_labels
            else:
                node_labels_batch[i, :true_length, :node_num] = node_labels
            '''
            edge_features_batch[i, true_length:, :edge_num, :] = edge_features[-1]
            node_features_batch[i, true_length:, :node_num, :] = node_features[-1]
            adj_mat_batch[i, true_length:, :node_num, :edge_num] = adj_mat[-1]
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
            
            relation_label_batch[i,true_length:, :-1] = relation_label[-1].squeeze()
            graph_label_batch[i,true_length:, :-1] = graph_label[-1].squeeze()
            relation_label_batch[i, :, -1] = true_length
            graph_label_batch[i, :, -1] = true_length
            if node_label_dim > 1:
                node_labels_batch[i, true_length:, :node_num, :] = node_labels[-1]
            else:
                node_labels_batch[i, true_length:, :node_num] = node_labels[-1]
            '''
        else:
            slt_seg = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
            for j in range(seq_size):
                edge_features_batch[i, j, :edge_num, :] = edge_features[int(slt_seg[j])-1]
                node_features_batch[i, j, :node_num, :] = node_features[int(slt_seg[j])-1]
                adj_mat_batch[i, j, :node_num, :edge_num] = adj_mat[int(slt_seg[j])-1]
                ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
                relation_label_batch[i, j, :-1] = relation_label[int(slt_seg[j])-1]
                graph_label_batch[i, j, :-1] = graph_label[int(slt_seg[j])-1]
                relation_label_batch[i, j, -1] = seq_size
                graph_label_batch[i, j, -1] = seq_size
                if node_label_dim > 1:
                    node_labels_batch[i, j, :node_num, :] = node_labels[int(slt_seg[j])-1]
                else:
                    node_labels_batch[i, j, :node_num] = node_labels[int(slt_seg[j])-1]

        
        sequence_ids.append(sequence_id)
        subject_ids.append(subject_id)
        node_nums.append(node_num)


    edge_features_batch = torch.DoubleTensor(edge_features_batch)
    node_features_batch = torch.DoubleTensor(node_features_batch)
    adj_mat_batch = torch.DoubleTensor(adj_mat_batch)
    node_labels_batch = torch.DoubleTensor(node_labels_batch)
    relation_label_batch = torch.DoubleTensor(relation_label_batch)
    graph_label_batch = torch.DoubleTensor(graph_label_batch)
    #print "%d video length is 1" % count
    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, relation_label_batch, graph_label_batch, sequence_ids, subject_ids

def collate_fn_sth(batch):  # copy feature at the end,for something-something
    seq_size=10  
    edge_features, node_features, adj_mat, graph_label = batch[0]
    max_node_num = np.max(np.array([[adj_mat.shape[1]] for (edge_features, node_features, adj_mat, graph_label) in batch]))
    max_edge_num = np.max(np.array([[adj_mat.shape[2]] for (edge_features, node_features, adj_mat, graph_label) in batch]))
    
    
    #temp_list = [[graph_label.shape[2]] for (edge_features, node_features, adj_mat, graph_label) in batch]
    # Lin changed '2' to '1', added '- 1' on Sept. 2nd
    max_activity_num = np.max(np.array([[graph_label.shape[1]] for (edge_features, node_features, adj_mat, graph_label) in batch])) - 1
    
    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[2]
   
    del edge_features, node_features, adj_mat

    #edge_features_batch = np.zeros((len(batch), edge_feature_len, max_node_num, max_node_num))
    edge_features_batch = np.zeros((len(batch), seq_size, max_edge_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), seq_size, max_node_num, node_feature_len))
    adj_mat_batch = np.zeros((len(batch), seq_size, max_node_num, max_edge_num))
    graph_label_batch = np.zeros((len(batch), seq_size, max_activity_num+1))

    #count = 0
    for i, (edge_features, node_features, adj_mat, graph_label) in enumerate(batch):
        true_length = adj_mat.shape[0]
        node_num = adj_mat.shape[1]
        edge_num = adj_mat.shape[2]

        #print true_length
        ##edge_num = node_num*node_num  # for GGNN
        #print edge_num
        #print node_num
        # TODO code need to be shortened
        #if true_length==1:
        #    count = count+1
        #    continue
        
        if true_length < seq_size:
            edge_features_batch[i, :true_length, :edge_num, :] = edge_features
            node_features_batch[i, :true_length, :node_num, :] = node_features
            adj_mat_batch[i, :true_length, :node_num, :edge_num] = adj_mat
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN

            graph_label_batch[i,:true_length, :-1] = graph_label.squeeze()
            graph_label_batch[i, :, -1] = true_length
            
            edge_features_batch[i, true_length:, :edge_num, :] = edge_features[-1]
            node_features_batch[i, true_length:, :node_num, :] = node_features[-1]
            adj_mat_batch[i, true_length:, :node_num, :edge_num] = adj_mat[-1]
            ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
            
            graph_label_batch[i,true_length:, :-1] = graph_label[-1].squeeze()
            graph_label_batch[i, :, -1] = true_length
    # Lin added on Sept. 2nd

        else:
            slt_seg = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
            for j in range(seq_size):
                edge_features_batch[i, j, :edge_num, :] = edge_features[int(slt_seg[j])-1]
                node_features_batch[i, j, :node_num, :] = node_features[int(slt_seg[j])-1]
                adj_mat_batch[i, j, :node_num, :edge_num] = adj_mat[int(slt_seg[j])-1]
                ##adj_mat_batch[i, :node_num, :(edge_num*node_num)] = adj_mat  # for GGNN
                # Lin added '[:-1]' on Sept. 2nd
                graph_label_batch[i, j, :-1] = graph_label[int(slt_seg[j])-1][:-1]
                graph_label_batch[i, j, -1] = seq_size
    #Lin changed 'double' to 'float' on Sept. 2nd
    edge_features_batch = torch.FloatTensor(edge_features_batch)
    node_features_batch = torch.FloatTensor(node_features_batch)
    adj_mat_batch = torch.FloatTensor(adj_mat_batch)
    graph_label_batch = torch.FloatTensor(graph_label_batch)
    
    #print "%d video length is 1" % count
    return edge_features_batch, node_features_batch, adj_mat_batch, graph_label_batch

def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    #epoch = state['epoch']
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)

def load_best_checkpoint(args, model, optimizer,flag=True):
    '''
    args:
    flag--whether to change the optimizer
    '''
    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            #best_epoch_error = checkpoint['best_epoch_error']
            '''
            try:
                avg_epoch_error = checkpoint['avg_epoch_error']
            except KeyError:
                avg_epoch_error = np.inf
            '''
            model.load_state_dict(checkpoint['state_dict'])
            if flag:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cuda:
                model.cuda()
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            return args, model, optimizer
            #return args, model
        else:
            print("=> no best model found at '{}'".format(best_model_file))
    return None

def main():
    pass


if __name__ == '__main__':
    main()

