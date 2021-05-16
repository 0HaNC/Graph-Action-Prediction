"""
Created on Mar 07, 2019

@author: Wang Ruiqi

Description of the file.


version3.0

"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

import torch.nn as nn
#import torchvision.transforms as transforms
'''Lin commented on Sept. 1st
import pretrainedmodels
import pretrainedmodels.utils as utils
'''
import sklearn.metrics

#from tensorboardX import SummaryWriter
def masked_cross_entropy(logits, target, length):

    def sequence_mask(sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        # seq_range = torch.range(0, max_len - 1).long()
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        #print type(seq_length_expand)
        #print type(seq_range_expand)
        return seq_range_expand < seq_length_expand


    length = Variable(length)
    if torch.cuda.is_available():
        length = length.cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.contiguous().view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def test(epoch,dataloader, model, criterion1,criterion2,  opt,writer,log_file, is_test=True):
    model.eval()
    test_correct = np.zeros([1, opt.ratio_split_num])
    correct = np.zeros([1, opt.ratio_split_num])
    test_acc = np.zeros([1, opt.ratio_split_num])
    mask_acc = np.zeros([1, opt.ratio_split_num])
    test_loss= 0.
    output = np.zeros([len(dataloader.dataset),opt.ratio_split_num+1])
    index = 0
    Acc_Record = np.zeros((21, 11))
    VidClass_CNT = np.zeros((21, 1))
    # print ("dataloader size is ",len(dataloader))
    with torch.no_grad():
        '''
        for i, (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_ids, subject_ids) in enumerate(
                dataloader, 0):
        '''
        for i, (edge_features, node_features, adj_mat, graph_label,tem_features) in enumerate(dataloader, 0):
            batch = adj_mat.size(0)
            n_node = node_features.size(2)
            n_edge = edge_features.size(2)
            adj_matrixes = adj_mat.contiguous().view(batch * opt.seq_size, n_node, -1).float()
            #adj_matrixes = adj_mat.contiguous().view(batch*opt.seq_size, n_node, n_node*n_edge).float()  #for IGGNN ablation study
            init_input = node_features.contiguous().view(batch * opt.seq_size, n_node, -1).float()
            edge_states = edge_features.contiguous().view(batch * opt.seq_size, n_edge, -1).float()
            tem_features = tem_features.contiguous().view(batch , opt.seq_size, -1).float()
            graph_labels = graph_label[:, :, :-1]
            true_lengths = graph_label[:,-1:,-1].squeeze().float() #[batch,1]
           
            if opt.cuda:
                init_input = init_input.cuda()
                adj_matrixes = adj_matrixes.cuda()
                edge_states = edge_states.cuda()
                tem_features = tem_features.cuda()
                #relation_label = relation_label.cuda()
                labels = graph_labels.cuda()
                true_lengths = true_lengths.cuda()
                #node_labels = node_labels.cuda()

                # print "all the variable has been put into cuda"
            init_input = Variable(init_input)
            adj_matrixes = Variable(adj_matrixes)
            edge_states = Variable(edge_states)
            tem_features = Variable(tem_features)
            #relation_label = Variable(relation_label)
            labels = Variable(labels)
            true_lengths = Variable(true_lengths)
            #node_labels = Variable(node_labels)

            if opt.loss==1:
                pre_out, vis_loss = model(init_input, edge_states, adj_matrixes, true_lengths, tem_features,is_test=True)
            else:
                pre_out, vis_loss = model(init_input, edge_states, adj_matrixes, true_lengths,tem_features, is_test=True)
            #pre_out = model(init_input, edge_states)
            

            pre_label = torch.argmax(pre_out, dim=1)  # [batch*seq_size,1]
            gt_labels = torch.argmax(labels, dim=2)  # [batch,seq_size,1]

            # pre_relation_label = torch.argmax(relation_pre_out, dim=1)
            ##re_gt_labels = torch.argmax(relation_label, dim=1)  # [batch*seq_size,1]

            ##node_labels = torch.argmax(node_labels, dim=1) # [batch*seq_size,node_num,1]
            pre_label = pre_label.view(batch, opt.ratio_split_num, -1)
            gt_labels = gt_labels.view(batch, opt.seq_size, -1)
            gt_labels = gt_labels[:,:opt.ratio_split_num,:]


            '''
            for j in range(len(gt_labels)):
                labell = gt_labels[j][0][0]
                VidClass_CNT[labell, 0] += 1
                Acc_Record[labell, :-1] += (pre_label[j,:,0]==labell)
            '''



            #calculate the accuracy
            for seq_i in range(opt.ratio_split_num):
                correct[0, seq_i] = (pre_label[:, seq_i, 0] == gt_labels[:, seq_i, 0]).sum()
                test_correct[0, seq_i] = correct[0, seq_i] + test_correct[0, seq_i]


            #calculate the mask loss
            gt_labels = gt_labels.long().squeeze()
            pre_out = pre_out.view(batch, opt.ratio_split_num, -1).float()
            true_lengths = true_lengths.long()

            loss = masked_cross_entropy(pre_out,gt_labels,true_lengths)
            pre_label = pre_label.view(batch, opt.ratio_split_num).float()
            gt_labels = gt_labels[:,-1].view(batch, 1).float()
            #print pre_label.size()
            #print gt_labels.size()
            output[index:index+batch]=torch.cat((pre_label,gt_labels),1).cpu().numpy()
            index = index+batch
            
                #vis_loss = vis_loss.float()
                #loss = 2.0 * loss + 1.0 * vis_loss  # TODO need regularzation??
            loss =  opt.l1*loss +  opt.l2 * vis_loss  # TODO need regularzation??
     

            test_loss = test_loss + loss

        # Lin commented on Sept. 9
        # dataloader.dataset = dataset
        test_acc = test_correct / len(dataloader.dataset)
        test_loss = test_loss / len(dataloader)
        # Lin commented on Sept. 9
        # len(dataloader) = len(dataset) = num_video
        writer.add_scalar('test_loss', test_loss / len(dataloader), epoch)
        writer.add_scalar('test_acc_ratio0.1', test_acc[0][0], epoch)
        writer.add_scalar('test_acc_ratio0.2', test_acc[0][1], epoch)
        writer.add_scalar('test_acc_ratio0.3', test_acc[0][2], epoch)
        writer.add_scalar('test_acc_ratio0.4', test_acc[0][3], epoch)
        writer.add_scalar('test_acc_ratio0.5', test_acc[0][4], epoch)
        writer.add_scalar('test_acc_ratio0.6', test_acc[0][5], epoch)
        writer.add_scalar('test_acc_ratio0.7', test_acc[0][6], epoch)
        writer.add_scalar('test_acc_ratio0.8', test_acc[0][7], epoch)
        writer.add_scalar('test_acc_ratio0.9', test_acc[0][8], epoch)
        writer.add_scalar('test_acc_ratio1.0', test_acc[0][9], epoch)
      

        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print ("ratio=0.1, test Accuracy:   %.2f " % (100. * test_acc[0][0]))
        print ("ratio=0.3, test Accuracy:   %.2f " % (100. * test_acc[0][2]))
        print ("ratio=0.5, test Accuracy:   %.2f " % (100. * test_acc[0][4]))
        print ("ratio=0.7, test Accuracy:   %.2f " % (100. * test_acc[0][6]))
        print ("ratio=0.9, test Accuracy:   %.2f " % (100. * test_acc[0][8]))
        print ("ratio=1.0, test Accuracy:   %.2f " % (100. * test_acc[0][9]))
        print ("Average test loss: %.4f" % (test_loss))
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        file = open(log_file, 'a+')
        file.write("ratio=0.1, test Accuracy: %.2f \n" % (100. * test_acc[0][0]))
        file.write("ratio=0.2, test Accuracy: %.2f \n" % (100. * test_acc[0][1]))
        file.write("ratio=0.3, test Accuracy: %.2f \n" % (100. * test_acc[0][2]))
        file.write("ratio=0.4, test Accuracy: %.2f \n" % (100. * test_acc[0][3]))
        file.write("ratio=0.5, test Accuracy: %.2f \n" % (100. * test_acc[0][4]))
        file.write("ratio=0.6, test Accuracy: %.2f \n" % (100. * test_acc[0][5]))
        file.write("ratio=0.7, test Accuracy: %.2f \n" % (100. * test_acc[0][6]))
        file.write("ratio=0.8, test Accuracy: %.2f \n" % (100. * test_acc[0][7]))
        file.write("ratio=0.9, test Accuracy: %.2f \n" % (100. * test_acc[0][8]))
        file.write("ratio=1.0, test Accuracy: %.2f \n" % (100. * test_acc[0][9]))
        file.write("Average test loss: %.4f \n\n" % (test_loss))
        
        '''
        Acc_Record[:, :-1] = Acc_Record[:, :-1] / VidClass_CNT
        Acc_Record[:, -1] = Acc_Record[:, :-1].mean(-1)
        np.save(log_file[:-4]+'.npz', Acc_Record)
        np.savetxt(log_file[:-4]+'.csv', Acc_Record, delimiter=',')
        '''
    return test_acc,output
