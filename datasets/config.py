"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import errno
import logging
import os

#import config


class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        super(Paths, self).__init__()

        #self.detect_root1 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.5/'
        self.o1 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.5/'
        self.detect_root1 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.5-edge/'
        self.detect_root2 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.7/'
        self.detect_root3 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/nothresh/'
        self.feat_root= ''
        self.img_root = '/media/mcislab/new_disk/wangruiqi/data/something-something/20bn-something-something-v1/'
        
        self.detect_root_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101-detectron-result/thresh0.5-edge/'
        self.detect_root_ucf_simplified = '/media/mcislab/new_disk/wangruiqi/data/UCF101-detectron-result/thresh0.5-edge_simplified/'
        self.detect_root_ucf_mmdet = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101_mmdet_pickle/' #ucf101 detect 10 frames per video
        self.detect_root_ucfall_mmdet = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101All_mmdet_pickle/' #ucf101 detect all frames per video
        #self.detect_root_bit_mmdet = '/media/mcislab/new_disk/wangruiqi/data/bit/BIT_mmdet_pickle/'
        self.detect_root_bit_mmdet = '/home/mcislab/lty/media/others/wangruiqi/data/bit/BIT_mmdet_pickle_allPerson/'
        self.detect_root_jhmdb_mmdet = '/media/mcislab/new_disk/wangruiqi/data/JHMDB/JHMDB_mmdet_pickle/'
        self.detect_root_hmdb = '/home/mcislab/lty/media/others/wangruiqi/data/hmdb51/hmdb51-detectron/'
        self.img_root_ucf = '/home/mcislab/lty/media/others/wangruiqi/data/UCF101/UCF101_rawimage/'
        self.img_root_jhmdb = '/media/mcislab/new_disk/wangruiqi/data/JHMDB/rgb_frms/'
        self.img_root_bit = '/home/mcislab/lty/media/others/wangruiqi/data/bit/bit_frame/'
        self.img_root_hmdb = '/home/mcislab/lty/media/others/wangruiqi/data/hmdb51/hmdb51_frame/'
        self.rgb_resnext_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/3DResNext101Feat-V2/' #ucf101 3DResNext101
        self.rgb_bninc_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/BNINc_RGB_Feat_last2-V2/' #ucf101 BNInceptionV4-rgb
        self.rgb_res18_ucf = '/home/mcislab/lty/media/others/wangruiqi/data/UCF101/UCF101_Res18_Feat/' #ucf101 ResNet18-RGB
        self.flow_bninc_bit_v1 = '/home/mcislab/lty/media/others/wangruiqi/data/bit/BNINc_Flow_Feat_last2-V2/' #bit BNInceptionV4-flow last two layer,padding more 4 frames
        self.flow_bninc_bit_v2 = '/home/mcislab/lty/media/others/wangruiqi/data/bit/BNINc_Flow_Feat_last2-V2.1/' #bit BNInceptionV4-flow last two layer
        self.flow_bninc_bit_general = '/media/mcislab3d/ir4t/wangruiqi/data/' #BIT-flow
        self.flow_res18_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101_Res18_Feat_flow/flow_images/'#ucf101 ResNet18-flow
        self.flow_bninc_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/BNINc_Flow_Feat_last2-V2/' #ucf101 BNInceptionV4-flow
        self.avgflow_bninc_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/BNINc_Flow_Feat_AVG/' #ucf101 BNInceptionV4-flow avg pooling for each ratio
        self.rgb_resnext_hmdb = '/home/mcislab/lty/media/others/wangruiqi/data/hmdb51/3DResNext101Feat/' #HMDB51 3DResNext101
        self.rgb_resnext_sample20_hmdb = '/home/mcislab/lty/media/others/wangruiqi/data/hmdb51/ap_sample20/'
        self.rgb_resnext_sample30_hmdb = '/home/mcislab/lty/media/others/wangruiqi/data/hmdb51/ap_sample30/'
         
        self.bninception_ucf = '/media/mcislab3d/ir4t/wangruiqi/data/UCF101/ap_BNINc_sample10/' #ap_BNInception  UCF101
        self.ap_bninc_sample30_bit ='/home/mcislab/lty/media/others/wangruiqi/data/bit/ap_bninc_sample30/'
        self.ap_bninc_sample20_bit ='/home/mcislab/lty/media/others/wangruiqi/data/bit/ap_bninc_sample20/'
        self.ap_3dresnext_ucf = '/media/mcislab3d/ir4t/wangruiqi/data/UCF101/ap_resnext3d_sample10/' #3dresnext101 UCF101

def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
