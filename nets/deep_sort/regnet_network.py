import tensorflow as tf
from nets.keras.common.blocks import StemBlock
from nets.keras.backbone.regnet.body import RegNetXBody
from nets.keras.head.head import Feature_Head
import numpy as np

def RegNetX(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage, groups_per_stage,
            activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    body_out = RegNetXBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                           groups_per_stage, activation, weight_decay)
    return body_out


def regnet_model(input_shape, num_classes=None, weight_decay=1e-8):
    input_tensor = tf.keras.layers.Input(input_shape)
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                         weight_decay)
    print("===========STEM : ", stem)
    # regnetx_200MF = RegNetX(stem, [1, 1, 4, 7], # stage : 4개, block 반복수 : 1, 1, 4, 7
    #                         [[24, 24, 24], [56, 56, 56], [152, 152, 152], [368, 368, 368]],  # block 1개당 filter 갯수 (무조건 3개)
    #                         [(3, 3), (3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2), (2, 2)], [8, 8, 8, 8], "relu",  # stage 들어갈때마다 stride(2, 2) 적용
    #                         weight_decay)
    
    regnetx_200MF = RegNetX(stem, [1, 1, 3], # stage : 3개, block 반복수 : 1, 1, 3
                            [[32, 32, 32], [64, 64, 64], [128, 128, 128]],  # block 1개당 filter 갯수 (무조건 3개)
                            [(3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2)], [1, 1, 1], "relu",  # stage 들어갈때마다 stride(2, 2) 적용
                            weight_decay)
    print("===========BACKBONE : ", regnetx_200MF[-1].shape[-1])
    feature_shape = (regnetx_200MF[-1].shape[-1], num_classes)
    feature, logits = Feature_Head(regnetx_200MF[-1], feature_shape, weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=logits)
    
    return model

if __name__=="__main__":
    input_shape = (256, 128, 3)
    num_class = 1500 + 1 # max labels : 1500 and -1 -> 1501
    weight_decay = 1e-8
    model = regnet_model(input_shape=input_shape, num_classes=num_class, weight_decay=weight_decay)
    model.summary()
    model.save("./t.h5")
    
