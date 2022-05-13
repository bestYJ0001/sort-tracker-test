from tkinter import W
from turtle import shape
from cv2 import normalize
from matplotlib.pyplot import sca
import tensorboard
import tensorflow as tf
from nets.keras.backbone.regnet.regnet import RegNetY
from nets.keras.neck.neck import FPN
from nets.keras.common.layers import Conv2dBnAct
from nets.keras.common.blocks import StemBlock

def Classification_Head(in_tensor, num_classes, weight_decay):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor = tf.keras.layers.GlobalAveragePooling2D()(in_tensor)
    out_tensor = tf.keras.layers.Reshape((1, 1, -1))(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                                           kernel_regularizer=kernel_regularizer)(out_tensor)
    out_tensor = tf.keras.layers.Dropout(0.2)(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                           kernel_regularizer=kernel_regularizer, activation='softmax')(out_tensor)
    out_tensor = tf.keras.layers.Reshape((-1,))(out_tensor)
    return out_tensor

def Feature_Head(in_tensor, out_feature, weight_decay): # out_feature : batch x last_filter_shape x num_ids(1500 + 1 negative)    
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor = tf.keras.layers.Reshape((1, 1, -1))(in_tensor)
    out_tensor = tf.keras.layers.Dropout(0.6)(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=out_feature[1], kernel_size=(1, 1), strides=(1, 1),
                                           kernel_regularizer=kernel_regularizer)(out_tensor)
    
    out_tensor = tf.keras.layers.Reshape((-1,))(out_tensor)
    feature = tf.keras.layers.BatchNormalization(axis=1)(out_tensor)

    # Is Train only
    if False:
        weight_initializer = tf.random_normal_initializer(stddev=1e-3)
        scale_initializer = tf.constant_initializer(0.)
        weight = tf.Variable(weight_initializer(shape=(out_feature[0], out_feature[1]), dtype=tf.float32), name = "mean_vectors")
        scale = tf.Variable(scale_initializer(shape=(), dtype=tf.float32), name="scale")
        scale = tf.keras.regularizers.l2(1e-1)(scale)
        scale = tf.nn.softplus(scale)
        # Mean vectors in colums, normalize axis 0.
        weight = tf.keras.layers.BatchNormalization(axis=0)(weight)
        logits = scale * tf.matmul(feature, weight)
        print("feature : ", feature.shape, "logits : ", logits.shape, "weights : ", weight.shape, "scale : ", scale.shape)
        
    return feature

def Yolo_Head(in_tensor_list, activation, num_classes, n_anchor, weight_decay):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor_list = []
    for index, in_tensor in enumerate(in_tensor_list):
        out_tensor = in_tensor
        b, h, w, c = out_tensor.shape
        out_tensor = Conv2dBnAct(out_tensor, c, (3, 3), (1, 1), activation=activation, weight_decay=weight_decay)
        out_tensor = tf.keras.layers.Conv2D(n_anchor * (5 + num_classes), (1, 1), (1, 1), activation="linear", kernel_regularizer=kernel_regularizer)(out_tensor)
        out_tensor = tf.keras.layers.Reshape((h, w, n_anchor, 5 + num_classes), name="Yolo_{}".format(index))(out_tensor)
        out_tensor_list.append(out_tensor)
    return out_tensor_list

if __name__ == "__main__":
    input_tensor = tf.keras.layers.Input((416, 416, 3))
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                         1e-5)
    backbone = RegNetY(
        stem,
        [1, 1, 1],
        [[128, 128, 128], [256, 256, 256], [512, 512, 512]],
        [(3, 3), (3, 3), (3, 3)],
        [(2, 2), (2, 2), (2, 2)],
        [32, 32, 32],
        "relu",
        1e-5
    )
    fpn = FPN(backbone, "relu", 1e-5, "add")
    head = Yolo_Head(fpn, "relu", 80, 3, 1e-5)
    model = tf.keras.Model(inputs=[input_tensor], outputs=head)
    model.summary()
    model.save("test.h5")
