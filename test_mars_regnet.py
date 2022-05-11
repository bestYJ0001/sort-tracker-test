# vim: expandtab:ts=4:sw=4
import argparse
import os
import tensorflow as tf
import nets.deep_sort.regnet_network as regnet
from train.Generator import MarsGenerator
import glob
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("GPU found" if tf.test.gpu_device_name() else "No GPU found")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

IMAGE_SHAPE = (256, 128, 3)
DATASET_NAME = "MARS"
MODEL_NAME = "REGNET-S"
TRAIN_EPOCH = 100
NUM_CLASS = 1500+1

def argmax_pred(labels, pred):
    # return tf.keras.metrics.Accuracy().update_state(y_true=labels, y_pred=tf.argmax(pred, 1)).result()
    if int(tf.argmax(pred, 1)) is int(labels):
        print("::::::::::::: argmax : ", tf.argmax(pred, 1), " label : ", labels)
        return 1
    else:
        return 0
    return 0

def main():
    arg_parser = argparse.ArgumentParser(description="Metric trainer (%s)")
    arg_parser.add_argument(
        "--model_path", help="train or eval",
        default="D:/NOVATEK_1222/python/git-source/sort-tracker-test/saved_models/MARSREGNET-Sfinal.h5")
    arg_parser.add_argument(
        "--data_path", help="train or eval",
        default="D:/NOVATEK_1222/dataset/mars_dataset/bbox_train_one/0019/0019C1T0001F001.jpg")
    arg_parser.add_argument(
        "--run_id", required=True,
        help="please "
    )
    args = arg_parser.parse_args()
    
    weight_decay = 1e-8
    
    model = regnet.regnet_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASS, weight_decay=weight_decay)
    model.load_weights(args.model_path)
    model.summary()
    datasets = glob.glob(args.data_path)
    for dataset in datasets:
        print(dataset)
        img = cv2.imread(dataset, cv2.IMREAD_COLOR)
        in_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)/255., axis = 0)
        output = model.predict(in_img)
        print(np.max(output), np.argmax(output), output[0][np.argmax(output)])
    
    
    

if __name__ == "__main__":
    main()
