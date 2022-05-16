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

IMAGE_SHAPE = (128, 64, 3)
DATASET_NAME = "MARS"
MODEL_NAME = "REGNET-S"
TRAIN_EPOCH = 100
NUM_CLASS = 1500+1

def main():
    arg_parser = argparse.ArgumentParser(description="Metric trainer (%s)")
    arg_parser.add_argument(
        "--model_path", help="train or eval",
        default="D:/NOVATEK_1222/python/git-source/sort-tracker-test/MARS_REGNET-S_TEST-00030.h5")
    arg_parser.add_argument(
        "--data_path", help="train or eval",
        default="D:/NOVATEK_1222/dataset/mars_dataset/bbox_train_ref/*/**.jpg")
    arg_parser.add_argument(
        "--run_id", default="TEST"
    )
    args = arg_parser.parse_args()
    
    weight_decay = 1e-8
    
    model = regnet.regnet_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASS, weight_decay=weight_decay)
    model.load_weights(args.model_path)
    model.summary()
    datasets = glob.glob(args.data_path)
    import random
    random.shuffle(datasets)
    acc = 0.
    count = 0
    conf = 0.
    for dataset in datasets[:1000]:
        data_file_name = os.path.split(dataset)[-1]
        data_id = data_file_name[:4]
        img = cv2.resize(cv2.imread(dataset, cv2.IMREAD_COLOR), (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        cv2.imshow("dddd", img)
        in_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)/255., axis = 0)
        output = model.predict(in_img)
        print(np.max(output), np.argmax(output), output[0][np.argmax(output)])
        acc += 1. if int(data_id) == np.argmax(output) else 0.
        count += 1
        conf = acc/count
        # if cv2.waitKey(0) == ord('q'):
            # break
    print(conf)
    
    
    

if __name__ == "__main__":
    main()
