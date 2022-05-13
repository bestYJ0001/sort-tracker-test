# vim: expandtab:ts=4:sw=4
import argparse
from datetime import datetime
import os
import tensorflow as tf
import nets.deep_sort.regnet_network as regnet
from train.Generator import MarsGenerator
import datetime

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("GPU found" if tf.test.gpu_device_name() else "No GPU found")

IMAGE_SHAPE = (128, 64, 3)
DATASET_NAME = "MARS"
MODEL_NAME = "REGNET-S"
TRAIN_EPOCH = 100
NUM_CLASS = 1500+1


def main():
    arg_parser = argparse.ArgumentParser(description="Metric trainer (%s)")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to MARS dataset directory.",
        default="D:/NOVATEK_1222/dataset/mars_dataset/")
    arg_parser.add_argument(
        "--mode", help="train or eval",
        default="train")
    arg_parser.add_argument(
        "--batch", help="train",
        default=32, type=int)
    arg_parser.add_argument(
        "--run_id", required=True
    )
    args = arg_parser.parse_args()
    
    weight_decay = 1e-8
    
    model = regnet.regnet_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASS, weight_decay=weight_decay)
    
    if args.mode == "train":
        train_batch_gen = MarsGenerator(args.batch, IMAGE_SHAPE, NUM_CLASS, args.dataset_dir, [], True)
        valid_batch_gen = MarsGenerator(args.batch, IMAGE_SHAPE, NUM_CLASS, args.dataset_dir, [], False)
    else :
        print(args.mode)
        train_batch_gen = None 
        valid_batch_gen = MarsGenerator(args.batch, IMAGE_SHAPE, NUM_CLASS, args.dataset_dir, [], False)
    
    """
    MarsGenerator
    return [batch images, batch ids]
    y_true : label (ids) 0 ~ 1500
    """
        
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()]
    )
    
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
    
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                   patience=5,
                                                   verbose=1, factor=0.5),
                 tf.keras.callbacks.ModelCheckpoint(filepath='./saved_models/'+DATASET_NAME+'_'+MODEL_NAME+'_'+args.run_id+'-{epoch:05d}.h5',
                                                 save_freq='epoch',
                                                 verbose=1),
                 tf.keras.callbacks.TensorBoard(log_dir),
                 tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]
    
    if args.mode is "train":
        model.fit_generator(train_batch_gen,
                            use_multiprocessing=True,
                            callbacks=callbacks,
                            workers=8,
                            epochs=TRAIN_EPOCH,
                            validation_data=valid_batch_gen,
                            validation_steps=5)
        model.save('./saved_models/'+DATASET_NAME+MODEL_NAME+'final.h5')
    

if __name__ == "__main__":
    main()
