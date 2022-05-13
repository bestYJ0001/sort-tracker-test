from pickle import TRUE
from tensorflow import keras
import numpy as np
import glob
import cv2
import imgaug
from imgaug.augmentables.batches import UnnormalizedBatch
from imgaug import augmenters as iaa
import os

class MarsGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, data_path, augs, is_train=True):
        if is_train:
            print('[i] Init training data generator')
        else:
            print('[i] Init validation data generator')
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.augmenter = iaa.Sequential(augs)
        self.is_train = is_train
        self.is_test = False if self.is_train is True else True
        if self.is_train :
            self.data = glob.glob(os.path.join(data_path, "bbox_train_ref/**/***.jpg")) 
        else : 
            self.data = glob.glob(os.path.join(data_path, "bbox_test_ref/**/***.jpg")) 
        self.indexes = None
        self.on_epoch_end()
        
    
    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))
    
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)
    
    
    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        samples_list = [self.data[i] for i in indexes]
        images, ids, cam_indices, track_len_indices = self.__data_gen(samples_list)
        if False :
            print(ids)
            cv2.imshow("dkdkd", images[0])
            cv2.waitKey(0)
        return images, ids
    
    
    def __data_gen(self, samples_list):
        batch_img = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        # batch_ids = np.zeros(shape=(self.batch_size, 1, self.num_classes), dtype=np.float32)
        batch_ids = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        batch_cam_indices = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        batch_track_len = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        for index, img_path in enumerate(samples_list):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            transformed = self.augmenter(image=img)
            transformed_img = transformed
            transformed_img = cv2.cvtColor(cv2.resize(transformed_img, (self.input_shape[1], self.input_shape[0])), cv2.COLOR_BGR2RGB)
            if self.is_train:
                transformed_img = np.fliplr(transformed_img)
            batch_img[index] = transformed_img / 255.
            # ids_idx = 0 if os.path.split(img_path)[-1][:4] == "00-1" else int(os.path.split(img_path)[-1][:4]) + 1
            # batch_ids[index,:,ids_idx] = 1.
            batch_ids[index] = -1. if os.path.split(img_path)[-1][:4] == "00-1" else int(os.path.split(img_path)[-1][:4])
            batch_cam_indices[index] = os.path.split(img_path)[-1][5]
            batch_track_len[index] = os.path.split(img_path)[-1][7:11]
        return batch_img ,batch_ids, batch_cam_indices, batch_track_len
            
        

if __name__ == '__main__':
    bgen = MarsGenerator(1, (128, 64, 3), 1501, 'D:/NOVATEK_1222/dataset/mars_dataset/', [], True)
    print(bgen.__len__())
    for i in range(bgen.__len__()):
        img, _ = bgen.__getitem__(i)