import glob
import os
import random
import sys
from enum import Enum
from typing import Callable, Union

import cv2
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LambdaCallback
from tensorflow.keras import losses as lss
from tensorflow.keras.optimizers import *
from tensorflow.keras import metrics
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from losses2 import Semantic_loss_functions

lss2 = Semantic_loss_functions()

class GenType(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3


class CustomUnet:
    def __init__(self):
        self.__model = self.create_model(input_size=(256, 256, 3))
        if os.path.exists('models/unet_weights.h5'):
            self.__model.load_weights('models/unet_weights.h5')

        self.__kernel = np.ones((3, 3), np.uint8)

    def create_model(self, input_size=(256, 256, 3)):
        inputs = Input(input_size)
        conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.25)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.25)(conv5)

        up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.summary()
        return model

    def train(self):
        if not os.path.exists('models/'):
            os.mkdir('models/')
        batch_size = 4
        all_images = glob.glob('bdd100k-dataset/images/*.png', recursive=True)
        print(f'Total images: {len(all_images)}')
        random.shuffle(all_images)
        valid_size = len(all_images) // 5
        train_size = len(all_images) - valid_size

        def epoch_end(ep, lg):
            for f in glob.glob('bdd100k-dataset/test/*.jpg'):
                try:
                    img = cv2.imread(f)
                    original_shape = img.shape
                    img = cv2.resize(img, (256, 256))
                    pred = self.__model.predict(np.expand_dims(img / 255.0, axis=0), batch_size=None,
                                                verbose=0,
                                                steps=None)
                    orig_img = cv2.resize(pred[0] * 255.0, (original_shape[1], original_shape[0]))
                    if not os.path.exists('out/'):
                        os.mkdir('out/')
                    cv2.imwrite(f'out/{ep}_{os.path.basename(f)}', orig_img)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

        def data_gen(batchsize, gen_type: GenType):
            image_list = []
            if gen_type == GenType.TRAIN:
                image_list = all_images[0: -(len(all_images) // 5)]
            elif gen_type == GenType.VALID:
                image_list = all_images[-(len(all_images) // 5):]
            while True:
                try:
                    inputs = []
                    targets = []
                    batchcount = 0
                    for f in image_list:
                        try:
                            img = cv2.imread(f)
                            img = cv2.resize(img, (256, 256))
                            img_array = img / 255.0
                            inputs.append(img_array)
                            out_img = cv2.imread(f"{f.split('.')[0].replace('images', 'masks')}.png", 0)
                            # out_img = cv2.imread(f"{f.split('.')[0].replace('images', 'masks')}_L.png", 0)
                            out_img = cv2.resize(out_img, (256, 256))
                            out_img_array = out_img / 255.0
                            targets.append(out_img_array)
                            batchcount += 1
                            if batchcount > batchsize:
                                X = np.array(inputs, dtype=np.float)
                                y = np.array(targets, dtype=np.float)
                                yield (X, y)
                                inputs = []
                                targets = []
                                batchcount = 0
                        except:
                            continue
                except Exception as ee:
                    print(f'ee: {str(ee)}')

        checkpoint = ModelCheckpoint(filepath='models/unet_weights.h5',
                                     save_weights_only=True, save_best_only=True,
                                     mode='min', monitor='val_loss', verbose=True)
        csv_logger = CSVLogger('models/unet_training.log')
        opt = Adam(learning_rate=1e-5)
        tb = TensorBoard(log_dir="unet_logs", update_freq='batch', batch_size=batch_size)
        lambda_cb = LambdaCallback(on_epoch_end=epoch_end)
        if os.path.exists('models/unet_weights.h5'):
            self.__model.load_weights('models/unet_weights.h5')
        self.__model.compile(loss=lss2.dice_loss,
                             metrics=['accuracy'],
                             optimizer=opt)
        self.__model.fit_generator(data_gen(batch_size, GenType.TRAIN),
                                   steps_per_epoch=train_size // batch_size,
                                   epochs=1000,
                                   validation_data=data_gen(1, GenType.VALID),
                                   validation_steps=valid_size,
                                   callbacks=[checkpoint, csv_logger, tb, lambda_cb])

    def predict(self, img):
        original_shape = img.shape
        img = cv2.resize(img, (256, 256))
        pred = self.__model.predict(np.expand_dims(img / 255.0, axis=0), batch_size=None,
                                    verbose=0,
                                    steps=None)

        mask_img = cv2.resize(pred[0] * 255.0, (original_shape[1], original_shape[0]))
        img = cv2.resize(img, (original_shape[1], original_shape[0]))
        mask_img = mask_img.astype('uint8')
        mask_img = cv2.dilate(mask_img, self.__kernel, iterations=1)
        mask_img = cv2.erode(mask_img, self.__kernel, iterations=1)
        ret, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        predictions = []
        result = []
        for cnt in contours:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / (M["m00"] + 1e-4))
            cY = int(M["m01"] / (M["m00"] + 1e-4))
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
            predictions.append([(cX, cY), [(x, y), (x + w, y + h)]])  # center of mass, bounding rectangle
        result.append(predictions)
        result.append(img)
        return result
