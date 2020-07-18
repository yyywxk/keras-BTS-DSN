import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D, Deconv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
import keras.backend as K
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import numpy as nptrain_generator
import configparser
from data_feed import *
import h5py
import math
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import optimizers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
smooth=1e-5
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config= config))

def MyEntropyLoss(y_true, dsn1, dsn2, dsn3, dsn4, y_pred):
    # y_sig = keras.activations.sigmoid(y_pred)
    # return K.categorical_crossentropy(y_true, y_sig)
    dsn1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=dsn1)
    dsn2_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=dsn2)
    dsn3_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=dsn3)
    dsn4_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=dsn4)
    dsn_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    myloss = dsn1_loss + dsn2_loss + dsn3_loss + dsn4_loss + dsn_loss
    return myloss

def Dice_coef(y_true,y_pred):
    y_intersection=y_true*y_pred
    dice=(2.0*K.sum(y_intersection)+smooth)/(K.sum(y_true)+K.sum(y_pred)+smooth)
    return dice

def Dice(y_true,y_pred):
    weight=[1.,1.,1.]
    dice_NCR_NET=Dice_coef(y_true[:,:,:,:,0],y_pred[:,:,:,:,0])
    dice_ED=Dice_coef(y_true[:,:,:,:,1],y_pred[:,:,:,:,1])
    dice_ET=Dice_coef(y_true[:,:,:,:,2],y_pred[:,:,:,:,2])
    dice_wm=(weight[0]*dice_NCR_NET+weight[1]*dice_ED+weight[2]*dice_ET)/K.sum(weight)
    return dice_wm

def cropfunc(input, target):
    w = input._keras_shape[2]
    h = input._keras_shape[3]
    tw = target._keras_shape[2]
    th = target._keras_shape[3]
    offset = [0, 0]
    offset[0] = int((w - tw)/2)
    offset[1] = int((h - th)/2)
    cropped = Cropping2D(cropping=((offset[0], w - offset[0]-tw), (offset[1], h - offset[1] - th)),
                         data_format='channels_first')(input)
    return cropped

def deep_net(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    # Block 1
    x1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first')(inputs)
    x1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',data_format='channels_first')(x1_1)

    x1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',data_format='channels_first')(x1_2)

    # Block 2
    x2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',data_format='channels_first')(x1_pool)
    x2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',data_format='channels_first')(x2_1)
    x2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',data_format='channels_first')(x2_2)

    # Block 3
    x3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',data_format='channels_first')(x2_pool)
    x3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',data_format='channels_first')(x3_1)
    x3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',data_format='channels_first')(x3_2)
    x3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',data_format='channels_first')(x3_3)

    # Block 4
    # x4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',data_format='channels_first')(x3_3)
    x4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first')(
        x3_pool)
    #x4_drop1 = Dropout(0.5, name='dr1')(x4_1)
    x4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',data_format='channels_first')(x4_1)
    #x4_drop2 = Dropout(0.5, name='dr2')(x4_2)
    x4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',data_format='channels_first')(x4_2)
    #x4_drop3 = Dropout(0.5, name='dr3')(x4_3)


    x1_2_16=Conv2D(16,(3,3),name='x1_2_16',padding='same',data_format='channels_first')(x1_2)
    x2_2_16=Conv2D(16,(3,3),name='x2_2_16',padding='same',data_format='channels_first')(x2_2)
    x3_3_16 = Conv2D(16, (3, 3), name='x3_3_16', padding='same', data_format='channels_first')(x3_3)
    x4_3_16 = Conv2D(16, (3, 3), name='x4_3_16', padding='same', data_format='channels_first')(x4_3)

    conv4_to_1=Conv2D(2, (3,3), padding='same', data_format='channels_first',name='conv4_to_1')(x4_3)
    # conv4_to_1_up=UpSampling2D(size=(8,8),data_format='channels_first')(conv4_to_1)
    conv4_to_1_up = Deconv2D(filters=2, kernel_size=16, strides=(8, 8), data_format='channels_first')(conv4_to_1)
    # crop4to1_back = Cropping2D(cropping=(patch_height, patch_width), data_format='channels_first')(conv4_to_1_up)
    # conv1_2_17=concatenate([conv4_to_1_up,x1_2_16],axis=1)
    crop4to1_back = cropfunc(conv4_to_1_up, inputs)
    conv1_2_17 = concatenate([crop4to1_back, x1_2_16], axis=1)
    dsn1_in=Conv2D(1,(1,1),data_format='channels_first', name='dsn1_in')(conv1_2_17)

    conv1_to_2=Conv2D(2, (1,1), name='conv1_to_2',data_format='channels_first')(x1_2_16)
    # side_multi2_up=UpSampling2D(size=(2,2),data_format='channels_first')(x2_2_16)
    side_multi2_up = Deconv2D(16, 4, strides=(2, 2), padding='same', data_format='channels_first')(x2_2_16)
    # upside_multi2 = Cropping2D(cropping=(patch_height, patch_width), data_format='channels_first')(side_multi2_up)
    # conv2_2_17=concatenate([conv1_to_2,side_multi2_up],axis=1)
    upside_multi2 = cropfunc(side_multi2_up, inputs)
    upside_multi27 = concatenate([conv1_to_2, upside_multi2], axis=1)
    # dsn2_in=Conv2D(1,(1,1),data_format='channels_first',padding='same',name='dsn2_in')(conv2_2_17)
    dsn2_in = Conv2D(1, (1, 1), data_format='channels_first', name='dsn2_in')(upside_multi27)

    # conv2_to_3=Conv2D(1,(1,1),padding='same',name='conv2_to_3',data_format='channels_first')(conv2_2_17)
    # side_multi3_up=UpSampling2D(size=(4,4),data_format='channels_first')(x3_3_16)
    conv2_to_3 = Conv2D(1, (1, 1), name='conv2_to_3', data_format='channels_first')(upside_multi27)
    side_multi3_up = Deconv2D(16, 8, strides=(4, 4), data_format='channels_first')(x3_3_16)
    # upside_multi3 = Cropping2D(cropping=(patch_height, patch_width), data_format='channels_first')(side_multi3_up)
    upside_multi3 = cropfunc(side_multi3_up, inputs)
    # conv3_2_17=concatenate([conv2_to_3,side_multi3_up],axis=1)
    upside_multi37 = concatenate([conv2_to_3, upside_multi3], axis=1)
    # dsn3_in=Conv2D(1,(1,1),data_format='channels_first',padding='same',name='dsn3_in')(conv3_2_17)
    dsn3_in = Conv2D(1, (1, 1), data_format='channels_first', name='dsn3_in')(upside_multi37)

    # conv3_to_4=Conv2D(1,(1,1),padding='same',name='conv3_to_4',data_format='channels_first')(conv3_2_17)
    # side_multi4_up=UpSampling2D(size=(8,8),data_format='channels_first')(x4_3_16)
    # conv4_2_17=concatenate([conv3_to_4,side_multi4_up],axis=1)
    # dsn4_in=Conv2D(1,(1,1),data_format='channels_first',padding='same',name='dsn4_in')(conv4_2_17)
    conv3_to_4 = Conv2D(1, (1, 1), name='conv3_to_4', data_format='channels_first')(upside_multi37)
    side_multi4_up = Deconv2D(16, 16, strides=(8, 8), data_format='channels_first')(x4_3_16)
    # upside_multi4 = Cropping2D(cropping=(patch_height, patch_width), data_format='channels_first')(side_multi4_up)
    upside_multi4 = cropfunc(side_multi4_up, inputs)
    upside_multi47 = concatenate([conv3_to_4, upside_multi4], axis=1)
    dsn4_in=Conv2D(1, (1, 1), data_format='channels_first', name='dsn4_in')(upside_multi47)

    dsn = concatenate([dsn1_in, dsn2_in, dsn3_in, dsn4_in], axis=1)
    dsn_out = Conv2D(1,(1,1),data_format='channels_first',padding='same',name='dsn_out')(dsn)
    # dsn_out = Conv2D(1, (1, 1), data_format='channels_first', name='dsn_out', activation='sigmoid')(dsn)

    target = Input(shape=(1, patch_height, patch_width), name="target")
    MyLoss = Lambda(lambda x: MyEntropyLoss(*x), name="complex_loss")\
        ([target, dsn1_in, dsn2_in, dsn3_in, dsn4_in, dsn_out])

    # outputs = [dsn_out, MyLoss]
    outputs = [dsn_out, dsn1_in, dsn2_in, dsn3_in, dsn4_in, MyLoss]
    model=Model(inputs=[inputs, target], outputs=outputs)
    # model = Model(inputs=inputs, outputs=[dsn1_in, dsn2_in, dsn3_in, dsn4_in, dsn_out])

    # model.add_loss(myloss)
    # model.compile(loss=[None] * len(model.outputs), optimizer='SGD')
    # model.compile(optimizer='SGD')

    model._losses = []
    model._per_input_losses = {}
    for loss_name in ["complex_loss"]:
        layer = model.get_layer(loss_name)
        if layer.output in model.losses:
            continue
        loss = tf.reduce_mean(layer.output)
        model.add_loss(loss)


    return model



#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
_batchSize = int(config.get('training settings', 'batch_size'))

if not os.path.exists(name_experiment):
    os.mkdir(name_experiment)

n_ch=3
patch_height=512
patch_width=512

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(patch_height, patch_width, n_ch))
# layer_name = 'block4_conv3'
# base_model.summary()
# vgg = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
# # vgg = Model(inputs=base_model.input, outputs=base_model.output)
# vgg.summary()
# vgg.save_weights('vgg.h5')

vgg_weights_path = './vgg.h5'
model=deep_net(n_ch,patch_height,patch_width)
model.load_weights(vgg_weights_path, by_name=True)
model.trainable = True
model.summary()
sgd = optimizers.SGD(lr=8e-3, decay=0.0005, momentum=0.9, nesterov=True)  # 1e-8;
model.compile(loss=[None] * len(model.outputs), optimizer=sgd)
log_path = './log'
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_path1 = log_path+'/'+name_experiment
if not os.path.exists(log_path1):
    os.mkdir(log_path1)
tb_cb = keras.callbacks.TensorBoard(log_dir=log_path1)


print("Check: final output of the network:")
print(model.output_shape)
#plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)




new_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
new_train_imgs_groundTruth=path_data + config.get('data paths', 'train_groundTruth')
train_data_ori= h5py.File(new_train_imgs_original,'r')
train_data_gt=h5py.File(new_train_imgs_groundTruth,'r')

train_imgs_original= np.array(train_data_ori['image'])
train_groundTruth=np.array(train_data_gt['image'])




train_imgs = train_imgs_original
train_masks = train_groundTruth

#check masks are within 0-1
assert(np.min(train_masks) == 0 and np.max(train_masks) == 1)
print ("\ntrain images/masks shape:")
print (train_imgs.shape)
print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
print ("train masks are within 0-1\n")
#============  Training ==================================
checkpoint_test = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', monitor='val_loss', save_best_only=True,save_weights_only=True) #save at each epoch if the validation decreased
checkpoint = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment + "bestTrainWeight" + ".h5", monitor='loss', save_best_only=True, save_weights_only=True)

# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

#patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
#model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

keepPctOriginal = 0.5
trans = 0.16 # +/- proportion of the image size
rot = 9 # +/- degree
zoom = 0.12 # +/- factor
shear = 0 # +/- in radian x*np.pi/180
elastix = 0 # in pixel
intensity = 0.07 # +/- factor
hflip = True
vflip = True
iter_times=250
num=train_imgs_original.shape[0]
np.random.seed(0)
index=list(np.random.permutation(num))
_X_train=train_imgs[index][0:174]
_Y_train=train_masks[index][0:174]
print(_X_train.shape)
print(_Y_train.shape)
_X_vali=train_imgs[index][174:219]
_Y_vali=train_masks[index][174:219]
print(_X_vali.shape)
print(_Y_vali.shape)


def ImgGenerator():
    for image in train_generator(_X_train, _Y_train,_batchSize, iter_times, _keepPctOriginal=0.5,
                                 _trans=TRANSLATION_FACTOR, _rot=ROTATION_FACTOR, _zoom=ZOOM_FACTOR, _shear=SHEAR_FACTOR,
                                 _elastix=VECTOR_FIELD_SIGMA, _intensity=INTENSITY_FACTOR, _hflip=True, _vflip=True):
          yield image
def valiGenerator():
    for image in validation_generator(_X_vali, _Y_vali,_batchSize):
        yield image

# stepsPerEpoch = math.ceil((num-1) / _batchSize)
# validationSteps = math.ceil(1 / _batchSize)
stepsPerEpoch = math.ceil((num-45) / _batchSize)
validationSteps = math.ceil(45 / _batchSize)
history1 = model.fit_generator(ImgGenerator(), verbose=1, workers=1,
                                                 validation_data=valiGenerator(),
                                                # validation_data=ImgGenerator(),
                                                 steps_per_epoch=stepsPerEpoch, epochs=N_epochs,
                                                 validation_steps=validationSteps,
                                                 callbacks=[checkpoint, checkpoint_test, tb_cb])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


