###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.io as sio
import configparser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
from keras.models import load_model
from keras.activations import sigmoid
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
# import sys
# sys.path.insert(0, './lib/')
# sys.setrecursionlimit(4000)
import os
# from deep_supervised import MyEntropyLoss
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config= config))
# help_functions.py
#from help_functions import *
# extract_patches.py
#from extract_patches import recompone
#from extract_patches import recompone_overlap
#from extract_patches import paint_border
#from extract_patches import kill_border
#from extract_patches import pred_only_FOV
#from extract_patches import get_data_testing
#from extract_patches import get_data_testing_overlap
# pre_processing.py
#from pre_processing import my_PreProc
import h5py
import tensorflow as tf
from data_feed import *
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


#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1

    img.save(filename + '.png')
    return img


#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
# config.read('configuration_new_data.txt')
config.read('configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')
_batchSize = int(config.get('training settings', 'batch_size'))
#original test images (for FOV selection)
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_groundTruth=path_data + config.get('data paths', 'test_groundTruth')
# test_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
# test_imgs_groundTruth=path_data + config.get('data paths', 'train_groundTruth')
test_data_ori = h5py.File(test_imgs_original,'r')
test_data_gt = h5py.File(test_imgs_groundTruth,'r')

test_imgs_orig = np.array(test_data_ori['image'])
test_groundTruth = np.array(test_data_gt['image'])

test_imgs = test_imgs_orig
test_masks = test_groundTruth

full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]



def testGenerator():
    for image in test_generator(test_imgs, test_masks, _batchSize):
        yield image




#model name
name_experiment = config.get('experiment name', 'name')
# path_experiment ='/home/wanghua/hdd0411/base5/train_result/'
path_experiment ='./'+name_experiment+'/'
N_visual = int(config.get('testing settings', 'N_group_visual'))
full_images_to_test = int(config.get('testing settings', 'full_images_to_test'))


#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')
#Load the saved model
# model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read(), custom_objects={'MyEntropyLoss': MyEntropyLoss})
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
# model = load_model(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
predictions1 = model.predict_generator(testGenerator(), verbose=1, steps=full_images_to_test)
# predictions = predictions1[0]
predictions3 = tf.nn.sigmoid(predictions1[0])
with tf.Session() as sess:
    predictions = predictions3.eval()
print("predicted images size :")
print(predictions.shape)
print('max pred:')
print(np.max(predictions))
print('min pred:')
print(np.min(predictions))





orig_imgs = test_imgs_orig[:,0,0:full_img_height,0:full_img_width]
gt_imgs = test_groundTruth[:, 0, 0:full_img_height,0:full_img_width]
n_data=orig_imgs.shape[0]
orig_imgs=np.reshape(orig_imgs,(n_data,1,full_img_height,full_img_width))
pred_imgs = predictions[:,:,0:full_img_height,0:full_img_width]
gt_imgs = np.reshape(gt_imgs ,(n_data,1,full_img_height,full_img_width))
# thr = 0.481
# thr = 0.616
# thr = 0.5
# pred_imgs[pred_imgs >= thr] = 1
# pred_imgs[pred_imgs < thr] = 0
print ('preds_shape:' +str(pred_imgs.shape))
pred_save=np.array(pred_imgs)
save_path='./new_test_result/'+name_experiment+'/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# sio.savemat(save_path+'preds.mat',{'preds':pred_save})

print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))

# visualize(group_images(orig_imgs,N_visual),save_path+"all_originals")#.show()
# visualize(group_images(pred_imgs,N_visual),save_path+"all_predictions")#.show()

#visualize results comparing mask and prediction:
assert (orig_imgs.shape[0] == pred_imgs.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
# for i in range(int(N_predicted/group)):
#     pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
#     # orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
#     # total_img = np.concatenate((orig_stripe,pred_stripe),axis=0)
#     gt_stripe = group_images(gt_imgs[i * group:(i * group) + group, :, :, :], group)
#     total_img = np.concatenate((gt_stripe, pred_stripe), axis=0)
#     visualize(total_img, save_path+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()


# for i in range(int(N_predicted/group)):
#     pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
#     visualize(pred_stripe,save_path+name_experiment +"Prediction"+str(i))#.show()

gt_path = save_path+'gt/'
pred_path = save_path+'pred/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)
if not os.path.exists(gt_path):
    os.mkdir(gt_path)

def novisualize(data,filename):
    '''
    the image is between 0-1
    :param data:
    :param filename:
    :return:
    '''
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0], data.shape[1]))
    assert (np.max(data) <= 1)
    # img = Image.fromarray(data.astype(np.uint8))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')


for i in range(N_predicted):
    # gt_stripe = gt_imgs[i, :, :, :]
    # novisualize(np.transpose(gt_stripe, (1, 2, 0)), gt_path + str(i))
    pred_stripe = pred_imgs[i, :, :, :]
    novisualize(np.transpose(pred_stripe, (1, 2, 0)), pred_path + str(i+1))

