import os
import h5py
from PIL import Image
import numpy as np
import os
import pylab as py
import matplotlib.pyplot as plt
plt.switch_backend('agg')



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)
    f.close()


#------------Path of the images --------------------------------------------------------------
common = './test/my_input/60217_2_1/'
ori_imgs_path = 'ori/'
gt_imgs_path = 'gt/'
original_imgs_train = common + ori_imgs_path
groundTruth_imgs_train = common + gt_imgs_path
# Nimgs_train = 4
# Nimgs_test = 4
#Nimgs_train = 218
Nimgs_test = 85
channels = 1
height = 512
width = 512
datapath = 'datasets_training/'
dataset_path = common + datapath
def get_datasets(imgs_dir):
    
    imgs_test = np.empty((Nimgs_test, 3, height, width))
    
    
    # train_index=index[0:219]
    
    #test_index=index[219:]
    # train_index=index[0:4]
    # test_index=index[4:]
    for item in range(0, Nimgs_test):
        ori_img_name = imgs_dir+'ori_'+str(item+1)+'.png'

        img=Image.open(ori_img_name)

        image = np.reshape(np.array(img) / np.max(img), (1, height, width))
        imgs_test[item] = np.concatenate((image, image, image), axis=0)

        # img_=np.array(img)/np.max(img)
        # imgs_test[item]=img_

        print("imgs max: " + str(np.max(imgs_test)))
        print("imgs min: " + str(np.min(imgs_test)))

    imgs1 = np.reshape(imgs_test, (Nimgs_test, 3, height, width))

    return imgs1

def get_datasets1(imgs_dir,groundTruth_dir):
    imgs_test = np.empty((Nimgs_test, 3, height, width))
    groundTruth_test = np.empty((Nimgs_test, height, width))

    img_list = os.listdir(imgs_dir)
    # for item, img_name in enumerate(img_list):
    for item in range(0, Nimgs_test):
        img_name = 'ori_'+str(item+1)+'.png'
        ori_img_name = os.path.join(imgs_dir, img_name)
        # gt_img = 'label{}'.format(img_name[3:])
        # gt_img_name = os.path.join(groundTruth_dir, gt_img)
        gt_img_name = os.path.join(groundTruth_dir, img_name)

        img = Image.open(ori_img_name)
        gt = Image.open(gt_img_name)
        image = np.reshape(np.array(img) / np.max(img), (1, height, width))
        imgs_test[item] = np.concatenate((image, image, image), axis=0)
        groundTruth_test[item] = np.array(gt)/np.max(gt)
        print("imgs max: " + str(np.max(imgs_test)))
        print("imgs min: " + str(np.min(imgs_test)))
        print('gt max:' + str(np.max(groundTruth_test)))
        print('gt min:' + str(np.min(groundTruth_test)))

    imgs2 = np.reshape(imgs_test, (Nimgs_test, 3, height, width))
    groundTruth2 = np.reshape(groundTruth_test, (Nimgs_test, 1, height, width))

    return imgs2, groundTruth2


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the testing datasets
# imgs_1 = get_datasets(original_imgs_train)
# print ("saving test datasets")
# print (imgs_1.shape)
#
# write_hdf5(imgs_1,dataset_path + "dataset_groundTruth_test.hdf5")

# get gt&ori datasets
imgs_2, groundTruth_2= get_datasets1(original_imgs_train,groundTruth_imgs_train)
print("saving test datasets")
print(imgs_2.shape)
print(groundTruth_2.shape)
write_hdf5(imgs_2,dataset_path + "dataset_imgs_test.hdf5")
write_hdf5(groundTruth_2, dataset_path + "dataset_groundTruth_test.hdf5")
