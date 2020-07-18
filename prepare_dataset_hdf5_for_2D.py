#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE_new database
#
#============================================================
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
original_imgs_train = "./train/ori/"
groundTruth_imgs_train = "./train/gt/"
# Nimgs_train = 4
# Nimgs_test = 4
Nimgs_train = 219
Nimgs_test = 40
channels = 1
height = 512
width = 512
dataset_path = "./train/new_datasets_training_testing/"
def get_datasets(imgs_dir,groundTruth_dir):
    imgs_train = np.empty((Nimgs_train, 3, height, width))
    groundTruth_train = np.empty((Nimgs_train, height, width))
    imgs_test = np.empty((Nimgs_test, 3, height, width))
    groundTruth_test = np.empty((Nimgs_test, height, width))
    np.random.seed(0)
    index = list(np.random.permutation(Nimgs_train+Nimgs_test))
    # train_index=index[0:219]
    train_index = index[0:219]
    test_index=index[219:]
    # train_index=index[0:4]
    # test_index=index[4:]
    for count,item in enumerate(train_index):
        ori_img_name=imgs_dir+'ori_'+str(item+1)+'.png'
        gt_img_name=groundTruth_dir+'gt_'+str(item+1)+'.png'
        img=Image.open(ori_img_name)
        gt=Image.open(gt_img_name)
        image = np.reshape(np.array(img)/np.max(img), (1, height, width))
        imgs_train[count] = np.concatenate((image, image, image), axis=0)
        groundTruth_train[count]=np.array(gt)/np.max(gt)
        print("imgs max: " + str(np.max(imgs_train)))
        print("imgs min: " + str(np.min(imgs_train)))
        print('gt max:' + str(np.max(groundTruth_train)))
        print('gt min:' + str(np.min(groundTruth_train)))
    imgs1 = np.reshape(imgs_train, (Nimgs_train, 3, height, width))
    groundTruth1 = np.reshape(groundTruth_train, (Nimgs_train, 1, height, width))
    for count,item in enumerate(test_index):
        ori_img_name = imgs_dir + 'ori_' + str(item+1)+'.png'
        gt_img_name = groundTruth_dir + 'gt_' + str(item+1)+'.png'
        img = Image.open(ori_img_name)
        gt = Image.open(gt_img_name)
        image = np.reshape(np.array(img)/np.max(img), (1, height, width))
        imgs_test[count] = np.concatenate((image, image, image), axis=0)
        groundTruth_test[count] = np.array(gt)/np.max(gt)
        print("imgs max: " + str(np.max(imgs_test)))
        print("imgs min: " + str(np.min(imgs_test)))
        print('gt max:' + str(np.max(groundTruth_train)))
        print('gt min:' + str(np.min(groundTruth_train)))
    imgs2 = np.reshape(imgs_test, (Nimgs_test, 3, height, width))
    groundTruth2 = np.reshape(groundTruth_test, (Nimgs_test, 1, height, width))
    return imgs1, groundTruth1, imgs2, groundTruth2
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the testing datasets
imgs_1, groundTruth_1, imgs_2, groundTruth_2 = get_datasets(original_imgs_train, groundTruth_imgs_train)
print("saving test datasets")
print(imgs_1.shape)
print(groundTruth_1.shape)
write_hdf5(imgs_1,dataset_path + "dataset_imgs_train.hdf5")
write_hdf5(groundTruth_1, dataset_path + "dataset_groundTruth_train.hdf5")
write_hdf5(imgs_2,dataset_path + "dataset_imgs_test.hdf5")
write_hdf5(groundTruth_2, dataset_path + "dataset_groundTruth_test.hdf5")


## show img
# img_1=imgs_1[50,0,:,:]
# gt1=groundTruth_1[50]
# img_2=imgs_1[10,0,:,:]
# gt2=groundTruth_1[10]
# img_3=imgs_1[100,0,:,:]
# gt3=groundTruth_1[100]
# img_4=imgs_2[10,0,:,:]
# gt4=groundTruth_2[10]
#
#
# py.ion()
# plt.figure(str(0))
# plt.imshow(np.array(np.squeeze(img_1)), cmap=plt.get_cmap('gray'))
# outpath1 = './' + '%s.png' % 0
# plt.savefig(outpath1)
#
# plt.figure(str(1))
# plt.imshow(np.array(np.squeeze(gt1)), cmap=plt.get_cmap('gray'))
# outpath2 = './' + '%s.png' % 1
# plt.savefig(outpath2)
#
# plt.figure(str(2))
# plt.imshow(np.array(np.squeeze(img_2)), cmap=plt.get_cmap('gray'))
# outpath3 = './' + '%s.png' % 2
# plt.savefig(outpath3)
#
# plt.figure(str(3))
# plt.imshow(np.array(np.squeeze(gt2)), cmap=plt.get_cmap('gray'))
# outpath4 = './' + '%s.png' % 3
# plt.savefig(outpath4)
#
# plt.figure(str(4))
# plt.imshow(np.array(np.squeeze(img_3)), cmap=plt.get_cmap('gray'))
# outpath5 = './' + '%s.png' % 4
# plt.savefig(outpath5)
#
# plt.figure(str(5))
# plt.imshow(np.array(np.squeeze(gt3)), cmap=plt.get_cmap('gray'))
# outpath6 = './' + '%s.png' % 5
# plt.savefig(outpath6)
#
# plt.figure(str(6))
# plt.imshow(np.array(np.squeeze(img_4)), cmap=plt.get_cmap('gray'))
# outpath7 = './' + '%s.png' % 6
# plt.savefig(outpath7)
#
# plt.figure(str(7))
# plt.imshow(np.array(np.squeeze(gt4)), cmap=plt.get_cmap('gray'))
# outpath8 = './' + '%s.png' % 7
# plt.savefig(outpath8)

















# basepath=os.getcwd()
# img_base_path=os.path.join(basepath,'ori')
# files=os.listdir(img_base_path)
# save_base_path=os.path.join(basepath,'hdf5')
# for filename in files:
#     img_files=os.path.join(img_base_path,filename)
#     imgs=os.listdir(img_files)
#     nums=len(imgs)
#     N_imgs=int(nums/2)
#     channels = 1
#     height = 512
#     width = 512
#     imgs_train = np.empty((N_imgs, height, width))
#     groundTruth_train = np.empty((N_imgs, height, width))
#     count = 0
#     for ff in imgs:
#
#         if 'ori' in ff:
#
#             ori_img_name=os.path.join(img_files,ff)
#             gt_img_name=os.path.join(img_files,'vs'+ff[-7:])
#             img=Image.open(ori_img_name)
#             gt=Image.open(gt_img_name)
#             assert(np.max(np.array(img))>=1)
#             imgs_train[count]=np.array(img)/np.max(img)
#             assert(np.max(np.array(gt)>=1))
#             groundTruth_train[count]=np.array(gt)>=1
#             count+=1
#     print (count)
#
#     imgs1 = np.reshape(imgs_train, (N_imgs, 1, height, width))
#     print(np.max(imgs1))
#     groundTruth1 = np.reshape(groundTruth_train, (N_imgs, 1, height, width))
#     print(np.max(groundTruth1))
#     write_hdf5(imgs1, save_base_path +'/'+ filename + "_ori.hdf5")
#     write_hdf5(groundTruth1,  save_base_path+'/'+filename+ "_gt.hdf5")









