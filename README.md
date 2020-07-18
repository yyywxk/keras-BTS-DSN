# keras-BTS-DSN
## Introduction

This is a Keras (2.2.4) implementation of [BTS-DSN](https://doi.org/10.1016/j.ijmedinf.2019.03.015), which is a Neural Network for Retinal Vessel Segmentation. The original [code](https://github.com/guomugong/BTS-DSN) is written by Caffe.

## Environment

keras = 2.2.4

tensorflow = 1.12.0

python = 3.6

## Training BTS-DSN

1. Download [VGG16](https://download.csdn.net/download/yyywxk/12606962) pretrained model.

2. Prepare the training and testing set.

   ```bash
   python prepare_dataset_hdf5_for_2D.py  # the source code for generating training dataset data in hdf5 format
   python prepare_dataset_hdf5_for_2D_test.py # the source code for generating testing dataset data in hdf5 format
   ```

3. Run the training script.

   ```bash
   python deep_supervised.py
   ```

## Testing BTS-DSN

Run the testing script.

```bash
python 2D_new_run_test.py
```

## Acknowledgement

[BTS-DSN](https://github.com/guomugong/BTS-DSN)

[SVS-net](https://github.com/Binjie-Qin/SVS-net)

## License

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/zh_CN)