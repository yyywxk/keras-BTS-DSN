import time
import numpy as np
import random
import scipy as sp
import scipy.interpolate
import scipy.ndimage
import scipy.ndimage.interpolation
import random
INTENSITY_FACTOR = 0.2
VECTOR_FIELD_SIGMA = 5.  # in pixel
ROTATION_FACTOR = 10  # degree
TRANSLATION_FACTOR = 0.2  # proportion of the image size
SHEAR_FACTOR = 2 * np.pi / 180  # in radian
ZOOM_FACTOR = 0.1


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    shift = np.random.uniform(-intensity, intensity)  # TODO add a choice if we want the same shift for all channels
    channel_images = [np.clip(x_channel + shift, min_x, max_x)
                      for x_channel in x]
    # channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
    # for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [sp.ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                                final_offset, order=0, mode=fill_mode, cval=cval) for
                      x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def ApplyRandomTransformations(_x, _y, _pts, _trans, _rot, _zoom, _shear, _elastix, _row_index=1, _col_index=2,
                               _channel_index=0, _fill_mode='constant', _cval=0.):
    if _elastix != 0:
        sigma = _elastix  # in pixel
        kernelSize = 3
        sizeAll = kernelSize + 2
        imgShape = (_x.shape[1], _x.shape[2])

        # create the indices of the 5x5 vector field (fieldPts.shape = (25,2))
        fieldPts = np.mgrid[0.:1.:complex(sizeAll), 0.:1.:complex(sizeAll)].swapaxes(0, 2).swapaxes(0, 1).reshape(
            (sizeAll * sizeAll, 2))
        # create the displacement (x and y) of the 5x5 vector field (border have no displacement so it's 0) (displacementX.shape = (25))
        displacementX = np.zeros((sizeAll * sizeAll))
        displacementY = np.zeros((sizeAll * sizeAll))
        for i in range(0, sizeAll * sizeAll):
            if fieldPts[i][0] != 0. and fieldPts[i][0] != 1. \
                    and fieldPts[i][1] != 0. and fieldPts[i][1] != 1.:
                displacementX[i] = np.random.normal(0, sigma, 1)
                displacementY[i] = np.random.normal(0, sigma, 1)
        # transform the indice of the 5x5 vector field in the image coordinate system (TODO WARNING works only with square images)
        fieldPts = fieldPts * imgShape[0]  # TODO check if it's not imgShape[0] - 1?

        # create the indices of all pixels in the image (gridX.shape = (1024,1024))
        gridX, gridY = np.mgrid[0.:(imgShape[0] - 1):complex(imgShape[0]), 0.:(imgShape[1] - 1):complex(imgShape[1])]
        # interpolate the vector field for every pixels in the image (dxGrid.shape = (1024,1024))
        dxGrid = scipy.interpolate.griddata(fieldPts, displacementX, (gridX, gridY), method='cubic')
        dyGrid = scipy.interpolate.griddata(fieldPts, displacementY, (gridX, gridY), method='cubic')

        # apply the displacement on every pixels (indices = [indices.shape[0] = 1024*1024, indices.shape[1] = 1024*1024])
        indices = np.reshape(gridY + dyGrid, (-1, 1)), np.reshape(gridX + dxGrid, (-1, 1))

        for chan in range(_x.shape[0]):
            _x[chan] = scipy.ndimage.interpolation.map_coordinates(_x[chan], indices, order=2, mode='reflect').reshape(
                imgShape)
            _x[chan] = np.clip(_x[chan], 0., 1.)
        if _y is not None:
            for chan in range(_y.shape[0]):
                _y[chan] = scipy.ndimage.interpolation.map_coordinates(_y[chan], indices, order=2,
                                                                       mode='reflect').reshape(imgShape)
                _y[chan] = np.clip(_y[chan], 0., 1.)

                # if _pts is not None:

    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    if _rot != 0:
        theta = np.pi / 180 * np.random.uniform(-_rot, _rot)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        matrix = np.dot(matrix, rotation_matrix)

    if _trans != 0:
        ty = np.random.uniform(-_trans, _trans) * _x.shape[_row_index]
        tx = np.random.uniform(-_trans, _trans) * _x.shape[_col_index]
        translation_matrix = np.array([[1, 0, ty],
                                       [0, 1, tx],
                                       [0, 0, 1]])
        matrix = np.dot(matrix, translation_matrix)

    if _shear != 0:
        shear = np.random.uniform(-_shear, _shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        matrix = np.dot(matrix, shear_matrix)

    if _zoom != 0:
        zx, zy = np.random.uniform(1 - _zoom, 1 + _zoom, 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        matrix = np.dot(matrix, zoom_matrix)

    h, w = _x.shape[_row_index], _x.shape[_col_index]
    transformMatrix = transform_matrix_offset_center(matrix, h, w)
    _x = apply_transform(_x, transformMatrix, _channel_index, _fill_mode, _cval)
    if _y is not None:
        _y = apply_transform(_y, transformMatrix, _channel_index, _fill_mode, _cval)

    if _pts is not None:
        matrix = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        if _rot != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            matrix = np.dot(matrix, rotation_matrix)

        if _trans != 0:
            translation_matrix = np.array([[1, 0, -tx],
                                           [0, 1, -ty],
                                           [0, 0, 1]])
            matrix = np.dot(translation_matrix, matrix)

        if _shear != 0:
            shear_matrix = np.array([[np.cos(shear), 0, 0],
                                     [-np.sin(shear), 1, 0],
                                     [0, 0, 1]])
            shear_matrix = np.linalg.inv(shear_matrix)  # TODO write the inverse properly without computing it
            matrix = np.dot(shear_matrix, matrix)

        if _zoom != 0:
            zoom_matrix = np.array([[1. / zy, 0, 0],
                                    [0, 1. / zx, 0],
                                    [0, 0, 1]])
            matrix = np.dot(zoom_matrix, matrix)

        transformMatrix = transform_matrix_offset_center(matrix, h, w)
        _pts = np.dot(_pts, transformMatrix.T)

    return _x, _y, _pts

def train_generator(_X, _Y,_batchSize, iter_times, _keepPctOriginal=0.5, _trans=TRANSLATION_FACTOR, _rot=ROTATION_FACTOR, _zoom=ZOOM_FACTOR, _shear=SHEAR_FACTOR, _elastix=VECTOR_FIELD_SIGMA, _intensity=INTENSITY_FACTOR, _hflip=True, _vflip=True):
    n_data=_X.shape[0]
    shapeX = _X.shape
    shapeY = _Y.shape
    currentBatch=0
    while 1:
        index=np.random.permutation(n_data)
        X=_X[index,:,:,:]
        Y=_Y[index,:,:,:]
        for i in range(iter_times):
            if currentBatch==0:
                x = np.empty((_batchSize, 3, shapeX[2], shapeX[3]), dtype=np.float32)
                y = np.empty((_batchSize, 1, shapeY[2], shapeY[3]), dtype=np.float32)
            index_list=random.randint(0,n_data-1)
            img_x=X[index_list]
            img_y=Y[index_list]
            if random.random() > _keepPctOriginal:
                if _intensity != 0:
                    img_x = random_channel_shift(img_x, _intensity)

                img_x, img_y, _ = ApplyRandomTransformations(img_x, img_y, None, _trans, _rot, _zoom, _shear, _elastix)
                if _hflip == True and random.random() > 0.5:
                    img_x = flip_axis(img_x, 1)
                    img_y = flip_axis(img_y, 1)
                if _vflip == True and random.random() > 0.5:
                    img_x = flip_axis(img_x, 2)
                    img_y = flip_axis(img_y, 2)
            x[currentBatch][...]=img_x[...]
            y[currentBatch][...]=img_y[...]
            currentBatch+=1

            if currentBatch==_batchSize:
                currentBatch=0
                # yield (x,y)
                yield [x, y], []
            elif i ==iter_times-1:
                # yield (x[:currentBatch], y[:currentBatch])
                yield [x[:currentBatch], y[:currentBatch]], []
                # yield (np.copy(x[:currentBatch]), np.copy(y[:currentBatch]))
                currentBatch = 0

def validation_generator(_X, _Y,_batchSize):
    n_data = _X.shape[0]
    shapeX = _X.shape
    shapeY = _Y.shape
    currentBatch = 0
    index = np.random.permutation(n_data)
    X = _X[index, :, :, :]
    Y = _Y[index, :, :, :]
    while 1:
        for i in range(n_data):
            if currentBatch == 0:
                x = np.empty((_batchSize, 3, shapeX[2], shapeX[3]), dtype=np.float32)
                y = np.empty((_batchSize, 1, shapeY[2], shapeY[3]), dtype=np.float32)
            index_list = random.randint(0, n_data-1)
            img_x = X[index_list]
            img_y = Y[index_list]

            x[currentBatch][...] = img_x[...]
            y[currentBatch][...] = img_y[...]
            currentBatch += 1
            if currentBatch == _batchSize:
                currentBatch = 0
                # yield (x, y)
                yield [x, y], []
            elif i==n_data-1:
                # yield (x[:currentBatch], y[:currentBatch])
                yield [x[:currentBatch], y[:currentBatch]], []
                currentBatch = 0

def test_generator(_X, _Y,_batchSize):
    n_data = _X.shape[0]
    shapeX = _X.shape
    shapeY = _Y.shape
    currentBatch = 0
    # index = np.random.permutation(n_data)
    # X = _X[index, :, :, :]
    # Y = _Y[index, :, :, :]
    while 1:
        for i in range(n_data):
            if currentBatch == 0:
                # x = np.empty((_batchSize, 4, shapeX[2], shapeX[3]), dtype=np.float32)
                x = np.empty((_batchSize, 3, shapeX[2], shapeX[3]), dtype=np.float32)
                y = np.empty((_batchSize, 1, shapeY[2], shapeY[3]), dtype=np.float32)
            # index_list = random.randint(0, n_data-1)
            # img_x = X[index_list]
            # img_y = Y[index_list]
            img_x = _X[i]
            img_y = _Y[i]

            x[currentBatch][...] = img_x[...]
            y[currentBatch][...] = img_y[...]
            currentBatch += 1
            if currentBatch == _batchSize:
                currentBatch = 0
                # yield (x, y)
                yield [x, y], []
            elif i==n_data-1:
                # yield (x[:currentBatch], y[:currentBatch])
                yield [x[:currentBatch], y[:currentBatch]], []
                currentBatch = 0
