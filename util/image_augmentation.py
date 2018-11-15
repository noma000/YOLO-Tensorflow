from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from imgaug import augmenters as iaa
import numpy as np
import cv2

# recolor
def imcv2_recolor(im, a=.1):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.
    # random amplify each channel

    im = im * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    # 	im = np.power(im/mx, 1. + up * .5)
    im = cv2.pow(im / mx, 1. + up * .5)
    return np.array(im * 255., np.uint8)

# scale, flip, shift
def imcv2_affine_trans(image,boxs,scale_range):
    # Scale and translate
    #w, h = size
    H, W, _ = image.shape
    scale =  np.random.uniform() / scale_range + 1 #np.random.uniform(10, 10.0) /10 + 1.
    max_offx = (scale - 1.) * W
    max_offy = (scale - 1.) * H
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    image = image[offy: (offy + H), offx: (offx + W)]
    flip = np.random.binomial(1, .5)
    if flip: image = cv2.flip(image, 1)
    offset = [offx / W, offy / H]
    new_boxinfo = []
    for info in boxs:
        # cx, cy, w, h
        new_boxinfo.append([(info[0]*scale - offset[0])*(1.0-flip) +
                                (1-info[0]*scale + offset[0])*flip,
                                info[1]*scale - offset[1],
                     info[2]*scale,
                     info[3]*scale])
    cv2.resize(image,(W,H))
    return image, new_boxinfo

# Not used yet
def imcv2_rotate(im, size):
    w, h = (416,416)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
    ro_img = cv2.warpAffine(im, M, (w, h))
    return ro_img

# Image padding  & resize for detector
def image_padding(SZ_NEW,image, boxs):
    H, W, _ = image.shape
    width_padding, height_padding = [0, 0]

    # calculate ratio and padding image
    if W > H:
        height_padding = (W - H) // 2
        padding_square = np.full((height_padding, W, 3), [0, 0, 0], dtype=np.uint8)
        image = np.concatenate((padding_square, np.concatenate((image,padding_square),axis=0)),axis=0)
    elif H > W:
        width_padding = (H - W) // 2
        padding_square = np.full((H, width_padding, 3), [0, 0, 0], dtype=np.uint8)
        image = np.concatenate((padding_square, np.concatenate((image, padding_square), axis=1)), axis=1)
    image = cv2.resize(image, (SZ_NEW, SZ_NEW))
    bgrid = max([W,H])

    new_box = []
    # info = [cx,cy,w,h, class] include ratio
    for i, info in enumerate(boxs):
        cx,cy,w,h = [(info[0] * W + width_padding) / bgrid,
                        (info[1] * H + height_padding) / bgrid,
                        info[2] * W / bgrid,
                        info[3] * H / bgrid]
        new_box.append([cx,cy,w,h])

    return image, np.array(new_box)

# Image padding  & resize for classifier
def image_padding_c(SZ_NEW,image):
    H, W, _ = image.shape

    # calculate ratio and padding image
    if W > H:
        height_padding = (W - H) // 2
        padding_square = np.full((height_padding, W, 3), [0, 0, 0], dtype=np.uint8)
        image = np.concatenate((padding_square, np.concatenate((image,padding_square),axis=0)),axis=0)
    elif H > W:
        width_padding = (H - W) // 2
        padding_square = np.full((H, width_padding, 3), [0, 0, 0], dtype=np.uint8)
        image = np.concatenate((padding_square, np.concatenate((image, padding_square), axis=1)), axis=1)
    image = cv2.resize(image, (SZ_NEW, SZ_NEW))

    return image

class image_aug():
    def __init__(self):
        ### augmentors by https://github.com/aleju/imgaug
        # sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                # sometimes(iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                # rotate=(-5, 5), # rotate by -45 to +45 degrees
                # shear=(-5, 5), # shear by -16 to +16 degrees
                # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.Sometimes(0.2,
                              [
                                  # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                                  iaa.OneOf([
                                      # iaa.Noop(),
                                      iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                      iaa.AverageBlur(k=(2, 7)),
                                      # blur image using local means with kernel sizes between 2 and 7
                                      iaa.MedianBlur(k=(3, 11)),
                                      iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                                      iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
                                      iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                                      iaa.Multiply((0.5, 1.0), per_channel=0.5),
                                      iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                      # randomly remove up to 10% of the pixels
                                      iaa.CoarseDropout((0.03, 0.15), size_percent=(0.5, 1.5), per_channel=0.2),
                                      iaa.AddToHueAndSaturation((- 20, 20)),
                                      iaa.Invert(0.05, per_channel=True), # invert color channels
                                      iaa.Add((-10, 10), per_channel=0.5),
                                      # change brightness of images (by -10 to 10 of original value)
                                      iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                      # change brightness of images (50-150% of original value)
                                      iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                      ]),
                                      # iaa.AddToHueAndSaturation((- 10,10 )),

                                      # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                      # blur image using local medians with kernel sizes between 2 and 7

                                  # iaa.Add((-10, 10), per_channel=0.5),
                                  # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                                  # search either for all edges or for directed edges
                                  # sometimes(iaa.OneOf([
                                  #    iaa.EdgeDetect(alpha=(0, 0.7)),
                                  #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                                  # ])),
                                  ##iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                  # add gaussian noise to images
                                  #iaa.Sometimes(0.2, [
                                  #    iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                      # randomly remove up to 10% of the pixels
                                  #    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.5, 1.5), per_channel=0.2),
                                  #]),
                                  # iaa.AddToHueAndSaturation((- 20, 20)),
                                  # iaa.Invert(0.05, per_channel=True), # invert color channels
                                  ##iaa.Add((-10, 10), per_channel=0.5),
                                  # change brightness of images (by -10 to 10 of original value)
                                  ##iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                  # change brightness of images (50-150% of original value)
                                  ##iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                  # improve or worsen the contrast
                                  # iaa.Grayscale(alpha=(0.0, 1.0)),
                                  # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                                  # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                              ],
                              # random_order=True
                              )
            ],
            random_order=True
        )

    def image_augmentation(self,batch_images):
        batch_image = self.seq.augment_images(batch_images)
        return batch_image
