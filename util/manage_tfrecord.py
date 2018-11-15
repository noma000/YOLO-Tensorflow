from util.tool import IOU
from util.tool import Inarea
import random
import math
from util.image_augmentation import *
from setting.config_manager import ConfigDecoder
from random import shuffle
import tensorflow as tf
from scipy import ndimage, misc


class batch_reader:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get_option("batch_size")
        self.data_path = config.get_path("data_path")
        self.train_file = self.data_path + "train.txt"#config.get_path("train_path")
        self.test_file = self.data_path + "test.txt" #config.get_path("test_path")

        self.tf_training_c = config.get_path("train_path_f")
        self.tf_testing_c = config.get_path("test_path_f")

        self.nb_class_c = config.get_finetraining("number_class")
        self.nb_class_d = config.get_training("number_class")
        self.shape = config.get_model("input_shape")
        self.anchors = config.get_model("anchor")
        self.anchor_len = len(self.anchors)
        self.fine_tuning_mode = config.get_model("finetunning")
        self.last_grid = config.get_model("output_shape")

        self.data_path_c = config.get_path("data_path_f")
        self.rnd_index = np.arange(self.batch_size)
        self.train_batch_index = 0
        self.test_batch_index = 0
        self.aug = image_aug()
        self.scaling_value = 10

        if self.fine_tuning_mode == True:
            #self.call_func = self.read_classfication_btreader
            # for finetunning (classification tensor)
            self.image_tr, self.label_tr = self.read_classfication_tform([self.tf_training_c])
            self.image_te, self.label_te = self.read_classfication_tform([self.tf_testing_c])
        else:
            #self.call_func = self.read_detect_btreader
            self.train_data = []
            self.test_data = []

            #self.nb_train = config.get_training("trdata_nb")
            with open(self.train_file,'r') as tf:
                self.train_file_list = tf.readlines()

            with open(self.test_file,'r') as tf:
                self.test_file_list = tf.readlines()

            self.nb_train = len(self.train_file_list)
            self.nb_test = len(self.test_file_list)

            print("Loading_Traindata")
            print("load_path : " + self.train_file)
            print("Number of data : " + str(self.nb_train))
            for file_name in self.train_file_list:
                file_name = file_name.rstrip()
                image_path = self.data_path + file_name + '.jpg'
                box_info = []
                clazz = []
                with open(self.data_path + file_name + '.txt','r' ) as fr:
                    annotations = fr.readlines()
                    annotations = [s.rstrip() for s in annotations]
                    for anno in annotations:
                        anno = anno.split(' ')
                        clazz.append(int(anno[0]))
                        box_info.append([float(i) for i in anno[1:]])
                self.train_data.append((image_path, box_info, clazz))

            print("Loading_Testdata")
            print("load_path : " + self.test_file)
            print("Number of data : " + str(self.nb_test))
            for file_name in self.test_file_list:
                file_name = file_name.rstrip()
                image_path = self.data_path + file_name + '.jpg'
                box_info = []
                clazz = []
                with open(self.data_path + file_name + '.txt','r' ) as fr:
                    annotations = fr.readlines()
                    annotations = [s.rstrip() for s in annotations]
                    for anno in annotations:
                        anno = anno.split(' ')
                        clazz.append(int(anno[0]))
                        box_info.append([float(i) for i in anno[1:]])
                self.test_data.append((image_path, box_info, clazz))

            print("Loading data is done...")

    def read_nbtrain(self):
        return self.nb_train
    def read_nbtest(self):
        return self.nb_test
    def label(self,training = True):
    
        if training == True:
            data_list = self.train_data
            data_index = self.train_batch_index % (int(self.nb_train/self.batch_size)) * self.batch_size
            self.train_batch_index = self.train_batch_index + self.batch_size
        else:
            data_list = self.test_data
            data_index = self.test_batch_index % (int(self.nb_test/self.batch_size)) * self.batch_size
            self.test_batch_index = self.test_batch_index + self.batch_size
        batch_image = []
        batch_label = []
        for bi in range(0, self.batch_size):
            image_path, box_info, clazz = data_list[data_index + bi]
            batch_label.append(box_info)
        return np.array(batch_label)

    def ground_truth(self, training = True):
        if training == True:
            data_list = self.train_data
            data_index = self.train_batch_index % (int(self.nb_train/self.batch_size)) * self.batch_size
            self.train_batch_index = self.train_batch_index + self.batch_size
        else:
            data_list = self.test_data
            data_index = self.test_batch_index % (int(self.nb_test/self.batch_size)) * self.batch_size
            self.test_batch_index = self.test_batch_index + self.batch_size

        batch_image = []
        batch_label = []

        for bi in range(0, self.batch_size): # must impliment this area
            image_path, box_info, clazz = data_list[data_index + bi]
            image = cv2.imread(image_path)
            image = np.asarray(image, np.uint8)
            image, box_info = image_padding(self.shape, image, box_info)
            batch_image.append(image)
            try:
                claz_reshape = np.reshape(clazz,[-1,1])
                batch_label.append(np.concatenate([box_info,claz_reshape], axis=1))
            except Exception as e :
                print(str(e))
                batch_label.append([])

        return np.array(batch_image), np.array(batch_label)




    def read_batch(self,training=True):
        image_list = []
        lg = self.last_grid
        batch_label = np.zeros((self.batch_size, self.last_grid*self.last_grid, len(self.anchors),
                                5 + 4 + self.nb_class_d), dtype=np.float32)
        if training == True:
            data_list = self.train_data
            data_index = self.train_batch_index % (int(self.nb_train/self.batch_size)) * self.batch_size
            self.train_batch_index = self.train_batch_index + self.batch_size
        else:
            data_list = self.test_data
            data_index = self.test_batch_index % (int(self.nb_test/self.batch_size)) * self.batch_size
            self.test_batch_index = self.test_batch_index + self.batch_size

        # make box grid per batch
        for bi in range(0, self.batch_size):
            image_path, box_info, clazz = data_list[data_index + bi]

            shape = self.shape
            image = cv2.imread(image_path)
            image = np.asarray(image, np.uint8)
            # augumentation / scale flip recolor
            if training == True:
                image = imcv2_recolor(image)
                image = self.aug.image_augmentation([image])[0]
                image, box_info = image_padding(shape, image, box_info)
                image, box_info = imcv2_affine_trans(image, box_info,scale_range=self.scaling_value)
                #image = imcv2_rotate(image, size)
            else:
                image, box_info = image_padding(shape, image, box_info)

            image_list.append(image)

            label_list = []
            for i, box in enumerate(box_info):
                cx,cy,w,h = box
                box_left, box_right, box_up, box_down = [cx - w * .5, cx + w * .5, cy - h * .5, cy + h * .5]
                if (box_right < 0.0) | (box_left > 1.0) | (box_down < 0.0) | (box_up > 1.0):
                    continue
                elif (w < 1e-4) | (h < 1e-4):
                    continue
                else:
                    if box_left < 0.0:
                        cx = box_right * .5
                        w = box_right
                    if box_right > 1.:
                        cx = (1. + box_left) * .5
                        w = 1. - box_left
                    if box_up < 0.0:
                        cy = box_down * .5
                        h = box_down
                    if box_down > 1.:
                        cy = (1. + box_up) * .5
                        h = 1. - box_up
                claz = clazz[i]
                min_x, min_y = [cx - w * .5, cy - h * .5]
                max_x, max_y = [cx + w * .5, cy + h * .5]

                batch_index = 0
                max_iou = 0.0
                for a in range(self.anchor_len):
                    anchor = self.anchors[a]
                    anchors_area = [(0, 0), (anchor[0], anchor[1])]
                    box_area = [(0, 0), (w * lg, h * lg)] # only w and h
                    iou = IOU(box_area, anchors_area)
                    if max_iou < iou:
                        batch_index = a
                        max_iou = iou

                '''
                [confidence : cx : cy : w : h : cx-w/2 : cy-h/2 : clazz]
                '''
                box_index = lg * math.floor(cy * lg) + math.floor(cx * lg)

                label_list.append([max_iou, batch_index, box_index,
                                   cx * lg - math.floor(cx * lg), cy * lg - math.floor(cy * lg),
                                   w, h, min_x, min_y, max_x, max_y, claz])

            # match box metrix array with label_list
            check_maxioubox = np.zeros(self.last_grid * self.last_grid)
            label_list.sort(reverse=True)
            for label in label_list:
                omax_iou, obai, obox_index = label[0:3]
                ox,oy,ow,oh,minx,miny,maxx,maxy,oclaz = label[3:12]
                if check_maxioubox[obox_index] < omax_iou:
                    check_maxioubox[obox_index] = omax_iou
                    batch_label[bi, obox_index, obai, 0] = 1.0          # confidence
                    batch_label[bi, obox_index, obai, 1] = ox           # cx * lg - math.floor(cx * lg) # x (0~1)
                    batch_label[bi, obox_index, obai, 2] = oy           # cy * lg - math.floor(cy * lg) # y (0~1)
                    batch_label[bi, obox_index, obai, 3] = ow + 1e-5    # W (0~1)
                    batch_label[bi, obox_index, obai, 4] = oh + 1e-5    # H (0~1)
                    batch_label[bi, obox_index, obai, 5] = minx         # min_x
                    batch_label[bi, obox_index, obai, 6] = miny         # min_y
                    batch_label[bi, obox_index, obai, 7] = maxx         # max_x
                    batch_label[bi, obox_index, obai, 8] = maxy         # max_y
                    batch_label[bi, obox_index, obai, 9 + oclaz] = 1.0 # classification
        # box shuffle
        batch_image = np.array(image_list)
        np.random.shuffle(self.rnd_index)
        batch_image = batch_image[self.rnd_index]
        batch_label= batch_label[self.rnd_index]

        return batch_image, batch_label

    # used in fine_tunning before but ignore this code now
    def read_train_c(self):
        image_list = []
        batch_label = np.zeros((self.batch_size, self.nb_class_c), dtype=np.float32)
        #if training == True:
        data_tensor = [self.image_tr, self.label_tr]
        #else:
        #    data_tensor = [self.image_te, self.label_te]

        for bi in range(0, self.batch_size):
            image_path, clazz = self.sess.run(data_tensor)
            #image = Image.open(self.data_path_c + image_path.decode("utf-8"))
            try:
                image = cv2.imread(self.data_path_c + image_path.decode("utf-8"))
            except:
                print("[*] change grey to color : " + self.data_path_c + image_path.decode("utf-8"))
                image = cv2.imread(self.data_path_c + image_path.decode("utf-8"),cv2.IMREAD_GRAYSCALE)
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            #image = image.convert("RGB")
            h,w,c = image.shape

            image = np.asarray(image, np.uint8)
            image = imcv2_recolor(image)
            image = image_padding_c(self.shape, image)
            image, channel, affine_info = imcv2_affine_trans(image,(h,w),scale_range=10)

            image = cv2.resize(image, (self.shape, self.shape))
            image_list.append(image)
            batch_label[bi][clazz[0][0]] = 1.0

        batch_image = np.array(image_list)
        #if training ==  True:
        batch_image = self.aug.image_augmentation(batch_image)
        #np.random.shuffle(self.rnd_index)
        #batch_image = batch_image[self.rnd_index]
        #batch_label = batch_label[self.rnd_index]
        return batch_image, batch_label

    '''
     function : shuffle train_data and test_data & change scaling rate
    '''
    def data_shuffle(self):
        shuffle(self.train_data)
        shuffle(self.test_data)
        print("[-] Data shuffle")
        self.scaling_value = random.uniform(5,10)
        print("[-] Scaling value =" + str(self.scaling_value))


def draw_grid_lable():
    sess = tf.Session()
    config = ConfigDecoder("./setting/window_configure.json")
    batch_load = batch_reader(sess, config)
    init = tf.global_variables_initializer()
    sess.run(init)

    nb_train_samples = config.get_training("trdata_nb")
    nb_test_samples = config.get_training("tedata_nb")
    batch_size = 1

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

    truths = []
    scores = []
    train_batch = int(nb_train_samples / batch_size)
    test_batch = int(nb_test_samples / batch_size)

    for index, count_batch in enumerate(range(test_batch)):
        batch_image, batch_label_box = batch_load.read_batch(training=True)
        for i in range(0,20):
            #temp = []
            for j in range(0,169):
                for a in range(0,5):
                    if batch_label_box[i][j][a][0] == 1.0:
                        x,y = [divmod(j,13)[0], divmod(j,13)[1]]

                        cx = (batch_label_box[i][j][a][1] * 32 + y * 32) / 416
                        cy = (batch_label_box[i][j][a][2] *32 + x * 32) / 416
                        w = batch_label_box[i][j][a][3]
                        h = batch_label_box[i][j][a][4]
                        lt = [int((cx - w * .5) * 416), int((cy - h * .5) * 416)]
                        rd = [int((cx + w * .5) * 416), int((cy + h * .5) * 416)]
                        cv2.rectangle(batch_image[i], (lt[0], lt[1]), (rd[0], rd[1]), (255, 0, 0), 2)
                        lt = [int(batch_label_box[i][j][a][5] * 416),int(batch_label_box[i][j][a][6] * 416)]
                        rd = [int(batch_label_box[i][j][a][7] * 416), int(batch_label_box[i][j][a][8] * 416)]
                        cv2.rectangle(batch_image[i], (lt[0], lt[1]), (rd[0], rd[1]), (0, 255, 0), 1)
                        #temp.append([lt,rd])

            while True:
                cv2.imshow('image', batch_image[i])

                if cv2.waitKey(10) & 0xFF == ord('n'):
                    break

def grid_test():
    sess = tf.Session()
    config = ConfigDecoder("./setting/window_configure.json")
    batch_load = batch_reader(sess, config)
    init = tf.global_variables_initializer()
    sess.run(init)

    nb_train_samples = config.get_training("trdata_nb")
    nb_test_samples = config.get_training("tedata_nb")
    batch_size = 1

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

    truths = []
    scores = []
    for grid in range(1,30):
        print("grid : " + str(grid))
        train_batch = int(nb_train_samples / batch_size)
        test_batch = int(nb_test_samples / batch_size)
        collision = 0
        nb_box = 0
        for index, count_batch in enumerate(range(train_batch )):
            batch_label_box = batch_load.label(training=True)
            colmatrix = np.zeros((grid,grid))
            for i, label in enumerate(batch_label_box[0]):
                Lmin_x = int((label[0] - label[2] / 2) * grid)
                Lmin_y = int((label[1] - label[3] / 2) * grid)
                Lmax_x = int((label[0] + label[2] / 2) * grid)
                Lmax_y = int((label[1] + label[3] / 2) * grid)
                if colmatrix[int(label[0] * grid)][int(label[1] * grid)] > 0:
                    collision += 1
                    nb_box += 1
                else:
                    nb_box += 1
                    #print("c")
                    colmatrix[int(label[0] * grid)][int(label[1] * grid)] += 1
        print("nb_box : " + str(nb_box))
        print("collsion : " + str(collision))
        print("rate :" + str(collision/nb_box))


def image_padding_test():
    sess = tf.Session()
    config = ConfigDecoder("./setting/window_configure.json")
    batch_load = batch_reader(sess, config)
    init = tf.global_variables_initializer()
    sess.run(init)

    nb_train_samples = config.get_training("trdata_nb")
    nb_test_samples = config.get_training("tedata_nb")
    batch_size = 1

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

    truths = []
    scores = []
    train_batch = int(nb_train_samples / batch_size)
    test_batch = int(nb_test_samples / batch_size)
    confusion_matrix = np.zeros((81, 81))  # coco dataset has 80' class so (add fault detect +1)

    for index, count_batch in enumerate(range(test_batch)):
        print("image : " + str(index))
        batch_image, batch_label_box = batch_load.ground_truth(training=False)
        H, W, _ = batch_image[0].shape
        b, g, r = cv2.split(batch_image[0])
        batch_image[0] = cv2.merge([r,g,b])
        for i, label in enumerate(batch_label_box[0]):
            Lmin_x = int((label[0] - label[2]/2) * W)
            Lmin_y = int((label[1] - label[3]/2) * H)
            Lmax_x = int((label[0] + label[2]/2) * W)
            Lmax_y = int((label[1] + label[3]/2) * H)
            Lclaz = int(label[4])
            cv2.rectangle(batch_image[0], (Lmin_x, Lmin_y), (Lmax_x, Lmax_y), (255, 0, 0), 2)
        while True:
            cv2.imshow('image', batch_image[0])

            if cv2.waitKey(10) & 0xFF == ord('n'):
                break

        cv2.imwrite("./temp/test.jpg", batch_image[0])



if __name__ == '__main__':
    #grid_test()
    #image_padding_test()
    #grid_test()
    draw_grid_lable()
    #image_padding_test()

