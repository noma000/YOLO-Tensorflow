'''
Model Name : YOLO v2
Input shape : 416,416,3

'''
import warnings

import cv2
import numpy as np
import tensorflow as tf
from model.Network import *
from model.layers import Titan
from util.image_augmentation import *
from util.tool import IOU

class Model:
    def __init__(self, config, name=""):
        self.sess = tf.Session()
        self.name = name
        self.model_name = config.get_model("name")
        self.anchor = config.get_model("anchor")

        self.cell_wh = config.get_model("output_shape")
        self.finetunning = config.get_model("finetunning")

        self.input_grid = config.get_model("input_shape")
        self.const_coord = tf.constant(config.get_const("coordinate"), dtype=tf.float32)
        self.const_obj = tf.constant(config.get_const("obj"), dtype=tf.float32)
        self.const_noobj = tf.constant(config.get_const("noobj"), dtype=tf.float32)
        self.const_prob = tf.constant(config.get_const("probability"), dtype=tf.float32)
        self.TB_path = config.get_path("TB_logpath") # tensorboard path
        self.learing_rate = tf.placeholder(tf.float32, None)
        self.nb_class_c = config.get_finetraining("number_class")
        self.nb_class_d = config.get_training("number_class")
        self.max_to_keep = config.get_option("max_to_keep")
        self.TB_activate = config.get_option("tensorboard_use")  # tensorboard activate
        self.batch_size = config.get_option("batch_size")
        self.model_path = config.get_path("load_model")
        self.use_pretraining_model = config.get_option("use_pretrainmodel")
        self.use_finetunning_model = config.get_option("use_finemodel")
        self.training = tf.placeholder(tf.bool)
        self.TI = Titan(self.sess, logging= self.TB_activate, TB_PATH=self.TB_path,fine_tunning = self.finetunning)
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.input_grid, self.input_grid, 3])

        # Normalize term
        self.input = self.input / 255.0

        # Model select
        if self.model_name == "darknet_full":
            self.NModel = darknet_full.Network(self.input, config, self.TI, self.training)
        elif self.model_name == "darknet_tiny":
            self.NModel = darknet_tiny.Network(self.input, config, self.TI, self.training)
        elif self.model_name == "densenet":
            self.NModel = densenet.Network(self.input, config, self.TI, self.training)
        elif self.model_name == "fully_convnet":
            self.NModel = fully_convnet.Network(self.input, config, self.TI, self.training)
        elif self.model_name == "darknet_19":
            self.NModel = darknet_19.Network(self.input, config, self.TI, self.training)
        else:
            warnings.warn("There is no such model", DeprecationWarning)

        if self.finetunning == True:
            self.label = tf.placeholder(tf.float32, [self.batch_size, self.nb_class_c])
            self._classification_loss()
        else:
            label_path = config.get_path("Label_path")
            self.index_name = {}
            self.index_color = {}
            with open(label_path, "r") as f:
                for label_i in range(0, self.nb_class_d):
                    line = f.readline()
                    name = line.split(':')[0]
                    rgb = line.split(':')[1].split(",")
                    color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                    self.index_name[label_i] = name
                    self.index_color[label_i] = color

            self.label = tf.placeholder(tf.float32, [self.batch_size, self.cell_wh * self.cell_wh, len(self.anchor),
                                                     9 + self.nb_class_d])
            self._detection_loss()

        self.sess.run(tf.global_variables_initializer())
        #with self.sess.as_default():
        #    coord = tf.train.Coordinator()
        #    threads = tf.train.start_queue_runners(coord=coord)

        self.saver = tf.train.Saver()
        self.TI.activate_tensorboard()
        self.load_model()



    '''
        function : calculation classfication loss


    '''
    def _classification_loss(self):
        self.logits = self.NModel.net
        #self.logits = self.TI.fc(oup, self.nb_class_c, name=self.name + '_classify', activation="None")
        self.hypothesis = tf.nn.softmax(self.logits)

        self.cost_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.optimizer_f = tf.train.AdamOptimizer(learning_rate=self.learing_rate).minimize(self.cost_f)

        predict_hypo = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict_hypo, tf.float32))

        if self.TB_activate:
            tf.summary.scalar("loss", self.cost_f)
            tf.summary.scalar("acc", self.accuracy)


    '''
        function : calculation detection loss


    '''
    def _detection_loss(self):
        # [grid x*y, 1(no anchor), confidence, cx, cy, w, h, min_x, min_y, max_x, max_y :: class]
        # label_box's data
        self.label_conf = self.label[:, :, :, 0] # confidence
        self.label_xy = self.label[:, :, :, 1:3] # coordinate x,y
        self.label_wh = self.label[:, :, :, 3:5] # coordinate w,h
        self.wh_sqrt = tf.sqrt(self.label[:, :, :, 3:5]) # sqrt width & height
        self.label_coord = tf.concat([self.label_xy, self.wh_sqrt], 3) # concat label

        # box info left_top & right bottom
        # use in calculating iou
        self.label_xy_min = self.label[:, :, :, 5:7]
        self.label_xy_max = self.label[:, :, :, 7:9]
        self.label_clazz = self.label[:, :, :, 9:]

        '''
        Calculate iou between prediction and label
        '''
        lt_xy = tf.maximum(self.label_xy_min, self.NModel.xy_min)
        rb_xy = tf.minimum(self.label_xy_max, self.NModel.xy_max)
        intersect = tf.maximum(rb_xy - lt_xy, 1e-5)
        intersect_area = tf.multiply(intersect[:, :, :, 0], intersect[:, :, :, 1])
        box1_area = self.NModel.areas
        box2_area = tf.multiply(self.label_wh[:, :, :, 0], self.label_wh[:, :, :, 1])
        union_area = tf.maximum(box1_area + box2_area - intersect_area, 1e-5)

        # predict_iou's shape = (-1,169,5)
        predict_iou = tf.identity(tf.truediv(intersect_area, union_area), name="predict_iou")
        best_box = tf.reduce_max(predict_iou, 2, True)
        #best_box = predict_iou * predict_true

        best_box_iou = tf.equal(predict_iou, best_box)
        confs = tf.multiply(tf.to_float(best_box_iou), self.label_conf)


        mask_conf = self.const_noobj * (1.0 - confs) + self.const_obj * confs
        expend_conf = tf.identity(tf.expand_dims(confs, axis=3), name="adjust_axis")
        mask_coord = tf.identity(expend_conf * self.const_coord, name="coord_mask")
        mask_predict = tf.identity(expend_conf * self.const_prob, name="predict_mask")

        #### delima
        ex_conf = tf.expand_dims(confs, axis=3)
        ex_mask_conf = tf.expand_dims(mask_conf, axis=3)
        #self.iou_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.NModel.confi, labels=confs) * mask_conf
        self.iou_loss = tf.square(self.NModel.confi - confs) * mask_conf

        self.coord_loss = tf.square(self.NModel.coords - self.label_coord) * mask_coord
        #self.prob_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.NModel.predict_class,labels=self.label_clazz)  * mask_predict  # * ex_mask_predict
        self.prob_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.NModel.predict_class, labels=self.label_clazz)#* ex_mask_predict
        self.prob_loss = tf.expand_dims(self.prob_loss, axis=3) * mask_predict

        nb_true_box = tf.reduce_sum(tf.to_float(confs)) + 1e-5


        # just for log
        self.iou_l = tf.reduce_sum(self.iou_loss) / nb_true_box * .5
        self.coord_l = tf.reduce_sum(self.coord_loss) / nb_true_box * .5
        self.prob_l = tf.reduce_sum(self.prob_loss) / nb_true_box * .5
        self.loss = self.iou_l + self.coord_l + self.prob_l

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer_d = tf.train.AdamOptimizer(learning_rate=self.learing_rate).minimize(self.loss)

        # class per precision
        # confs = tf.multiply(tf.to_float(best_box_iou), self.label_conf)

        nb_true = tf.reduce_sum(self.label_conf)
        nb_pred = tf.reduce_sum(tf.to_float(confs > 0.5) * tf.to_float(self.NModel.confi > 0.3))
        self.Recall = nb_pred / (nb_true + 1e-5)
        if self.TB_activate:
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Iou_loss", self.iou_l)
            tf.summary.scalar("Coord_loss", self.coord_l)
            tf.summary.scalar("Prob_loss", self.prob_l)
            tf.summary.scalar("Recall", self.Recall)

    def Train_Network(self):
        print('Working')

    def close(self):
        self.sess.close()

    def train(self, x_data, y_data, lr, training=True):
        if self.finetunning == True:
            return self.sess.run([self.cost_f, self.optimizer_f, self.accuracy], feed_dict={
                self.input: x_data, self.label: y_data, self.learing_rate: lr,  self.training: training})
        else:
            return self.sess.run([self.loss, self.optimizer_d, self.Recall,self.coord_l,self.iou_l,self.prob_l],
                             feed_dict={self.input: x_data, self.label: y_data, self.learing_rate: lr, self.training: training})

    def testing(self, x_data, y_data, training=False):
        if self.finetunning == True:
            return self.sess.run([self.cost_f, self.accuracy],
                                 feed_dict={self.input: x_data, self.label: y_data, self.training: training})
        else:
            return self.sess.run([self.loss, self.Recall,self.coord_l,self.iou_l,self.prob_l],
                             feed_dict={self.input: x_data, self.label: y_data, self.training: training})

    def save_model(self,path,epoch):
        self.saver.save(self.sess, path, global_step=epoch)

    def load_model(self):
        if self.use_pretraining_model:
            self.saver.restore(self.sess, self.model_path)
        elif self.use_finetunning_model:
            node_dic = self.NModel.get_fine_tunning_area()
            self.saver = tf.train.Saver(node_dic)
            self.saver.restore(self.sess, self.model_path)
        self.saver = tf.train.Saver()
        print("model load :" + self.model_path)

    def inarea(self, c):
        if c < 0:
            return 0.0
        elif c > 1.0:
            return 1.0
        return c

    # THIS PROCESS IS RUNNING IN CPU,IT'S VERY SLOW
    # WE CHANGE THIS CODE TO tf.image.non_max_suppression FUNCTION
    # CHECK MODEL/NETWORK/DARKNET_FULL > def preprocession
    def NMS(self, xy_min, xy_max, confidence, confidence_th=0.5, iou_th=0.5):
        # usually grid meaning 169, anchor 5, nb_class is 20 in pascal voc
        grid, nb_anchors, nb_class = confidence.shape
        nb_min_offset = xy_min.shape[2]
        nb_max_offset = xy_max.shape[2]
        nb_box = grid * nb_anchors
        boundbox = np.concatenate([confidence, xy_min, xy_max], axis=-1).reshape(
            (nb_box, nb_class + nb_min_offset + nb_max_offset))
        for clz_indx in range(0, nb_class):
            boundbox = boundbox[np.lexsort(([1, -1] * boundbox[:, [1, clz_indx]]).T)]
            for box_index in range(0, nb_box):
                if boundbox[box_index][clz_indx] < confidence_th:
                    boundbox[box_index][clz_indx] = 0.0
                    continue
                for box_offset in range(box_index + 1, nb_box):
                    n_offset_xy = [(boundbox[box_index][nb_class], boundbox[box_index][nb_class + 1]),
                                   (boundbox[box_index][nb_class + 2], boundbox[box_index][nb_class + 3])]
                    p_offset_xy = [(boundbox[box_offset][nb_class], boundbox[box_offset][nb_class + 1]),
                                   (boundbox[box_offset][nb_class + 2], boundbox[box_offset][nb_class + 3])]
                    if IOU(n_offset_xy,p_offset_xy) >= iou_th:
                        boundbox[box_offset][clz_indx] = 0.0

        extracted_box = []
        for bidx in range(0, nb_box):
            if np.max(boundbox[bidx][:nb_class]) < confidence_th:
                continue
            else:
                extracted_box.append((boundbox[bidx]))
        return extracted_box

    def makebox(self, boxes, orgin_shape):
        W, H = orgin_shape
        #W, H = (640, 480)
        #W, H = (1280, 642)
        nb_box = str(boxes.__len__())
        print("nb_box : " + str(nb_box))
        class_len = self.nb_class_d
        detect_box = []

        for i, box in enumerate(boxes):

            class_confi = max(box[:class_len])
            class_number = np.argmax(box[:class_len])
            class_name = self.index_name[class_number]#class_number#
            color = self.index_color[class_number]
            min_x, min_y = int(self.inarea(box[class_len]) * W), int(self.inarea(box[class_len + 1]) * H)
            max_x, max_y = int(self.inarea(box[class_len + 2]) * W), int(self.inarea(box[class_len + 3]) * H)
            detect_box.append((min_x, min_y, max_x, max_y, class_confi,class_number, class_name , color))
        return detect_box


    def predict(self, x_test):
        image_shapes = []
        resize_img = []
        pred_box = []
        for img in x_test:
            H, W, C = img.shape
            image_shapes.append((W,H))
            img = cv2.resize(img,(self.input_grid, self.input_grid))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resize_img.append(img)
        resize_img = np.array(resize_img)
        xy_min, xy_max, conf = self.sess.run([self.NModel.xy_min_pr, self.NModel.xy_max_pr, self.NModel.prediction],
                             feed_dict={self.input: resize_img, self.training: False})

        for i in range(0,self.batch_size):
            refined_box = self.NMS(xy_min[i], xy_max[i], conf[i], 0.3, 0.5)
            boxs = self.makebox(refined_box,image_shapes[i])
            pred_box.append(boxs)
        return pred_box


    def predict_V2(self, image):
        H, W, C = image.shape
        #img = image_padding_c(self.input_grid, image)
        img = cv2.resize(image,(self.input_grid, self.input_grid))

        # nms coordinate[-1,4], score[-1,1], clazz[-1,1]
        nms, score, claz = self.sess.run(
            [self.NModel.NMS, self.NModel.SCORE, self.NModel.CLAZZ],
                    feed_dict={self.input: np.array([img]), self.training: False})


        real_coord = np.multiply(nms,[W,H,W,H]).astype(int)
        label_info = []
        # class_number#
        for i in range(0,len(claz)):
            #x1, y1, x2, y2 = nms[i]
            x1, y1, x2, y2 = np.multiply(np.clip(nms[i], 0, 1),[W,H,W,H]).astype(int)
            s = score[i]
            cn = claz[i]
            class_name = self.index_name[claz[i]]
            color = self.index_color[claz[i]]
            label_info.append((x1,y1,x2,y2,s,cn,class_name, color))

        return label_info

    def Predict_All_Box(self,image):
        #H, W, C = image.shape
        #img = cv2.resize(image,(self.input_grid, self.input_grid))

        # all Box coordinate[-1,4], score[-1,1], clazz[-1,1]
        boxes, score, claz = self.sess.run(
            [self.NModel.boxes, self.NModel.score, self.NModel.class_i],
                    feed_dict={self.input: np.array([image]), self.training: False})
        new_boxes = []
        for box in boxes:
            new_boxes.append(np.clip(box, 0, 1))
        #real_coord = np.multiply(boxes,[W,H,W,H]).astype(int)
        #label_info = []
        # class_number#
        #for i in range(0,len(claz)):
        #    x1, y1, x2, y2 = boxes[i]
        #    s = score[i]
        #    cn = claz[i]
        #    class_name = self.index_name[claz[i]]
        #    color = self.index_color[claz[i]]
        #    label_info.append((x1,y1,x2,y2,s,cn,class_name, color))
        # np.array(new_boxes)
        return [np.array(new_boxes), score, claz]
