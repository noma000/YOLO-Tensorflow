import numpy as np
import tensorflow as tf

'''
    "network": {
        "anchor": [[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]],
        "name": "darknet_full",
        "input_shape" : 416,
        "output_shape" : 13,
        "finetunning" : false
    },
'''

class Net:
    def __init__(self,input_image, config, struct, training=True):
        self.training = training
        self.batch_size = config.get_option("batch_size")
        self.anchors = config.get_model("anchor")
        self.nb_class_d = config.get_training("number_class")
        self.nb_class_c = config.get_finetraining("number_class")
        self.finetunning = config.get_model("finetunning")
        self.Input_Image = input_image
        self.max_output_size = config.get_nms("max_output_size")
        self.iou_threshold = config.get_nms("iou_threshold")
        self.score_threshold = config.get_nms("score_threshold")
        self.TI = struct
        self.net = self.net_create()
        if self.finetunning == False:
            self.preprocessing(self.net)




    # calculation cell
    def calc_cell_xy(self,cell_height, cell_width, dtype=np.float32):
        cell_base = np.zeros([cell_height, cell_width, 2], dtype=dtype)
        for y in range(0,cell_height):
            for x in range(0,cell_width):
                cell_base[y, x, :] = [x, y]
        return cell_base

    def net_add(self,D,name):
        print("net_add")

    def get_fine_tunning_area(self):
        print("get_fine_tunning_area")

    def net_create(self):
        print("net_create")

    def preprocessing(self, net):
        '''
        :param box: [batch_nb : anchor : c,x,y,w,h,class]
        :return: None
        '''
        _, W, H, _= net.shape
        last_grid = int(W)
        #W, H = int(W),int(H)
        self.cell_xy_offset = self.calc_cell_xy(last_grid, last_grid).reshape([1, last_grid*last_grid, 1, 2])

        net = tf.reshape(net, [-1, last_grid * last_grid, len(self.anchors), 5 + self.nb_class_d])
        net_sigmoid = tf.nn.sigmoid(net[:, :, :, :3])
        self.net = net[:, :, :, :]
        # IF USE NOT RMS
        self.confi = net[:, :, :, 0]
        # IF USE SIGMOID CROSS ENTROPY
        self.confi_sig = net_sigmoid[:, :, :, 0]


        # x, y is center coordinate ( x : 0~1, y : 0~1 )
        self.xy_tunning = net_sigmoid[:, :, :, 1:3] + self.cell_xy_offset
        self.xy_tunning = self.xy_tunning / np.reshape([last_grid, last_grid], [1, 1, 1, 2])

        # w, h is size ( w : 0~1, h : 0~1 )
        self.value_wh = tf.exp(net[:, :, :, 3:5]) * np.reshape(self.anchors, [1, 1, len(self.anchors), 2])
        self.value_wh = self.value_wh / np.reshape([last_grid, last_grid], [1, 1, 1, 2])

        ''' add sqrt in '''
        self.sqrt_wh = tf.sqrt(self.value_wh)
        self.coords = tf.concat([net_sigmoid[:, :, :, 1:3], self.sqrt_wh], 3)
        self.areas = tf.multiply(self.value_wh[:, :, :, 0], self.value_wh[:, :, :, 1], name='areas')
        self.xy_min = self.xy_tunning - (self.value_wh * .5)
        self.xy_max = self.xy_tunning + (self.value_wh * .5)


        # class class probability
        self.predict_class = net[:, :, :, 5:]
        self.softmax_predict = tf.nn.softmax(logits=self.predict_class)

        ## For prediction
        self.xy_min_pr = tf.identity(self.xy_min, name='xy_min') #/ np.reshape([last_grid, last_grid], [1, 1, 1, 2])
        self.xy_max_pr = tf.identity(self.xy_max, name='xy_max') #/ np.reshape([last_grid, last_grid], [1, 1, 1, 2])
        #VAL = last_grid * last_grid * len(self.anchors)
        self.prediction = tf.identity(tf.expand_dims(self.confi, axis=3) * self.softmax_predict)


        self.class_i = tf.reshape(tf.argmax(self.prediction,axis=3),[-1])
        self.score = tf.reshape(tf.reduce_max(self.prediction,axis=3),[-1])
        self.boxes = tf.concat([tf.reshape(self.xy_min_pr,[-1,2]), tf.reshape(self.xy_max_pr,[-1,2])],1)

        selcted_index = tf.image.non_max_suppression(self.boxes,self.score,
                                                     max_output_size=self.max_output_size,
                                                     iou_threshold=self.iou_threshold,
                                                     score_threshold=self.score_threshold)
        self.NMS = tf.gather(self.boxes, selcted_index)
        self.SCORE = tf.gather(self.score,selcted_index)
        self.CLAZZ = tf.gather(self.class_i,selcted_index)
        #self.PredictBox =  tf.concat([self.NMS, self.SCORE,self.CLAZZ ],[-1])