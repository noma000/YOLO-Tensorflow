import numpy as np
import tensorflow as tf

# Based on https://github.com/YixuanLi/densenet-tensorflow/blob/master/
'''
    "network": {
        "anchor": [1.8685,3.1954], [5.5800,9.3946], [14.1210,14.8802]]
        "name": "fully_convnet",
        "input_shape" : 312,
        "output_shape" : 20,
        "finetunning" : false
    },
'''

class Network:
    def __init__(self,input_image, config, struct, training=True):
        self.training = training
        self.batch_size = config.get_option("batch_size")
        self.anchors = config.get_model("anchor")
        self.nb_class_d = config.get_training("number_class")
        self.nb_class_c = config.get_finetraining("number_class")
        self.finetunning = config.get_model("finetunning")
        self.Input_Image = input_image
        self.TI = struct
        self.net = self.net_create()
        if self.finetunning == False:
            self.preprocessing(self.net)


# shoud be changed
    def calc_cell_xy(self,cell_height, cell_width, dtype=np.float32):
        cell_base = np.zeros([cell_height, cell_width, 2], dtype=dtype)
        for y in range(0,cell_height):
            for x in range(0,cell_width):
                cell_base[y, x, :] = [x, y]
        return cell_base
    def net_add(self,D,name):

        conv_name = "conv_" + name + "/kernel"
        #bias_name = "conv_" + name + "/bias"
        bn_name_01 = "bn_" + name + "/beta"
        bn_name_02 = "bn_" + name + "/gamma"
        bn_name_03 = "bn_" + name + "/moving_mean"
        bn_name_04 = "bn_" + name + "/moving_variance"


        kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, conv_name)[0]
        #bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bias_name)[0]
        bn_01 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_01)[0]
        bn_02 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_02)[0]
        bn_03 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_03)[0]
        bn_04 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_04)[0]

        D[conv_name] = kernel
        #D[bias_name] = bias
        D[bn_name_01] = bn_01
        D[bn_name_02] = bn_02
        D[bn_name_03] = bn_03
        D[bn_name_04] = bn_04

    def get_fine_tunning_area(self):
        D = {}

        self.net_add(D, name="conv_1")
        self.net_add(D, name="conv_2")
        self.net_add(D, name="conv_3")
        self.net_add(D, name="conv_4")
        self.net_add(D, name="conv_5")
        self.net_add(D, name="conv_6")
        self.net_add(D, name="conv_7")
        self.net_add(D, name="conv_8")
        self.net_add(D, name="conv_9")

        self.net_add(D, name="reduce_conv_1")
        self.net_add(D, name="reduce_conv_2")
        self.net_add(D, name="reduce_conv_3")
        self.net_add(D, name="reduce_conv_4")

        return D



    def net_create(self):
        # image 416x416x3 (RGB), Input Layer
        Titan = self.TI
        #370 -> 368
        x = Titan.bn_conv(self.Input_Image, 64, padding="valid", name='conv_1',activation='relu')
        # 364 -> 182
        x = Titan.bn_conv(x, 128, kernel_size=(3, 3), strides=(2, 2), padding='same', training=self.training,name="reduce_conv_1", activation='relu')
        # 182 -> 180
        x = Titan.bn_conv(x, 64, kernel_size=(1, 1), padding="valid", training=self.training, name="conv_2",activation='relu')
        x = Titan.bn_conv(x, 128, kernel_size=(3, 3), padding="valid",training=self.training, name="conv_3", activation='relu')
        # 180 -> 90
        x = Titan.bn_conv(x, 256, kernel_size=(3, 3), strides=(2, 2), padding='same', training=self.training, name="reduce_conv_2", activation='relu')
        # 90 -> 88
        x = Titan.bn_conv(x, 128, kernel_size=(1, 1), padding="valid", training=self.training, name="conv_4",activation='relu')
        x = Titan.bn_conv(x, 256, kernel_size=(3, 3),padding="valid", training=self.training, name="conv_5", activation='relu')
        # 88 -> 44
        x = Titan.bn_conv(x, 512, kernel_size=(3, 3),strides=(2, 2),padding='same', training=self.training, name="reduce_conv_3", activation='relu')
        # 44 -> 42
        x = Titan.bn_conv(x, 256, kernel_size=(1, 1), padding="valid", training=self.training, name="conv_6",activation='relu')
        x = Titan.bn_conv(x, 512, kernel_size=(3, 3),padding="valid", training=self.training, name="conv_7", activation='relu')
        # 42 -> 21
        x = Titan.bn_conv(x, 1024, kernel_size=(3, 3),strides=(2, 2),padding='same', training=self.training, name="reduce_conv_4", activation='relu')
        # 21 -> 19
        x = Titan.bn_conv(x, 512, kernel_size=(1, 1), padding="valid",training=self.training, name="conv_8", activation='relu')
        x = Titan.bn_conv(x, 1024, kernel_size=(3, 3), padding="valid", training=self.training, name="conv_9",activation='relu')
        if self.finetunning == True:
            x = self.TI.avg_pooling(x, padding='valid', pool_size=(19, 19), stride=1)
            channel = int(x.shape[3])
            x = tf.reshape(x, [-1, channel * 1 * 1])
            x = Titan.dropout(x, ratio=0.8, training=self.training, name="dropout")
            ouput_c = self.TI.fc(x, self.nb_class_c, name="fully_connected", activation="None")
            return ouput_c

        #
        #x = Titan.bn_conv(x, 1024, training=self.training, name="net_5",activation='relu')
        #x = Titan.bn_conv(x, 1024, training=self.training, name="net_6",activation='relu')
        x = Titan.dropout(x, ratio=0.5, training=self.training, name="dropout")
        #output_d = Titan.conv(x, len(self.anchors)*(5 + self.nb_class_d), kernel_size=(1, 1),  activation="None", name="detect_box",use_bias=False)
        output_d = Titan.bn_conv(x, len(self.anchors) * (5 + self.nb_class_d), kernel_size=(1, 1), activation="None", name="detect_box", use_bias=False)
        Titan.print_total_parameter()
        return output_d

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
        self.confi = net[:, :, :, :1]
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
        self.prediction = tf.identity(tf.expand_dims(self.confi_sig, axis=3) * self.softmax_predict)
