import numpy as np
import tensorflow as tf

'''
"network": {
        "anchor": [[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]],
        [1.2145,2.0770, 3.6275,6.1062, 9.1788,9.6729]
        "name": "darknet_tiny",
        "input_shape" : 224,
        "output_shape" : 14,
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
        for i in range(1,15):
            self.net_add(D,"net_"+str(i))
        return D

    # Ref : https://github.com/ruiminshen/yolo-tf/blob/master/model/yolo2/function.py
    # just passthrought
    def reorg(self,net, stride=2, name='reorg'):
        batch_size, height, width, channels = net.get_shape().as_list()
        _height, _width, _channel = height // stride, width // stride, channels * stride * stride
        with tf.name_scope(name) as name:
            net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
            net = tf.transpose(net, [0, 1, 3, 2, 4, 5])  # batch_size, _height, _width, stride, stride, channels
            net = tf.reshape(net, [batch_size, _height, _width, -1], name=name)
        return net


    def net_create(self):
        # image 416x416x3 (RGB), Input Layer
        Titan = self.TI

        #224 -> 112
        x = Titan.bn_conv(self.Input_Image, 16, training=self.training, name="net_01")
        x = Titan.max_pooling(x)
        #112 -> 56
        x = Titan.bn_conv(x, 32, training=self.training, name="net_02")
        x = Titan.max_pooling(x)
        #56 -> 28
        x = Titan.bn_conv(x, 16, training=self.training,  name="net_03")
        x = Titan.bn_conv(x, 128, kernel_size=(1,1), training=self.training, name="net_04")
        x = Titan.bn_conv(x, 16, training=self.training, name="net_05")
        x = Titan.bn_conv(x, 128, training=self.training, name="net_06")
        x = Titan.max_pooling(x)
        # 28 -> 14
        x = Titan.bn_conv(x, 32, kernel_size=(1,1), training=self.training, name="net_07")
        x = Titan.bn_conv(x, 256, training=self.training, name="net_08")
        x = Titan.bn_conv(x, 32, kernel_size=(1,1), training=self.training, name="net_09")
        x = Titan.bn_conv(x, 256, training=self.training, name="net_10")
        x = Titan.max_pooling(x)
        # 26 -> 13
        x = Titan.bn_conv(x, 64, training=self.training, name="net_11")
        x = Titan.bn_conv(x, 512, kernel_size=(1,1), training=self.training, name="net_12")
        x = Titan.bn_conv(x, 64, training=self.training, name="net_13")
        x = Titan.bn_conv(x, 512, kernel_size=(1,1), training=self.training, name="net_14")
        x = Titan.bn_conv(x, 128, training=self.training, name="net_15")

        if self.finetunning == True:
            x = self.TI.avg_pooling(x, padding='valid', pool_size=(14, 14), stride=1)
            channel = int(x.shape[3])
            x = tf.reshape(x, [-1, channel * 1 * 1])
            ouput_c = self.TI.fc(x, self.nb_class_c, name="net_16", activation="None")
            return ouput_c

        x = Titan.bn_conv(x, 1024, training=self.training, name="net_16")
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_17")
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_18")

        output_d = Titan.conv(x, len(self.anchors)*(5 + self.nb_class_d), kernel_size=(1, 1),  activation="None", name="net_23")
        return output_d

