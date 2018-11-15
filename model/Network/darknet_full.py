import numpy as np
import tensorflow as tf
from model.Network.Net import Net
'''
    "network": {
        "anchor": [[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]],
        "name": "darknet_full",
        "input_shape" : 416,
        "output_shape" : 13,
        "finetunning" : false
    },
'''

class Network(Net):

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
        for i in range(1,18):
            self.net_add(D,"net_"+str(i))
        return D

    # Passthrought
    # Ref : https://github.com/ruiminshen/yolo-tf/blob/master/model/yolo2/function.py
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

        #416 -> 208
        x = Titan.bn_conv(self.Input_Image, 32, training=self.training, name="net_1")
        x = Titan.max_pooling(x)
        #208 -> 104
        x = Titan.bn_conv(x, 64, training=self.training, name="net_2")
        x = Titan.max_pooling(x)
        #104 -> 52
        x = Titan.bn_conv(x, 128, training=self.training,  name="net_3")
        x = Titan.bn_conv(x, 64, kernel_size=(1,1), training=self.training, name="net_4")
        x = Titan.bn_conv(x, 128, training=self.training, name="net_5")
        x = Titan.max_pooling(x)
        # 52 -> 26
        x = Titan.bn_conv(x, 256, training=self.training, name="net_6")
        x = Titan.bn_conv(x, 128, kernel_size=(1,1), training=self.training, name="net_7")
        x = Titan.bn_conv(x, 256, training=self.training, name="net_8")
        x = Titan.max_pooling(x)
        # 26 -> 13
        x = Titan.bn_conv(x, 512, training=self.training, name="net_9")
        x = Titan.bn_conv(x, 256, kernel_size=(1,1), training=self.training, name="net_10")
        x = Titan.bn_conv(x, 512, training=self.training, name="net_11")
        x = Titan.bn_conv(x, 256, kernel_size=(1,1), training=self.training, name="net_12")
        x = Titan.bn_conv(x, 512, training=self.training, name="net_13")

        bypass = x
        x = Titan.max_pooling(x)

        # 13 -> 13
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_14")
        x = Titan.bn_conv(x, 512, kernel_size=(1, 1), training=self.training, name="net_15")
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_16")
        x = Titan.bn_conv(x, 512, kernel_size=(1, 1), training=self.training, name="net_17")

        if self.finetunning == True:
            x = self.TI.avg_pooling(x, padding='valid', pool_size=(13, 13), stride=1)
            channel = int(x.shape[3])
            x = tf.reshape(x, [-1, channel * 1 * 1])
            x = Titan.dropout(x, ratio=0.5, training=self.training, name="dropout")
            ouput_c = self.TI.fc(x, self.nb_class_c, name="net_18", activation="None")
            return ouput_c

        x = Titan.bn_conv(x, 1024, training=self.training, name="net_18")
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_19")
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_20")

        bypass = Titan.bn_conv(bypass, 64, kernel_size=(1, 1),training=self.training, name="net_21")
        bypass = self.reorg(tf.identity(bypass, name="bypass"))
        x = tf.concat([x, bypass],-1)
        # reduce channel to 256
        x = Titan.bn_conv(x, 1024, training=self.training, name="net_22")

        x = Titan.dropout(x, ratio=0.5, training=self.training, name="dropout")
        output_d = Titan.conv(x, len(self.anchors)*(5 + self.nb_class_d), kernel_size=(1, 1),  activation="None", name="net_23")
        Titan.print_total_parameter()
        return output_d
