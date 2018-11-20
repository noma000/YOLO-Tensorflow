import numpy as np
import tensorflow as tf

'''
    "network": {
        "anchor": [[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]],
        "name": "resnet-[nb_layer]",
        "input_shape" : 224,
        "output_shape" : 7,
        "finetunning" : false
    },
'''

class Network:
    def __init__(self,input_image, config, struct, training=True, Nb_Layer=18):
        self.training = training
        self.batch_size = config.get_option("batch_size")
        self.anchors = config.get_model("anchor")
        self.nb_class_d = config.get_training("number_class")
        self.nb_class_c = config.get_finetraining("number_class")
        self.finetunning = config.get_model("finetunning")
        self.Input_Image = input_image
        self.NB_layer = Nb_Layer
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
        for i in range(1,18):
            self.net_add(D,"net_"+str(i))
        return D
    def residual_module(self, input, filter=0, name="residual"):
        Titan = self.TI
        print("Loaded moudle :", name)
        x = Titan.bn_conv(input, filter, kernel_size=(3, 3), padding='Same', name=name + "_a",
                           training=self.training)
        x = Titan.bn_conv(x, filter, kernel_size=(3, 3), padding='Same', name=name + "_b", training=self.training, activation=None)
        output = tf.add(x, input)
        output = tf.nn.relu(output)
        return output

    def grid_reduce(self, input, filter=0, name="reduce"):
        Titan = self.TI
        print("Loaded moudle :", name)
        x = Titan.bn_conv(input, filter, kernel_size=(3, 3), padding='Same', strides=(2, 2), name=name + "_a",
                           training=self.training)
        output = Titan.bn_conv(x, filter, kernel_size=(3, 3), padding='Same', name=name + "_b",
                                training=self.training)
        return output

    def deep_residual_module(self, input, filters, name="dp_residual"):
        Titan = self.TI
        filter1, filter2, filter3 = filters
        x = Titan.bn_conv(input, filter1, kernel_size=(1, 1), padding='Same', name=name + "_a",
                           training=self.training)
        x = Titan.bn_conv(x, filter2, kernel_size=(3, 3), padding='Same', name=name + "_b",
                           training=self.training)
        x = Titan.bn_conv(x, filter3, kernel_size=(1, 1), padding='Same', name=name + "_c",
                           training=self.training, activation=None)

        input = Titan.bn_conv(input, filter3, kernel_size=(1, 1), padding='Same', name=name + "_d",
                           training=self.training, activation=None)

        output = tf.add(x, input)
        output = tf.nn.relu(output)
        return output
    def deep_grid_reduce(self, input, filters, name="dp_reduce"):
        Titan = self.TI
        print("Loaded moudle :", name)
        filter1, filter2, filter3 = filters
        x = Titan.bn_conv(input, filter1, kernel_size=(1, 1), padding='Same', name=name + "_a",
                           training=self.training)
        x = Titan.bn_conv(x, filter2, kernel_size=(3, 3), strides=(2, 2), padding='Same', name=name + "_b",
                           training=self.training)
        output = Titan.bn_conv(x, filter3, kernel_size=(1, 1), padding='Same', name=name + "_c",
                                training=self.training)
        return output

    def net_create(self):
        # image 224x224x3 (RGB), Input Layer
        Titan = self.TI

        # zero padding all direction 3 so input size 230
        padding = tf.constant([[0,0], [3, 3], [3, 3], [0,0]])
        padding = tf.pad(self.Input_Image, padding, "CONSTANT")

        x = Titan.bn_conv(padding, 64, kernel_size=(7, 7), padding='Valid', name="layer-1",
                           strides=(2, 2), training=self.training)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], padding="VALID", strides=2)

        if self.NB_layer == 18:
            x = self.residual_module(x, 64, name="residual_net_01")
            x = self.residual_module(x, 64, name="residual_net_02")
            x = self.grid_reduce(x, 128, name="residual_net_03")
            x = self.residual_module(x, 128, name="residual_net_04")
            x = self.grid_reduce(x, 256, name="residual_net_05")
            x = self.residual_module(x, 256, "residual_net_06")
            x = self.grid_reduce(x, 512, name="residual_net_07")
            x = self.residual_module(x, 512, "residual_net_08")
            # x = tf.layers.average_pooling2d(inputs=x, pool_size=[7, 7], padding="VALID", strides=1)
            # x = tf.reshape(x, [-1, 512 * 1 * 1])

        elif self.NB_layer == 34:
            x = self.residual_module(x, 64, "residual_net_1")
            x = self.residual_module(x, 64, "residual_net_2")
            x = self.residual_module(x, 64, "residual_net_3")
            x = self.grid_reduce(x, 128, name="residual_net_4")
            x = self.residual_module(x, 128, "residual_net_5")
            x = self.residual_module(x, 128, "residual_net_6")
            x = self.residual_module(x, 128, "residual_net_7")
            x = self.grid_reduce(x, 256, name="residual_net_8")
            x = self.residual_module(x, 256, "residual_net_9")
            x = self.residual_module(x, 256, "residual_net_10")
            x = self.residual_module(x, 256, "residual_net_11")
            x = self.residual_module(x, 256, "residual_net_12")
            x = self.residual_module(x, 256, "residual_net_13")
            x = self.grid_reduce(x, 512, name="residual_net_14")
            x = self.residual_module(x, 512, "residual_net_15")
            x = self.residual_module(x, 512, "residual_net_16")
            #x = tf.layers.average_pooling2d(inputs=x, pool_size=[7, 7], padding="VALID", strides=1)
            #x = tf.reshape(x, [-1, 512 * 1 * 1])

        elif self.NB_layer == 50:
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_1")
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_2")
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_3")
            x = self.deep_grid_reduce(x, [128, 128, 512], name="residual_net_4")
            x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_5")
            x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_6")
            x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_7")
            x = self.deep_grid_reduce(x, [256, 256, 1024], name="residual_net_8")
            x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_9")
            x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_10")
            x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_11")
            x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_12")
            x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_13")
            x = self.deep_grid_reduce(x, [512, 512, 2048], name="residual_net_14")
            x = self.deep_residual_module(x, [512, 512, 2048], name="residual_net_15")
            x = self.deep_residual_module(x, [512, 512, 2048], name="residual_net_16")
            #x = tf.layers.average_pooling2d(inputs=x, pool_size=[7, 7], padding="VALID", strides=1)
            #x = tf.reshape(x, [-1, 2048 * 1 * 1])

        elif self.NB_layer == 101:
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_1")
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_2")
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_3")
            x = self.deep_grid_reduce(x, [128, 128, 512], name="residual_net_4")
            x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_5")
            x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_6")
            x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_7")
            x = self.deep_grid_reduce(x, [256, 256, 1024], name="residual_net_8")
            for i in range(9, 32):
                x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_" + str(i))
            x = self.deep_grid_reduce(x, [512, 512, 2048], name="residual_net_14")
            x = self.deep_residual_module(x, [512, 512, 2048], name="residual_net_15")
            x = self.deep_residual_module(x, [512, 512, 2048], name="residual_net_16")
            #x = tf.layers.average_pooling2d(inputs=x, pool_size=[7, 7], padding="VALID", strides=1)
            #x = tf.reshape(x, [-1, 2048 * 1 * 1])

        elif self.NB_layer == 152:
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_1")
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_2")
            x = self.deep_residual_module(x, [64, 64, 256], name="residual_net_3")
            x = self.deep_grid_reduce(x, [128, 128, 512], name="residual_net_4")
            for i in range(5, 12):
                x = self.deep_residual_module(x, [128, 128, 512], name="residual_net_" + str(i))
            x = self.deep_grid_reduce(x, [256, 256, 1024], name="residual_net_12")
            for i in range(13, 48):
                x = self.deep_residual_module(x, [256, 256, 1024], name="residual_net_" + str(i))
            x = self.deep_grid_reduce(x, [512, 512, 2048], name="residual_net_48")
            x = self.deep_residual_module(x, [512, 512, 2048], name="residual_net_49")
            x = self.deep_residual_module(x, [512, 512, 2048], name="residual_net_50")
            #x = tf.layers.average_pooling2d(inputs=x, pool_size=[7, 7], padding="VALID", strides=1)
            #x = tf.reshape(x, [-1, 2048 * 1 * 1])

        if self.finetunning == True:
            x = self.TI.avg_pooling(x, padding='valid', pool_size=(7, 7), stride=1)
            channel = int(x.shape[3])
            x = tf.reshape(x, [-1, channel * 1 * 1])
            ouput_c = self.TI.fc(x, self.nb_class_c, name="net_fc", activation="None")
            return ouput_c

        output_d = Titan.conv(x, len(self.anchors)*(5 + self.nb_class_d), kernel_size=(1, 1),  activation="None", name="net_last")
        return output_d