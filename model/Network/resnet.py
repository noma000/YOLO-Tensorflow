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

    def get_fine_tunning_area(self):
        D = {}
        index_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13",
                      "14", "15", "16", "17"]
        for i in index_list:
            name = "net_" + i
            conv_name = "conv_" + name + "/kernel"
            bias_name = "conv_" + name + "/bias"
            bn_name_01 = "bn_" + name + "/beta"
            bn_name_02 = "bn_" + name + "/gamma"
            bn_name_03 = "bn_" + name + "/moving_mean"
            bn_name_04 = "bn_" + name + "/moving_variance"
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, conv_name)[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bias_name)[0]
            bn_01 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_01)[0]
            bn_02 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_02)[0]
            bn_03 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_03)[0]
            bn_04 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bn_name_04)[0]
            D[conv_name] = kernel
            D[bias_name] = bias
            D[bn_name_01] = bn_01
            D[bn_name_02] = bn_02
            D[bn_name_03] = bn_03
            D[bn_name_04] = bn_04
        return D


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

    def net_create(self):
        # image 416x416x3 (RGB), Input Layer
        Titan = self.TI

        #416 -> 208
        x = Titan.bn_conv(self.Input_Image, 32, training=self.training, name="net_01")
        x = Titan.max_pooling(x)
        #208 -> 104
        x = Titan.bn_conv(x, 64, training=self.training, name="net_02")
        x = Titan.max_pooling(x)
        #104 -> 52
        x = Titan.bn_conv(x, 128, training=self.training,  name="net_03")
        x = Titan.bn_conv(x, 64, kernel_size=(1,1), training=self.training, name="net_04")
        x = Titan.bn_conv(x, 128, training=self.training, name="net_05")
        x = Titan.max_pooling(x)
        # 52 -> 26
        x = Titan.bn_conv(x, 256, training=self.training, name="net_06")
        x = Titan.bn_conv(x, 128, kernel_size=(1,1), training=self.training, name="net_07")
        x = Titan.bn_conv(x, 256, training=self.training, name="net_08")
        x = Titan.max_pooling(x)
        # 26 -> 13
        x = Titan.bn_conv(x, 512, training=self.training, name="net_09")
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

        output_d = Titan.conv(x, len(self.anchors)*(5 + self.nb_class_d), kernel_size=(1, 1),  activation="None", name="net_23")
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