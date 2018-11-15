import tensorflow as tf
class Titan:
    def __init__(self, sess, logging=False, TB_PATH="", fine_tunning=False):
        self.sess = sess
        self.TB_logging = logging
        self.TB_path = TB_PATH
        self.fine_tunning = fine_tunning
        self.total_parameter = 0
        self.net_number = 0
        self.vardic = {}
        self.name_list = []
        print("Net_id\tLayer\t\tK/S\t\tInput_shape\t\tOutput_shape\tActivation\t\tName")
        print("----------------------------------------------------------------------------------")

    def save_var(self, name, variable_tensor):
        self.vardic[name] = variable_tensor

    def activate_tensorboard(self):
        if self.TB_logging == True:
            writer = tf.summary.FileWriter(self.TB_path)
            writer.add_graph(self.sess.graph)

    def numbering(self):
        self.net_number = self.net_number + 1
        return "[No"+str(++self.net_number)+']\t'

    def batchnorm(self,input,training = True, name="batchnorm"):
        return  tf.layers.batch_normalization(input, training=training,epsilon=1e-6,name= name)

    def dropout(self,x,ratio=0.8, training = True, name = "dropout"):
        print("[dropout]" + str(ratio))
        return tf.layers.dropout(inputs=x,rate=ratio,training=training,name=name)

    def bn_conv(self,input, filters, kernel_size=(3, 3), strides=(1, 1), padding="Same", name='model',  training = True, activation='leaky_relu',use_bias=False):
        net_nb = self.numbering()
        conv_name = "conv_" + name
        bn_name = 'bn_' + name
        self.total_parameter += kernel_size[0]*kernel_size[1]*filters
        ib, iw, ih, ic = input.shape
        iinfo_str = str(iw)+"x"+str(ih)+"x"+str(ic)
        x = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name= conv_name,use_bias=use_bias)
        ob, ow, oh, oc = x.shape
        oinfo_str = str(ow) + "x" + str(oh) + "x" + str(oc)
        shape_str = iinfo_str + "\t→\t" + oinfo_str
        kernel_str = str(kernel_size[0]) + "x" + str(kernel_size[1]) +"/" + str(strides[0])
        activation_str = activation
        x = tf.layers.batch_normalization(x, training=training,epsilon=1e-6,name= bn_name)
        x = self.get_activation(x, activation=activation)
        self.logging(x, conv_name, self.TB_logging)
        print(net_nb + "[Conv_bn]\t" + kernel_str + "\t"+ shape_str + "\t\t" + activation_str + "\t\t" + name)
        return x

    def conv(self,input, filters, kernel_size=(3, 3), strides=(1, 1), padding="Same",name='model', activation='leaky_relu',use_bias = False):
        net_nb = self.numbering()
        conv_name = "conv_" + name
        self.total_parameter += kernel_size[0] * kernel_size[1] * filters
        ib, iw, ih, ic = input.shape
        iinfo_str = str(iw)+"x"+str(ih)+"x"+str(ic)
        x = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name= "conv_" + name)
        ob, ow, oh, oc = x.shape
        oinfo_str = str(ow) + "x" + str(oh) + "x" + str(oc)
        shape_str = iinfo_str + "\t→\t" + oinfo_str
        kernel_str = str(kernel_size[0]) + "x" + str(kernel_size[1]) +"/" + str(strides[0])
        activation_str = activation
        x = self.get_activation(x, activation=activation)
        self.logging(x, conv_name, self.TB_logging)
        print(net_nb + "[Conv]\t\t" + kernel_str + "\t" + shape_str + "\t\t" + activation_str + "\t\t" + name)
        return x

    def bn_fc(self,input, output_sz, name="model", training=True, activation='relu'):
        net_nb = self.numbering()
        fc_name = "fc_" + name
        bn_name = 'bn_' + name
        _, ic = input.shape
        iinfo_str = str(ic)
        x = tf.layers.dense(inputs=input, units=output_sz, name =fc_name)
        _, oc = x.shape
        oinfo_str = str(oc)
        shape_str = iinfo_str + "\t→\t" + oinfo_str
        activation_str = activation
        x = tf.layers.batch_normalization(x, training=training,name=bn_name)
        x = self.get_activation(x, activation=activation)
        self.logging(x, "fc_", self.TB_logging)
        print(net_nb + "[Fc_bn]\t\t\t" + shape_str + "\t\t" + activation_str + "\t\t" + name)
        return x


    def fc(self,input, output_sz, name="model",activation ='relu'):
        net_nb = self.numbering()
        fc_name = "fc_" + name
        _, ic = input.shape
        iinfo_str = str(ic)
        x = tf.layers.dense(inputs=input, units=output_sz, name= fc_name)
        _, oc = x.shape
        oinfo_str = str(oc)
        shape_str = iinfo_str + "\t→\t" + oinfo_str
        activation_str = activation
        x = self.get_activation(x, activation=activation)
        self.logging(x,"fc_",self.TB_logging)
        print(net_nb + "[Fc_bn]\t\t\t\t" + shape_str + "\t\t" + activation_str + "\t\t" + name)
        return x

    def max_pooling(self,input, pool_size=(3,3),padding="Same",stride=2):
        net_nb = self.numbering()
        ib, iw, ih, ic = input.shape
        iinfo_str = str(iw)+"x"+str(ih)+"x"+str(ic)
        x = tf.layers.max_pooling2d(inputs=input, pool_size=pool_size, padding=padding, strides=stride)
        ob, ow, oh, oc = x.shape
        oinfo_str = str(ow) + "x" + str(oh) + "x" + str(oc)
        shape_str = iinfo_str + "\t→\t" + oinfo_str
        kernel_str = str(pool_size[0]) + "x" + str(pool_size[1]) + "/" + str(stride)
        print(net_nb + "[Max_pool]\t" + kernel_str + "\t" + shape_str)
        return x

    def avg_pooling(self,input, pool_size=(3,3),padding="Same",stride=2):
        net_nb = self.numbering()
        ib, iw, ih, ic = input.shape
        iinfo_str = str(iw)+"x"+str(ih)+"x"+str(ic)
        x = tf.layers.average_pooling2d(inputs=input, pool_size=pool_size, padding=padding, strides=stride)
        ob, ow, oh, oc = x.shape
        oinfo_str = str(ow) + "x" + str(oh) + "x" + str(oc)
        shape_str = iinfo_str + "\t→\t" + oinfo_str
        kernel_str = str(pool_size[0]) + "x" + str(pool_size[1]) + "/" + str(stride)
        print(net_nb + "[Avg_pool]\t" + kernel_str + "\t" + shape_str)
        return x

    def logging(self,input,name,tb_logging):
        if tb_logging == True:
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name + '/kernel')[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name + '/bias')[0]

            tf.summary.histogram('Weights', kernel)
            tf.summary.histogram('Bias', bias)
            tf.summary.histogram('Layers', input)
    def print_total_parameter(self):
        print("total parameter : " + str(self.total_parameter/1000) +"K")
    def get_activation(self,input,activation="None"):
        if activation == "relu":
            return tf.nn.relu(input)
        elif activation == 'sigmoid':
            return  tf.nn.sigmoid(input)
        elif activation == "leaky_relu":
            return leaky_relu(input)
        else:
            return input
    '''
    input : scala name list, value
    '''
    def TB_logging(self, scala_value):
        if self.TB_logging == True:
            for values in scala_value:
                name, value = values
                tf.summary.scalar(name, value)

def leaky_relu(input,alpha=0.1):
    return tf.maximum(input, alpha * input)

