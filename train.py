from model.yolov2 import Model
from setting.config_manager import ConfigDecoder
from util.manage_tfrecord import batch_reader
import math as ma
import os
import time
import cv2

def learing_rate(lr):
    if lr > 1e-5:
        return lr * 0.9 # rapid drop
    if lr > 1e-6:
        return lr * 0.95 # little slow
    else:
        return lr * 0.995 # almost fix

def learing_rate2(total_epoch, cur_epoch):
    equ1 = (total_epoch-cur_epoch)*ma.log(1e-3) + (cur_epoch-1)*ma.log(1e-4)
    equ2 = total_epoch - 1
    return ma.exp(equ1/equ2)


def train_network(config_file):
    print("***  Training start  ***")
    # Network initializer
    config = ConfigDecoder(config_file)
    Network = Model(config, name="yolov2")

    #TB_PATH = config.get_path("TB_logpath") # Not implementation
    training_epoch = config.get_training("total_epoch")
    batch_size = config.get_option("batch_size")
    learning_rate = config.get_training("lr")
    decrease_rate = config.get_training("decrease_rate")
    model_path = config.get_path("load_model")
    save_model = config.get_path("save_model")


    batch_load = batch_reader(config)

    nb_train_samples = batch_load.nb_train
    nb_test_samples = batch_load.nb_test


    for epoch in range(0, training_epoch):
        avg_coor_loss = 0
        avg_iou_loss = 0
        avg_probability_loss = 0

        train_batch = int(nb_train_samples / batch_size)
        test_batch = int(nb_test_samples / batch_size)
        print("EPOCH : [", str(epoch), "/", str(training_epoch), "]")
        learning_rate = learning_rate * decrease_rate
        #learning_rate = learing_rate2(training_epoch, epoch)
        print("learing_rate : ", str(learning_rate))

        ''' Training model '''
        Recall, Cost = [0.0, 0.0]
        start_time = time.time()
        for count_batch in range(train_batch):
            batch_image, batch_label_box = batch_load.read_batch(training=True)
            c, _, r, c1, c2, c3 = Network.train(batch_image, batch_label_box, learning_rate)
            Cost += c/train_batch
            Recall += r/train_batch
            avg_coor_loss += c1/train_batch
            avg_iou_loss += c2/train_batch
            avg_probability_loss += c3/train_batch
            print("Train Result /Recall: " + str(Recall) +"/Cost: " + str(Cost))
            print("Train Result /coordinary: " + str(avg_coor_loss) +
                  "/iou: " + str(avg_iou_loss) +
                  "/prob: " + str(avg_probability_loss))
        print("--- %s training time(min) ---" % (time.time() - start_time))
        print("Train Result /Recall: " + str(Recall) + "/Cost: " + str(Cost))
        print("Train Result /coordinary: " + str(avg_coor_loss) +
              "/iou: " + str(avg_iou_loss) +
              "/prob: " + str(avg_probability_loss))

        ''' Testing model '''
        avg_coor_loss = 0
        avg_iou_loss = 0
        avg_probability_loss = 0

        Recall, Cost = [0.0, 0.0]
        start_time = time.time()
        for count_batch in range(test_batch):
            batch_image, batch_label_box = batch_load.read_batch(training=False)
            c, r, c1, c2, c3 = Network.testing(batch_image, batch_label_box)
            Cost += c / test_batch
            Recall += r / test_batch
            avg_coor_loss += c1 / test_batch
            avg_iou_loss += c2 / test_batch
            avg_probability_loss += c3 / test_batch
        print("--- %s test time(min) ---" % (time.time() - start_time))
        print("Test Result /Recall: " + str(Recall) + "/Cost: " + str(Cost))
        print("Test Result /coordinary: " + str(avg_coor_loss) +
              "/iou: " + str(avg_iou_loss) +
              "/prob: " + str(avg_probability_loss))

        ''' Get samples '''
        if epoch % 5 == 0 :
            if not os.path.isdir("./temp/" + str(epoch) + "/"):
                os.mkdir("./temp/" + str(epoch) + "/")
            for bol in [True,False]:
                if bol == True:
                    batch_image, batch_label_box = batch_load.read_batch(training=True)
                if bol == False:
                    batch_image, batch_label_box = batch_load.read_batch(training=False)
                boxs = Network.predict(batch_image)
                for i, image in enumerate(boxs):
                    for obj in image:
                        x1, y1, x2, y2, score, claz, claz_str, color = obj
                        cv2.putText(batch_image[i], str(claz) + ": " + str(score), (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 1, cv2.LINE_AA)
                        cv2.rectangle(batch_image[i], (x1, y1), (x2, y2), color, 2)
                        cv2.imwrite("./temp/"+str(epoch)+"/image_" + str(i) + str(bol) + ".jpg", batch_image[i])
                        print("saved : ./temp/" + str(epoch) + "/image_" + str(i) + str(bol) + ".jpg")
        batch_load.data_shuffle()
        print("Save model :" + save_model)
        Network.save_model(save_model, epoch)
    Network.close()

if __name__ == '__main__':
    train_network(config_file = "./setting/window_configure.json")

