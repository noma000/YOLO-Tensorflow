from model.yolov2 import Model


#from util.predict_box import NMS
#from util.Draw_box import makebox
import cv2
from setting.config_manager import ConfigDecoder
# from util.predict_box import NMS
# from util.Draw_box import makebox
import cv2

from model.yolov2 import Model
from setting.config_manager import ConfigDecoder

#sess = tf.Session()
config = ConfigDecoder("./setting/darknet_19_7.json")
Network = Model(config, name= "yolov2")
#init = tf.global_variables_initializer()
#sess.run(init)
#Network.load_model()

#with sess.as_default():
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)

#cv2.namedWindow("image")
#cv2.resizeWindow("image", 416, 416)
#path = "C:\\Users\Titan\Desktop\\video\마블\\test.mp4"
path = "C:\\Users\Titan\Desktop\\video\추격씬\\test.mp4"
#path = "D:\Samples\\data_01.mp4"
vidcap = cv2.VideoCapture(path)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.mp4',fourcc,  30.0, (1280, 642))
while True:
    success, img = vidcap.read()
    if success == False:
        break
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r,g,b])
    #img = img[0:1080, int(1920/2-1080/2):int(1920/2+1080/2)]#img.crop((1, 1, 98, 33))
    boxs = Network.predict_V2(img)
    for box in boxs:
            try:
                x1,y1,x2,y2,score,claz,claz_str,color = box
                cv2.putText(img,str(claz_str) + ": " + "{0:.2f}".format(score),(x1,y1+15), cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.5,color= color)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                #cv2.addWeighted(img, 0.1, img, 1 - 0.1,0, img)
            except :
                print("no box")
    out.write(img)

    cv2.imshow('image ', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vidcap.release()
out.release()
cv2.destroyAllWindows()

