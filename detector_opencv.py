import cv2
from model.yolov2 import Model
from setting.config_manager import ConfigDecoder


def main(input_stream, configure_file, record = False,record_path = ''):
    # Load configure file & Model
    config = ConfigDecoder(configure_file)
    Network = Model(config, name= "yolov2")
    vidcap = cv2.VideoCapture(input_stream)

    if record == True:
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(record_path, fourcc,  30.0, (1280, 642))
        while True:
            success, img = vidcap.read()
            if success == False:
                break
            boxs = Network.predict_V2(img)
            for box in boxs:
                        x1, y1, x2, y2, score, claz, claz_str, color = box
                        cv2.putText(img,str(claz_str) + ": " + "{0:.2f}".format(score),(x1,y1+15),
                                    cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.5,color= color)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            out.write(img)
            cv2.imshow('Window ', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        vidcap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        while True:
            success, img = vidcap.read()
            if success == False:
                break
            boxs = Network.predict_V2(img)
            for box in boxs:
                x1, y1, x2, y2, score, claz, claz_str, color = box
                cv2.putText(img, str(claz_str) + ": " + "{0:.2f}".format(score), (x1, y1 + 15),
                            cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=color)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.imshow('Window', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        vidcap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    path = "C:\\Users\Titan\Desktop\\video\추격씬\\test.mp4"
    main(0, "./setting/darknet_19_7.json")
