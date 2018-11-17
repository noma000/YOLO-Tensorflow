# YOLO-Tensorflow

### Download model & configure file 

https://drive.google.com/drive/folders/1wZ-ZBqkeBU9P6At6FSJsOXE9zW7AK6fI?usp=sharing


### Library version

+ tensorflow-gpu 1.10.0
+ tensorflow 1.2.0
+ CUDA 9.0.176
+ CUDA-DNN 
+ Opencv
+ imgaug (ref : https://github.com/aleju/imgaug)


![Alt text](./Readme_Image/figure_01.JPG)

### Quick start

1. download this project
2. download model weight, configure file, label from [here](https://drive.google.com/drive/folders/1wZ-ZBqkeBU9P6At6FSJsOXE9zW7AK6fI?usp=sharing)

### Configure file

```python
    {
        "archtecture": { # Network Archtector check "./model/network"
            "network": {
                "anchor": [[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]],
                "name": "darknet_full",
                "input_shape" : 416,
                "output_shape" : 13,
                "finetunning" : false
            },
            "net_constant": {
                "coordinate": 1.0,
                "obj": 5.0,
                "noobj": 1.0,
                "probability": 1.0
            },
            "nms_setting":{
                "max_output_size" : 10,                         # number of maximum draw boxs 
                "iou_threshold" : 0.5,                          # iou_threshold
                "score_threshold" : 0.3                         # remove box under score
            }
    
        },
        "option":{                                              # training option
            "tensorboard_use": false,                           # not implemented
            "max_to_keep" : 10,                                 # number of maximum model when you save it
            "batch_size":1,                                     # batch_size wheb you run dectector opencv set batch_size = 1 
            "use_pretrainmodel" : true,                         # when you use pretraining model, set true
            "use_finemodel" : false                             # use_finetunning model [not implemented]
        },
        "training_setting": {                                   # training detection model setting [not implemented]
            "total_epoch": 600,
            "number_class": 80,                                 # number of class
            "lr": 0.001,                                        # learning_rate
            "decrease_rate": 0.99,                              # decrease_rate of learning rate per epoch
        },
        "finetunning_setting": {                                # training classification model setting
            "total_epoch": 10,                                  
            "number_class": 1000,
            "lr": 0.0001,
            "decrease_rate": [ 0.9, 0.95, 0.99],
            "decrease_epoch": [ 20, 40, 60],
            "trdata_nb" : 1281167,
            "tedata_nb" : 0
        },
        "path": {
            "load_model": "./saved_network/COCO_max-31",        # load model location
            "save_model" : "./saved_network/pascal_voc_2017",   # saved model location 
            "TB_logpath": "./",
            "Label_path": "./setting/coco_label",               # label file location
            "data_path": "D:/dataset/PASCAL_VOC/",              # image datas location
            "train_path_f": "",
            "test_path_f":"",
            "data_path_f":"",
            "classfication_dir": "/home/titan/data/"            # [not implemented]
        },
        "augmentation": {                                       # [not implemented]
            "horizen_flip": true,
            "augmentation2": true,
            "augmentation4": true,
            "augmentation3": true
        }
    }
```

### Running sample
detector_opencv.py
parm1 : input_stream  / parm2 : configure file
    
    # Use camera with detect model
    main(0, "./setting/configure.json")
    
    # Read video path with dectec model
    main(path,"./setting/configure.json")
    
    # Record output steam
    main(path, "./setting/configure.json",record=True,record_path="./output.mp4",resolution=(1280, 642))




### How to training




### How to draw Precision & Recall curve
