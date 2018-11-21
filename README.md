# YOLO-Tensorflow

## Download model & configure file 

https://drive.google.com/drive/folders/1wZ-ZBqkeBU9P6At6FSJsOXE9zW7AK6fI?usp=sharing


## Library version

+ tensorflow-gpu 1.10.0
+ tensorflow 1.2.0
+ CUDA 9.0.176
+ CUDA-DNN 
+ Opencv
+ imgaug (ref : https://github.com/aleju/imgaug)


![Alt text](./Readme_Image/figure_01.JPG)

## Quick start

### 1. Download this project
### 2. Download model weight, configure file, label from [here](https://drive.google.com/drive/folders/1wZ-ZBqkeBU9P6At6FSJsOXE9zW7AK6fI?usp=sharing)
### 3. There are 3 files in google drive. 

##### CNN model files
    
    - [pascal_voc_2017]-416_5.data
    - [pascal_voc_2017]-416_5.index
    - [pascal_voc_2017]-416_5.meta

##### Network configure file
    - pascal_voc_config.json

##### Label:color match file
    - pascal_voc_label

### 4. Open configure file and setting your path

```python
        "path": {
            "load_model": "./saved_network/[pascal_voc_2017]-416_5",        # writhe your loaded model path
            "save_model" : "./saved_network/[pascal_voc_2017]-416_5",      # this will use when you training
            "TB_logpath": "./",
            "Label_path": "./setting/pascal_voc_label",                     # label file location
            "data_path": "./Data/",                          # this will use when you training
            
            "train_path_f": "",                                             # for fine tunning [not implemented]
            "test_path_f":"",                                               # for fine tunning [not implemented]
            "data_path_f":"",                                               # for fine tunning [not implemented]
            "classfication_dir": "/home/titan/data/"                        # for fine tunning [not implemented]
        },
```

### 5. Open 'detector_opencv.py' and just run ( you need camera )
```python
    if __name__ == '__main__':
        # Test camera use your cv
        main(0, "./saved_network/[pascal_voc_2017]-416_5")
        # Record detected vedeo
        main(path, "./saved_network/[pascal_voc_2017]-416_5",record=True,record_path="./output.mp4",resolution=(1280, 642))
```

## How to training
You want training new model on your dataset follow behind


### 1. Check your path of dataset, model_name, Label_path

In configure file you must check load_model, save_model, Label_path, data_path ignore other things

```python
        "path": {
            "load_model": "./saved_network/pascal_voc_2017",        # load model location
            "save_model" : "./saved_network/pascal_voc_2017",   # saved model location 
            "TB_logpath": "./",
            "Label_path": "./setting/coco_label",               # label file location
            "data_path": "D:/dataset/PASCAL_VOC/",              # image datas location
            "train_path_f": "",
            "test_path_f":"",
            "data_path_f":"",
            "classfication_dir": "/home/titan/data/"            # [not implemented]
        },
```
save_model & load_model use when you save&load model, you just set the path

you set data_path in this directory you have 3type of files

- Image(jpg) file, Label(.txt)file, train.txt file, test.txt file

In data directory
```txt
-rw-r--r-- 1 root root  108389 Mar  4  2018 2012_004329.jpg  
-rw-r--r-- 1 root root      53 Mar  4  2018 2012_004329.txt  
-rw-r--r-- 1 root root  131185 Mar  4  2018 2012_004330.jpg  
-rw-r--r-- 1 root root      40 Mar  4  2018 2012_004330.txt  
-rw-r--r-- 1 root root  148030 Mar  4  2018 2012_004331.jpg  
-rw-r--r-- 1 root root      39 Mar  4  2018 2012_004331.txt  
-rw-r--r-- 1 root root   39614 Mar  4  2018 test.txt  
-rw-r--r-- 1 root root  262711 Mar  4  2018 train.txt  
```
Each value means class(like persone or car), center x, center y, width, height in the 2012_044331.jpg file.
```txt
root@user-P10S-WS:/disk1/titans_data/pascal_voc/PASCAL_VOC# cat 2012_004331.txt  
14 0.31 0.34 0.212 0.5466666666666666
```
In train.txt file you write your train_datas name except extension
```txt
root@user-P10S-WS:/disk1/titans_data/pascal_voc/PASCAL_VOC# tail train.txt   
2012_004310  
2012_004312  
2012_004315  
2012_004317  
2012_004319  
2012_004326  
2012_004328  
2012_004329  
2012_004330  
```

Last Label format like this.(this is the pascal voc example, 

Interpreting the first line, class number is 0, class name is aeroplane, R=240, G=255, B=130

```txt
aeroplane:240,255,130
bicycle:150,130,252
bird:132,251,173
boat:221,152,231
bottle:142,240,238
bus:36,0,240
car:248,48,92
cat:244,255,40
chair:194,162,100
cow:191,62,234
diningtable:247,191,108
dog:120,90,1
horse:234,111,19
motorbike:240,0,0
person:283,128,22
pottedplant:26,251,0
sheep:255,239,253
sofa:225,223,179
train:208,0,167
tvmonitor:61,85,148
```
### 2. Select archtecture you want and use get_anchors.py to get an anchors

you can find model in model/Network/[net name] and change the configure file

```python
"network": {
    "anchor": [[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]],
    "name": "darknet_full",
    "input_shape" : 416,
    "output_shape" : 13,
    "finetunning" : false
     },
```

get anchor using under command line

    # python3 get_anchors.py -c [configurefile path] -a [# of anchors]

then you can get anchors like this.

[[1.0068,1.6871], [2.5020,4.4176], [4.3034,8.7792], [7.8379,5.2096], [10.0773,10.7282]]

### 3. Set your training parameter

- Tensorboard_use is not implemented.  
- max_to_keep means number of models to keep  
- batch_size means batch_size, when i train model, i fixed to 15 to prevent memory exceed (In 1080 ti 11G) if you take memory exceed error, reduce this value
- If you don't have any pre-training model set false "/option/use_pretrainmodel"
- total_epoch means number of training
- lr : learning rate
- decrease_rate : decrease_Rate per epoch

```python
        "option":{                                              # training option
            "tensorboard_use": false,                           # not implemented
            "max_to_keep" : 10,                                 # number of maximum model when you save it
            "batch_size":15,                                     # batch_size wheb you run dectector opencv set batch_size = 1 
            "use_pretrainmodel" : true,                         # when you use pretraining model, set true
            "use_finemodel" : false                             # use_finetunning model [not implemented]
        },
        "training_setting": {                                   # training detection model setting [not implemented]
            "total_epoch": 600,
            "number_class": 80,                                 # number of class
            "lr": 0.001,                                        # learning_rate
            "decrease_rate": 0.99,                              # decrease_rate of learning rate per epoch
        },
```

### 4. Everything is all set. Let's do model training.

In the train.py you set your configure file

```python
    if __name__ == '__main__':
        train_network(config_file = "./setting/window_configure.json")
```

and run

```txt
root@user-P10S-WS:~/yolo_tensorflow_4# nohup python3 -u train.py > log &
```

then you can get samples image per 5 epoch, and this image will save in /temp/ directory, check train.py 

```python
        if epoch % 5 == 0 :
            if not os.path.isdir("./temp/" + str(epoch) + "/"):
                os.mkdir("./temp/" + str(epoch) + "/")
```

this is the log of training.

```txt
Training start
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:01:00.0
totalMemory: 10.91GiB freeMemory: 10.68GiB

Net_id  Layer           K/S             Input_shape             Output_shape    Activation              Name
--------------------------------------------------------------------------------------------------------------
[No1]   [Conv_bn]       3x3/1   608x608x3       →       608x608x32              leaky_relu              net_1
[No2]   [Max_pool]      3x3/2   608x608x32      →       304x304x32
[No3]   [Conv_bn]       3x3/1   304x304x32      →       304x304x64              leaky_relu              net_2
[No4]   [Max_pool]      3x3/2   304x304x64      →       152x152x64
[No5]   [Conv_bn]       3x3/1   152x152x64      →       152x152x128             leaky_relu              net_3
[No6]   [Conv_bn]       1x1/1   152x152x128     →       152x152x64              leaky_relu              net_4
[No7]   [Conv_bn]       3x3/1   152x152x64      →       152x152x128             leaky_relu              net_5
[No8]   [Max_pool]      3x3/2   152x152x128     →       76x76x128
[No9]   [Conv_bn]       3x3/1   76x76x128       →       76x76x256               leaky_relu              net_6
[No10]  [Conv_bn]       1x1/1   76x76x256       →       76x76x128               leaky_relu              net_7
[No11]  [Conv_bn]       3x3/1   76x76x128       →       76x76x256               leaky_relu              net_8
[No12]  [Max_pool]      3x3/2   76x76x256       →       38x38x256
[No13]  [Conv_bn]       3x3/1   38x38x256       →       38x38x512               leaky_relu              net_9
[No14]  [Conv_bn]       1x1/1   38x38x512       →       38x38x256               leaky_relu              net_10
[No15]  [Conv_bn]       3x3/1   38x38x256       →       38x38x512               leaky_relu              net_11
[No16]  [Conv_bn]       1x1/1   38x38x512       →       38x38x256               leaky_relu              net_12
[No17]  [Conv_bn]       3x3/1   38x38x256       →       38x38x512               leaky_relu              net_13
[No18]  [Max_pool]      3x3/2   38x38x512       →       19x19x512
[No19]  [Conv_bn]       3x3/1   19x19x512       →       19x19x1024              leaky_relu              net_14
[No20]  [Conv_bn]       1x1/1   19x19x1024      →       19x19x512               leaky_relu              net_15
[No21]  [Conv_bn]       3x3/1   19x19x512       →       19x19x1024              leaky_relu              net_16
[No22]  [Conv_bn]       1x1/1   19x19x1024      →       19x19x512               leaky_relu              net_17
[No23]  [Conv_bn]       3x3/1   19x19x512       →       19x19x1024              leaky_relu              net_18
[No24]  [Conv_bn]       3x3/1   19x19x1024      →       19x19x1024              leaky_relu              net_19
[No25]  [Conv_bn]       3x3/1   19x19x1024      →       19x19x1024              leaky_relu              net_20
[No26]  [Conv_bn]       1x1/1   38x38x512       →       38x38x64                leaky_relu              net_21
[No27]  [Conv_bn]       3x3/1   19x19x1280      →       19x19x1024              leaky_relu              net_22
[dropout]0.5
[No28]  [Conv]          1x1/1   19x19x1024      →       19x19x91                None            net_23
total parameter : 78.779K

model load :./saved_network/coco_mix
Loading_Traindata
load_path : /disk1/titans_data/coco/coco_mix/COCO_mix/train.txt
Number of data : 151617
Loading_Testdata
load_path : /disk1/titans_data/coco/coco_mix/COCO_mix/test.txt
Number of data : 16984
Loading data is done...
EPOCH : [ 0 / 600 ]
learing_rate :  0.00010038514464001442

--- 10478.280810832977 training time(min) ---
Train Result /Recall: 0.10776187399828986/Cost: 7.22031259506861
Train Result /coordinary: 0.09575036033986982/iou: 6.66163969096524/prob: 0.4629225392461194
--- 391.1250743865967 test time(min) ---
Test Result /Recall: 0.2309268484741133/Cost: 2.2899185419784707
Test Result /coordinary: 0.0670237297014508/iou: 1.935700510921412/prob: 0.28719430075227803
nb_box : 0
nb_box : 4
nb_box : 0
nb_box : 0
nb_box : 3
nb_box : 52
nb_box : 0
nb_box : 2
nb_box : 0
nb_box : 13
saved : ./temp/0/image_0True.jpg
saved : ./temp/0/image_1True.jpg
saved : ./temp/0/image_2True.jpg
saved : ./temp/0/image_3True.jpg
saved : ./temp/0/image_4True.jpg
saved : ./temp/0/image_5True.jpg
saved : ./temp/0/image_6True.jpg
saved : ./temp/0/image_7True.jpg
saved : ./temp/0/image_8True.jpg
saved : ./temp/0/image_9True.jpg
nb_box : 0
nb_box : 0
nb_box : 3
nb_box : 0
nb_box : 0
nb_box : 0
nb_box : 0
nb_box : 0
nb_box : 0
nb_box : 0
saved : ./temp/0/image_0False.jpg
saved : ./temp/0/image_1False.jpg
saved : ./temp/0/image_2False.jpg
saved : ./temp/0/image_3False.jpg
saved : ./temp/0/image_4False.jpg
saved : ./temp/0/image_0False.jpg
saved : ./temp/0/image_1False.jpg
saved : ./temp/0/image_2False.jpg
saved : ./temp/0/image_3False.jpg
saved : ./temp/0/image_4False.jpg
saved : ./temp/0/image_5False.jpg
saved : ./temp/0/image_6False.jpg
saved : ./temp/0/image_7False.jpg
saved : ./temp/0/image_8False.jpg
saved : ./temp/0/image_9False.jpg
[-] Data shuffle
[-] Scaling value =7.438161993705258
Save model :./saved_network/coco_mix
```

### How to draw Precision & Recall curve

Check map_recall_calc.py 

Read configure setting
```python
    config = ConfigDecoder("./setting/window_configure.json")
    nb_claz = config.get_training("number_class")
    last_grid = config.get_model("output_shape")
    anchor_len = len(config.get_model("anchor"))
    Network_name = config.get_model("name")
```

Predict file is made by model, if you already excute this line then skip the command
```python
make_predict_file(config, file_path='./predict_file.pickle')
```

Check the number of thresholds
```python
    # If you want more dense precision & recall graph increase number of threshold
    # confidence threshold for NMS
    T = [float(i)/10 for i in range(0,11)]
    # IOU threshold for NMS
    IT =[float(i)/10 for i in range(0,11)]
```

Then you can get below graph (It can take a lot of time.)







And you can get an Excel file that represents the confusion matrix.


## Configure file

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

## Running sample
detector_opencv.py
parm1 : input_stream  / parm2 : configure file
    
    # Use camera with detect model
    main(0, "./setting/configure.json")
    
    # Read video path with dectec model
    main(path,"./setting/configure.json")
    
    # Record output steam
    main(path, "./setting/configure.json",record=True,record_path="./output.mp4",resolution=(1280, 642))












