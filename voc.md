# about PASCAL VOC dataset

## 该项目主要参考霹雳吧啦的博文
* https://blog.csdn.net/qq_37541097/article/details/115787033
* 数据集下载地址： http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

## 数据集说明
* 20个目标类别 + 1个背景类
* train.txt、val.txt和trainval.txt文件里记录的是对应标注文件的索引
* 标注信息以XML文件保存


## 文件结构：
```
VOCdevkit
    └── VOC2012
         ├── Annotations               所有的图像标注信息(XML文件)
         ├── ImageSets    
         │   ├── Action                人的行为动作图像信息
         │   ├── Layout                人的各个部位图像信息
         │   │
         │   ├── Main                  目标检测分类图像信息
         │   │     ├── train.txt       训练集(5717)
         │   │     ├── val.txt         验证集(5823)
         │   │     └── trainval.txt    训练集+验证集(11540)
         │   │
         │   └── Segmentation          目标分割图像信息
         │         ├── train.txt       训练集(1464)
         │         ├── val.txt         验证集(1449)
         │         └── trainval.txt    训练集+验证集(2913)
         │ 
         ├── JPEGImages                所有图像文件
         ├── SegmentationClass         语义分割png图（基于类别）
         └── SegmentationObject        实例分割png图（基于目标）
```