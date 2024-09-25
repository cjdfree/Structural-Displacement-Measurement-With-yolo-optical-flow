# README

## 内容介绍

基于YOLO，光流的土木工程结构位移监测项目。

为在校研究生的科研项目代码，学习过程中的笔记等。

## 文件目录

- datasets：目标检测数据集
  - `manual_label`：使用`labelme`标注的数据集，还没有转换成yolo训练所需要的数据集格式
  - `five_floors_frameworks_calibration`：经过转换后的数据集的文件目录，只标注了**标定板Calibration**的数据集（**重要，yolo训练的数据集就是这个**）
    - `dataset`：（yolo训练的数据集文件（已完成数据标注格式转换，数据集划分），**实际上训练只需要这个文件夹里的数据**，其他的文件是在转换yolo训练所需要的数据集格式中产生的文件）
      - `images`：图片
      - `labels`：标注
    - `images`：图片格式的文件
    - `json`：`labelme`标注后生成的json文件，放在这里生成txt文件
    - `txt`：yolo训练需要txt的标注格式，转换来自json文件
  - `five_floors_frameworks_calibration & floor`：做了标定板**Calibration**和楼板**Floor**标注的数据集，但是感觉暂时用不到
- ultralytics：[yolo的官方项目](https://github.com/ultralytics/ultralytics)
  - 除了文件夹`mytest`，其他都来自ultralytics的项目
  - `mytest`：程序，数据，结果都在里面，就是跑和yolo有关系的
    - `annotated_images`：存放yolo目标检测可视化结果的图片，分为不同分辨率的图片（每个文件夹再下分不同场景`normal`，`light`，`rain`，再下分不同的yolo模型的结果，包括yolov8-v10的前三个大小的模型）
      - 360_640：360*640的图片，图像序列200张
      - 360_640_all：360*640的图片，图像序列1000张
      - 540_960：540*960的图片，图像序列200张
      - 540_960_all：540*960的图片，图像序列1000张
    - `calibration_roi`：存放检测出来的**矩形框的坐标文件**，格式是`json`，存放方法和`annotated_images`一样（这两个文件是在程序`yolo_test_save_roi.py`程序运行时产生的文件）
    - `runs`：yolo**模型训练**时候的结果文件，用来查看训练结果，各种指标
    - `data_augmentation.py`：数据增强的代码
    - `json_image_resize.py`：用来调整不一样分辨率图片时，使用`labelme`标注的标签格式的自动化代码（不一定用到）
    - `json2txt_detect.py`：将`labelme`标注的格式转换成`ultralytics`支持的yolo训练的格式，适用于**目标检测任务**
    - `json2txt_segment.py`：将`labelme`标注的格式转换成`ultralytics`支持的yolo训练的格式，适用于**分割任务**
    - `resize.py`：将原来实验室拍摄的视频截取成一帧帧图片放好用来
    - `segment-test.py`：分割任务的测试代码（不一定用到）
    - `splitDataset.py`：划分数据集的代码
    - `train.py`：训练yolo模型的代码，写成了循环测试的自动化脚本
    - `.pt`：是yolo预训练模型的格式，下载好放在这里了
- RAFT：[RAFT的官方项目](https://github.com/princeton-vl/RAFT)
  - 除了文件夹`my_test`，其他都来自RAFT的项目
  - `my_test`：程序，数据，结果都在里面，包括光流的结果，和**yolo计算的位移结果**都在里面
    - `calibration_roi`：从`ultralytics`文件夹里面拷贝过来的目标检测框文件，内容一样
    - `original_dataset`：用RAFT和YOLO来跑光流，目标检测测试的时候用的图片，分别有三个（每个文件夹再下分不同场景`normal`，`light`，`rain`）
      - five_floors_frameworks_360_640：360*640的图片，图像序列200张
      - five_floors_frameworks_360_640_all：360*640的图片，图像序列1000张
      - five_floors_frameworks_540_960：540*960的图片，图像序列200张
      - five_floors_frameworks_540_960_all：540*960的图片，图像序列1000张
    - `raft-things`：跑的光流结果，数据（和之前的存放方式类似）
      - 不太一样的地方：多了一个`original_pretrained_test\five_floors_frameworks_540_960\normal\Updated\roi_image`，存放的是使用yolo检测框和原来
    - `yolo_displacement`：跑的yolo的位移结果，下分不用分辨率，场景，模型的结果
    - `coordinate.py`：读取图片输出点击位置的坐标点（不重要，和之前的一样）
    - `function_decorator.py`：函数装饰器（不重要，和之前的一样）
    - `my_datasets.py`：光流数据集的加载文件（和之前的一样）
    - `optical_flow_test.py`：光流跑图的程序，只跑数据，不出结果
    - `result_visualizaton_updated.py`：光流 + yolo自动选取ROI，帧更新出结果的程序
    - `yolo_displacement.py`：使用yolo直接计算位移的程序

## 试验步骤

yolo部分：按下面顺序进行

- `labelme`打标签
- `labelme`标注的数据转换成yolo训练的数据
  - json转换成txt：`json2txt_detect.py`
  - 数据集划分：`splitDataset.py`
  - 数据增强：`data_augmentation.py`
- yolo训练：`train.py`
- yolo预测图片序列，保存`annotated_images`和`calibration_roi`：`yolo_test_save_roi.py`

光流部分：这部分和之前基本一样，只不过为了方便写代码把跑数据和出结果费分成了两个代码文件，主要是分成跑数据和结果的文件

- `optical_flow_test.py`：把光流图跑出来
- `result_visualization_updated.py`：yolo自动选取ROI + 光流
- `yolo_displacement.py`：yolo直接计算位移，需要前面的`yolo_test_save_roi.py`保存好的坐标文件

## labelme打标签

### 数据来源

选用视频：

- normal：大概从14s左右开始振
- light：大概从9s左右开始振
- rain：大概从14s左右开始振
- fog：大概从15s左右开始振

选取的图片为：总共60张（光流直接跑出来的效果不一定很好，因为那个有光，会影响他的识别结果的，但是应该也会小比较多）

- normal：从14-20s，一共600张，每20帧取一张，一共30张
- light：从9-15s，一共600张，每20帧取一张，一共30张
- rain：从14-20s，一共600张，每20帧取一张，一共30张
- fog：从15-21s，一共600张，每20帧取一张，一共30张（光流那就15-25s，一共1000张）

分辨率：**360*640**

### labelme标签转换成yolo

[YOLOv8制作自己的实例分割数据集保姆级教程（包含json转txt）](https://blog.csdn.net/m0_57010556/article/details/139150198?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-139150198-blog-136361107.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-139150198-blog-136361107.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=4)，可以正常使用



## 数据增强

参考资料：

- [YOLOV8源码常见疑问三-数据增强在yolov8中的应用](https://www.bilibili.com/video/BV1aQ4y1g7ah/?spm_id_from=333.337.search-card.all.click&vd_source=8a137563261f849e155a1b19757d1449)
- [深度学习小技巧-目标检测当中的数据增强（Bubbliiiing 深度学习 教程）](https://www.bilibili.com/video/BV1ZA411b7L8/?spm_id_from=333.337.search-card.all.click&vd_source=8a137563261f849e155a1b19757d1449)
- [albumentations](https://github.com/albumentations-team/albumentations)

数据增强的方式

- 在线数据增强：一般训练一般增强，好处是数据理论上无限
- 离线数据增强：先增强好，再放进去模型中，**可以控制数据增强的方向**

一般数据增强是只在训练集中出现

离线数据增强会影响数据的分布，在线数据增强随机性比较高。

数据增强有很多方式，比如加噪声不一定适合某些场景。



现在的想法：

- 用**离线数据增强**现在一定程度上扩大数据量？
- 再使用**动态数据增强**提升训练的效果？

### 离线数据增强

尝试使用：[albumentations](https://github.com/albumentations-team/albumentations)，参考里面的`installation`和`objection detection`两个章节

### 在线数据增强-yolo源码实现

## 训练YOLO

### 预训练权重

yolov8-yolov10

#### 训练参数

|    参数    | 默认值 |                             含义                             |
| :--------: | :----: | :----------------------------------------------------------: |
|    data    |  None  | 数据集配置文件的路径（例如 `coco8.yaml`)，包含数据集的参数，包括路径、类名和类数，**需要自己修改** |
|   model    |  None  | 指定用于训练的模型文件。接受指向 `.pt` 预训练模型或 `.yaml` 配置文件。对于定义模型结构或初始化权重至关重要。**一般都会自己加载预训练模型** |
|   epochs   |  100   |                           训练轮数                           |
|   imgsz    |  640   | 用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。上面那张图里面的图像分辨率都是640 |
|   device   |  None  | 训练所用设备，选项有单个GPU (`device=0`）、多个 GPU (`device=0,1`）、CPU (`device=cpu`) 或MPS for Apple silicon (`device=mps`) |
| pretrained |  True  |                     即默认使用预训练模型                     |
|   `seed`   |  `0`   | 为训练设置随机种子，确保在相同配置下运行的结果具有可重复性。 |
|  workers   |   8    | 数据加载时的工作线程数。在数据加载过程中，可以使用多个线程并行地加载数据，以提高数据读取速度。这个参数确定了加载数据时使用的线程数，具体的最佳值取决于硬件和数据集的大小。 |

训练代码暂时按照下面的设置

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="five_floors_frameworks.yaml", epochs=100, imgsz=640, device=0)
```

