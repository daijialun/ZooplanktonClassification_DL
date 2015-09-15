# ZooplanktoNet Experiment Record

ZooplanktoNet相关实验记录

## Train

### 3通道图像

#### LeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

- 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_Test_1300*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 56.625%
           loss(val): 2.57342
           loss(train): 0.0854

- 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 7.6923%
           loss(val): 2.91502
           loss(train): 2.24022
           
- 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 7.6923%
           loss(val): 2.91502
           loss(train): 2.24022
                      
- 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2363 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 58.9583%
           loss(val): 2.49416
           loss(train): 0.21369
           
           
##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### AlexNet(CaffeNet)

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）+ 图像转换为3通道lmdb 

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### GoogleNet

##### 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

### 单通道图像

#### LeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### AlexNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）+ 图像转换为3通道lmdb 

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### CaffeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）+ 图像转换为3通道lmdb 

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### GoogleNet

##### 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

## Finetune

### 3通道pre-trained模型

#### CaffeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### GoogleNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

### 单通道pre-trained模型