# ZooplanktoNet Experiment Record

ZooplanktoNet相关实验记录

## Train

### 3通道图像

#### LeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

- 1) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_Test_1300*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张。

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

    accuracy(val)，loss(val)与loss(train)曲线波动变化基本正常
    
- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张。

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
           
    accuracy(val)与loss(val)基本呈直线，基本没有变化；loss(train)在范围内震荡变化
           
- 3) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张。

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
           
    accuracy(val)，loss(val)与loss(train)都图形呈直线状，基本不改变
           
- 4) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张。

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
           
    accuracy(val)，loss(val)与loss(train)曲线波动正常
    
 - 5) 根据以上各实验结果，可以观察到：1) 与 4) 的共同点为train都为7098images，val都为25%，而test不同，二者的accuracy与loss都几乎相同；2) 与 3)的共同点为train都为9460images，val为测试的1300images，二者的accuracy与loss都几乎相同
           
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


- 1) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25_Test_1300*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 61.125%
           loss(val): 2.67403
           loss(train): 0.0199318

     accuracy(val)，loss(val)与loss(train)曲线波动正常
    
- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCAL
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 48.4615%
           loss(val): 3.97581
           loss(train): 0.02518
      
      accuracy(val)，loss(val)与loss(train)曲线波动正常  
         
- 3) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_1300*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val为test的1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 47.9231%
           loss(val): 4.56747
           loss(train): 0.02716
           
           accuracy(val)，loss(val)与loss(train)曲线波动正常
                      
- 4) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2363 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 58.9583%
           loss(val): 2.80499
           loss(train): 0.1193
           
  accuracy(val)，loss(val)与loss(train)曲线波动正常

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