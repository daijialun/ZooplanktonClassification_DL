# Cifar10 相关介绍

## 使用流程

所有指令在CAFFE_ROOT目录执行

1. 下载数据集

        ./data/cifar10/get_cifar10.sh
        
2. 转换数据格式

        ./examples/cifar10/create_cifar10.sh
        
3. cifar10网络模型在examples/cifar10目录下的`cifar10_quick_train_test.prototxt`

4. 训练网络

        ./examples/cifar10/train_quick.sh
        
5. 在`cifar*solver.prototxt`文件中，保证`solver_mode: GPU`，即使用GPU训练


## 文件说明

- cifar10_quick_solver.prototxt 网络训练的基础参数设置

- cifar10_quick_train_test.prototxt 网络具体结构