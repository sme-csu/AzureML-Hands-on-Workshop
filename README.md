# 机器学习 Workshop 动手实验

此代码库是机器学习 Workshop内容。 在本地运行机器学习实验，并将其与云配合使用，以扩展到云的环境。

# 数据说明

本Workshop的案例选取的是[CIFAR-10数据](https://www.cs.toronto.edu/~kriz/cifar.html)，这是个开放数据集，包含60000张32x32的彩色图片，分为10类，每类6000张。该数据集通常用来测试图像分类的算法。

![](https://pytorch.org/tutorials/_images/cifar10.png)

# 模型说明

本案例模型使用的是卷积神经网络( Convolutional Neural Network)。

## 代码目录

| 资源         | 链接                           |
|-----------------|----------------------------------|
| 事前准备        | - [conda 环境创建](examples/conda-setup.ipynb)<br/>- [Azure ML 配置](examples/azureml-config.ipynb) |
| 本地执行        | - [执行训练脚本](examples/local-pytorch-run.ipynb) |
| 远程执行        | - [数据上传](examples/1-data_upload.ipynb)<br/>- [Environment创建](examples/2-environment.ipynb)<br/>- [Azure ML训练](examples/3-aml-run.ipynb)<br/>- [超参数调优](examples/4-hyperdrive.ipynb)<br/>- [机器学习管道](examples/5-aml-pipeline.ipynb)|



## 使用的技术
- Azure Machine Learning

## 参考资料
- [Azure Machine Learning 学习文档](https://docs.microsoft.com/zh-cn/azure/machine-learning/)
- [Azure Machine Learning 样例文件 (官方)](https://github.com/Azure/MachineLearningNotebooks)
- [Azure Machine Learning 样例文件 (社区)](https://github.com/Azure/azureml-examples)
- [Azure Machine Learning 模板](https://github.com/Azure/azureml-template)