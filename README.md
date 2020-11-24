中文 | [English](https://hub.fastgit.org/DrRyanHuang/pytorch-grad-cam/blob/master/README_EN.md)

## 使用 Pytorch 实现 Grad-CAM ##

Fork 自 [pytorch-grad-cam](https://hub.fastgit.org/jacobgil/pytorch-grad-cam)
README 和 代码注释均做了简单修改和翻译


### 是什么使神经网络认为图像是 'pug, pug-dog' 和 'tabby, tabby cat' :
![Dog](https://hub.fastgit.org/DrRyanHuang/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

### 将 Grad-CAM 与 Guided Backpropagation 相结合的 'pug, pug-dog' 图片:
![Combined](https://hub.fastgit.org/DrRyanHuang/pytorch-grad-cam/blob/master/examples/cam_gb_dog.jpg?raw=true)

Gradient class activation maps 是一种深度学习网络的可视化技术。

原论文: https://arxiv.org/pdf/1610.02391v1.pdf

原论文作者的 torch 实现: https://github.com/ramprs/grad-cam

原 github 作者的 Keras 实现: https://github.com/jacobgil/keras-grad-cam


----------

该程序使用 `torchvision` 的 `VGG19`. 在首次运行该程序的时候，将会自动下载预训练模型.

可以修改该该程序以使用其他模型.
但是，`torchvision` 中 `VGG` 模型的网络卷积部分和全连接部分有 `features/classifier` 方法.
该代码假定传入的 `model` 具有这两个方法. 若你的模型没有这两方法，那你就自己实现一下hhhh.


----------


用法: 
```
python gradcam.py --image-path <path_to_image>
```

使用 CUDA 的话请使用参数 `--use-cuda`:
```
python gradcam.py --image-path <path_to_image> --use-cuda
```