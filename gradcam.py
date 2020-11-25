import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targeted intermediate layers """
    
    """ 该类用于从目标中间层提取激活层和注册梯度(看到后边的 `x.register_hook` 就懂啥意思了) """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        # 想了解 hook 的用法, 请取消下一行注释，单步调试查看输出的位置
        # print("爷被调用了")
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers: # 在指定的层输出
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x # <-- 注意这里是列表, 元素数量和 len(self.target_layers) 一样


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    
    """该类用于前向传播, 并用于获取:
    1. 这一整套网络的输出
    2. 来着中间目标层的输出(激活)
    3. 目标中间层的梯度"""

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                # 此处用于整热图, 也进行了前向传播, 被藏在了 `self.feature_extractor` 里
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1) # <---- 此处经过 pooling 后, shape == [1, 2048, 1, 1], 送入 fc 之前要 “拉直” 它
            else:
                x = module(x)
        
        return target_activations, x



def preprocess_image(img):
    '''
    @Brife:
        原始图片预处理的函数, 标准化 + HWC -> BCHW
    '''
    
    # 以下 means 和 stds 是从 ImageNet 中整出来用于做标准化的均值和标准差
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1] # cv2 读经来是 BGR 的数据, 所以要反一下
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    
    # `np.ascontiguousarray` 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    # 可以自己查一查这个函数，蛮有用的
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    # `torch.Tensor.unsqueeze_` 方法用于给指定位置升维，原地操作 
    preprocessed_img.unsqueeze_(0) # 该代码的意思实际是, 将数据变为 (Batch_size, C, H, W) 的形式
    # 默认创建的张量 `.requires_grad == False` 
    # 更改该张量之后的运算是否被记录
    # Change if autograd should record operations on this tensor. 这句话更加准确
    input_tensor = preprocessed_img.requires_grad_(True)
    return input_tensor



def show_cam_on_image(img, mask):
    '''
    @Brife:
        在图片上加上 heatmap, 顺便保存
    '''

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) # BGR 形式 shape : (224, 224, 3)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img) 
    cam = cam / np.max(cam)    # 所以上一步不用担心是 cv2.add 
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    ''' 本代(yan)码(chu)的主角
    '''
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_tensor):
        return self.model(input_tensor)

    def __call__(self, input_tensor, index=None):
        
        # features 是某个中间层的输出, output 是整个神经网络的输出
        if self.cuda:
            features, output = self.extractor(input_tensor.cuda())
        else:
            features, output = self.extractor(input_tensor)

        if index == None: # <--------------- ？？
            # 调用 `torch.Tensor.data` 可以得到 `Tensor` 的数据 + `requires_grad=False` 的版本
            # 注意与 `.detach()` 用法的区别, 同样是内存共享
            index = np.argmax(output.cpu().data.numpy())
        
        # 先用 NumPy 创建初始热码, shape 和 output.shape 一样, 再用 torch 转换一下
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output) # shape 为 torch.Size([])
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad() 
        # 在本代码中 `self.feature_module` 实际上是 `self.model.layer4` 的引用
        # (self.model.layer4 == self.feature_module) 为 True
        # (self.model.layer4 is self.feature_module) 为 True
        # 特就是不执行 `self.feature_module.zero_grad()` 也无妨, 仅在是引用的时候可以不执行
        one_hot.backward(retain_graph=True) # 执行这一步时, 会调用之前的 hook

        # 通过这一步得到激活层的梯度值, 这个梯度值和那个
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]                     # <----  torch.Size([1, 2048, 7, 7])
        target = target.cpu().data.numpy()[0, :]  # <----  (2048, 7, 7)

        weights = np.mean(grads_val, axis=(2, 3)) # <----  (1, 2048)
        weights = weights[0, :]                   # <----  (2048, )
        cam = np.zeros(target.shape[1:], dtype=np.float32)  # <---- (7, 7)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0) # 此处可理解为进过 relu 
        cam = cv2.resize(cam, input_tensor.shape[2:]) # <---- torch.Size([224, 224])
        # 顺手进行归一化
        cam = cam - np.min(cam) 
        cam = cam / np.max(cam) 
        return cam



class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input_tensor):
        positive_mask = (input_tensor > 0).type_as(input_tensor)
        output = torch.addcmul(torch.zeros(input_tensor.size()).type_as(input_tensor), input_tensor, positive_mask)
        self.save_for_backward(input_tensor, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_tensor, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_tensor > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_tensor.size()).type_as(input_tensor),
                                    torch.addcmul(torch.zeros(input_tensor.size()).type_as(input_tensor), grad_output,
                                                  positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            # 该函数用于递归地将所有的 GuidedBackpropReLU 替换为 ReLU
            for idx, module in module_top._modules.items():
                print(idx)
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply # <---- 此处并非 torch.nn.Module 的子类
                
        recursive_relu_apply(self.model)

    def forward(self, input_tensor):
        return self.model(input_tensor)

    def __call__(self, input_tensor, index=None):
        if self.cuda:
            output = self.forward(input_tensor.cuda())
        else:
            output = self.forward(input_tensor)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input_tensor.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    
    # 参数解析器, 偷个懒这里就不注释了, 如果不太熟悉的话, 就去查一查 `argparse` 的用法
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    return args



def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ 运行方式 ： python grad_cam.py <path_to_image>
    1. 使用 opencv 加载图片
    2. 将其通过 ResNet50 并将其转换为 Pytorch 变量
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, 
                        target_layer_names=["2"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255 # NumPy 会使用默认的 float64, torch 为 float32, 故而这里使用 float32
    input_tensor = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input_tensor, target_index) # shape : (224, 224)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    
    # 以下 print 只是为了确认:
    # (relu): ReLU(inplace=True) 变为 
    # (relu): <built-in method apply of FunctionMeta object at 0xXXXXXXXXXX>
    # print(model._modules.items())
    
    gb = gb_model(input_tensor, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)


