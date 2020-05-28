import numpy as np
import sys
import torch

#input_image = np.random.normal(size=(1,3,224,224))
#np.save("input_image.npy",input_image.squeeze().astype(np.float32))
input_image = np.load("mbnetv2/test_image.npy")
input_image = torch.Tensor(np.expand_dims(input_image,0))

initial_conv_filters = torch.Tensor(np.load("mbnetv2/initial_conv.npy"))
initial_conv_bias = torch.Tensor(np.load("mbnetv2/initial_conv_bias.npy"))
intermediate = torch.nn.functional.conv2d(input_image,initial_conv_filters,bias=initial_conv_bias,stride=2,padding=1,dilation=1,groups=1)
intermediate = torch.nn.functional.relu(intermediate)
global layer
layer = 0

def depthwise_block(input_image, layer_num,stride,residual=False):

    expansion_filters = torch.Tensor(np.load("mbnetv2/expand_1x1_" + str(layer_num) + ".npy")).unsqueeze(2).unsqueeze(3)
    expansion_bias = torch.Tensor(np.load("mbnetv2/expand_1x1_" + str(layer_num) + "_bias.npy"))
    depthwise_filters = torch.Tensor(np.load("mbnetv2/depthwise_nxn_" + str(layer_num) + ".npy")).unsqueeze(1)
    depthwise_bias = torch.Tensor(np.load("mbnetv2/depthwise_nxn_" + str(layer_num) + "_bias.npy"))
    groupwise_filters = torch.Tensor(np.load("mbnetv2/contraction_1x1_" + str(layer_num) + ".npy")).unsqueeze(2).unsqueeze(3)
    groupwise_bias = torch.Tensor(np.load("mbnetv2/contraction_1x1_" + str(layer_num) + "_bias.npy"))

    intermediate = torch.nn.functional.conv2d(input_image,expansion_filters,bias=expansion_bias,stride=1,padding=0,dilation=1,groups=1)
    intermediate = torch.nn.functional.relu(intermediate)
    print(intermediate.size())
    print(depthwise_filters.size())

    intermediate = torch.nn.functional.conv2d(intermediate, depthwise_filters, bias=depthwise_bias, stride=stride, padding=1, dilation=1, groups=depthwise_filters.size(0))
    intermediate = torch.nn.functional.relu(intermediate)

    intermediate = torch.nn.functional.conv2d(intermediate,groupwise_filters,bias=groupwise_bias,stride=1,padding=0,dilation=1,groups=1)
    global layer
    layer += 3
    if residual:
        np.save("ref_layer"+str(layer)+".npy",intermediate + input_image)
        return intermediate + input_image
    else:
        np.save("ref_layer"+str(layer)+".npy",intermediate )
        return intermediate

intermediate = depthwise_block(intermediate,0,1,True)

intermediate = depthwise_block(intermediate,1,2,False)
intermediate = depthwise_block(intermediate,2,1,True)
intermediate = depthwise_block(intermediate,3,2,False)
intermediate = depthwise_block(intermediate,4,1,True)
intermediate = depthwise_block(intermediate,5,1,True)
intermediate = depthwise_block(intermediate,6,1,True)
intermediate = depthwise_block(intermediate,7,2,False)
intermediate = depthwise_block(intermediate,8,1,True)
intermediate = depthwise_block(intermediate,9,1,True)
intermediate = depthwise_block(intermediate,10,1,True)
intermediate = depthwise_block(intermediate,11,1,True)
intermediate = depthwise_block(intermediate,12,1,True)
intermediate = depthwise_block(intermediate,13,1,False)
intermediate = depthwise_block(intermediate,14,1,True)
intermediate = depthwise_block(intermediate,15,1,True)
intermediate = depthwise_block(intermediate,16,1,True)
intermediate = depthwise_block(intermediate,17,1,True)
intermediate = depthwise_block(intermediate,18,1,True)
intermediate = depthwise_block(intermediate,19,2,False)
intermediate = depthwise_block(intermediate,20,1,True)
intermediate = depthwise_block(intermediate,21,1,True)
intermediate = depthwise_block(intermediate,22,1,True)
intermediate = depthwise_block(intermediate,23,1,True)
intermediate = depthwise_block(intermediate,24,1,True)
intermediate = depthwise_block(intermediate,25,1,False)


groupwise_filters = torch.Tensor(np.load("mbnetv2/final_1x1_conv.npy")).unsqueeze(2).unsqueeze(3)
groupwise_bias = torch.Tensor(np.load("mbnetv2/final_1x1_conv_bias.npy"))
intermediate = torch.nn.functional.conv2d(intermediate,groupwise_filters,bias=groupwise_bias,stride=1,padding=0,dilation=1,groups=1)
intermediate = torch.nn.functional.relu(intermediate)

intermediate = torch.nn.functional.avg_pool2d(intermediate,7)
np.save("intermediate.npy",intermediate.squeeze().data.numpy())

final_dense_weight = torch.Tensor(np.load("mbnetv2/final_dense_kernel.npy"))
final_dense_bias = torch.Tensor(np.load("mbnetv2/final_dense_bias.npy"))

intermediate = torch.matmul(final_dense_weight,intermediate.squeeze()) + final_dense_bias



intermediate = torch.nn.functional.softmax(intermediate)






#np.save("ref.npy",result_1.squeeze().data.numpy())
