import numpy as np
import sys
import torch

#input_image = np.random.normal(size=(1,3,224,224))
#np.save("input_image.npy",input_image.squeeze().astype(np.float32))
input_image = np.load("mbnetv1/test_image.npy")
input_image = torch.Tensor(input_image).unsqueeze(0)

initial_conv_filters = torch.Tensor(np.load("mbnetv1/initial_conv.npy"))
initial_conv_bias = torch.Tensor(np.load("mbnetv1/initial_conv_bias.npy"))
intermediate = torch.nn.functional.conv2d(input_image,initial_conv_filters,bias=initial_conv_bias,stride=2,padding=1,dilation=1,groups=1)
intermediate = torch.nn.functional.relu(intermediate)

np.save("first_layer.npy",intermediate)

def depthwise_block(input_image, layer_num,stride):


    depthwise_filters = torch.Tensor(np.load("mbnetv1/depthwise_nxn_" + str(layer_num) + ".npy"))
    depthwise_bias = torch.Tensor(np.load("mbnetv1/depthwise_nxn_" + str(layer_num) + "_bias.npy"))
    groupwise_filters = torch.Tensor(np.load("mbnetv1/contraction_1x1_" + str(layer_num) + ".npy")).unsqueeze(2).unsqueeze(3)
    groupwise_bias = torch.Tensor(np.load("mbnetv1/contraction_1x1_" + str(layer_num) + "_bias.npy"))
    intermediate = torch.nn.functional.conv2d(input_image, depthwise_filters, bias=depthwise_bias, stride=stride, padding=1, dilation=1, groups=depthwise_filters.size(0))
    intermediate = torch.nn.functional.relu(intermediate)

    result_1 = torch.nn.functional.conv2d(intermediate,groupwise_filters,bias=groupwise_bias,stride=1,padding=0,dilation=1,groups=1)
    result_1 = torch.nn.functional.relu(result_1)
    return result_1

intermediate = depthwise_block(intermediate,0,1)
intermediate = depthwise_block(intermediate,1,2)
intermediate = depthwise_block(intermediate,2,1)
intermediate = depthwise_block(intermediate,3,2)
intermediate = depthwise_block(intermediate,4,1)
intermediate = depthwise_block(intermediate,5,2)
intermediate = depthwise_block(intermediate,6,1)
intermediate = depthwise_block(intermediate,7,1)
intermediate = depthwise_block(intermediate,8,1)
intermediate = depthwise_block(intermediate,9,1)
intermediate = depthwise_block(intermediate,10,1)
intermediate = depthwise_block(intermediate,11,2)
intermediate = depthwise_block(intermediate,12,1)
np.save("test.npy",intermediate.squeeze().data.numpy())

intermediate = torch.nn.functional.avg_pool2d(intermediate,7)


final_dense_weight = torch.Tensor(np.load("mbnetv1/final_dense.npy"))
final_dense_bias = torch.Tensor(np.load("mbnetv1/final_dense_bias.npy"))

intermediate = torch.matmul(final_dense_weight,intermediate.squeeze()) + final_dense_bias


intermediate = torch.nn.functional.softmax(intermediate)

np.save("intermediate.npy",intermediate.squeeze().data.numpy())

#np.save("ref.npy",result_1.squeeze().data.numpy())
