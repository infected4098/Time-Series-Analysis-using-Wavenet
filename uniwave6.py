import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda')


class DCConv1d(torch.nn.Module):
    """Dilated Causal Convolution 1D"""
    def __init__(self, in_channels, out_channels, dilation):
        super(DCConv1d, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size = 2,
                                     stride = 1, dilation = dilation, padding = 0)

    def forward(self, x):
        #x.shape = [batch_size, 1, input_length]
        output = self.conv1(x)
        return output #[batch_size, out_channels, output_length]
    # output_length = input_length - dilation

class CConv1d(torch.nn.Module):
    """Causal Convolution 1D"""
    def __init__(self, in_channels, out_channels):
        super(CConv1d, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size = 2,
                                     stride = 1, padding = 1)

    def forward(self, x):
        output = self.conv1(x) #[batch_size, out_channels, output_length]
        return output[:, :, :-1] #[batch_size, out_channels, input_length]

class Residualblock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(Residualblock, self).__init__()

        self.dc = DCConv1d(res_channels, res_channels, dilation)
        self.conv = torch.nn.Conv1d(in_channels = res_channels, out_channels = res_channels,
                                    kernel_size = 1, stride = 1, padding = 0) #input_length = output_length
        self.skipconv = torch.nn.Conv1d(in_channels = res_channels, out_channels = skip_channels, kernel_size = 1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x, skip_size):
        #x.shape = [batch_size, 1, input_length]
        #skip_len = # points for inference
        output = self.dc(x) #output.shape = [batch_size, out_channels, output_length = input_length - dilation]
        gates = self.relu(output)
        #gates.shape = [batch_size, out_channels, output_length = input_length - dilation]

        output = self.conv(gates) #[batch_size, out_channels, input_length - dilation]
        input = x[:, :, -(output.size(2)):]
        res = output + input #[batch_size, out_channels, input_length - dilation]

        skip = self.skipconv(gates) #[batch_size, skip_channels, input_length - dilation]

        return res, skip[:, :, -skip_size:] #res.shape = [batch_size, res_channels, input_length - dilation], skip.shape = [batch_size, skip_channels, skip_len]

class ResidualStack(torch.nn.Module):

    def __init__(self, layers, stacks, in_channels, res_channels, skip_channels):
        super(ResidualStack, self).__init__()
        self.layers = layers
        self.stacks = stacks
        self.in_channels = in_channels
        self.res_blocks = nn.ModuleList()
        for dilation in self.build_dilations():
            block = Residualblock(res_channels, skip_channels, dilation)
            self.res_blocks.append(block)
    def build_dilations(self):
        dilations = []
        for s in range(0, self.stacks): #0, 1, 2, 3, 4...
            for l in range(0, self.layers): #0, 1, 2, ...
                dilations.append(2 ** l)
        return dilations


    def forward(self, x, skip_size):
        #x.shape = [batch_size, 1, input_length]
        #skip_len = # points for inference

        output = x
        skip_connections = []
        for res_block in self.res_blocks:
            output, skips = res_block(output, skip_size)
            skip_connections.append(skips)
        skip_connections = torch.stack(skip_connections)
        return skip_connections

class clf_net(torch.nn.Module):
    def __init__(self, skip_channels, channels, skip_size):
        super(clf_net, self).__init__()

        #self.conv1 = torch.nn.Conv1d(in_channels = channels, out_channels = channels,
        #                             kernel_size = 1)
        #self.conv2 = torch.nn.Conv1d(in_channels = channels, out_channels = channels , kernel_size = 1)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = torch.nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 128)
        #self.linear1 = torch.nn.LazyLinear(128)
        self.linear2 = torch.nn.Linear(128, 1)
    def forward(self, x):
        output = self.relu(x) #[batch_size, skip_channels, 1]
        #output = self.flatten(output)
        output = output.transpose(2, 1)
        #output = self.relu(output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        return output


class Wavenet(torch.nn.Module):
    def __init__(self, layers, stacks, in_channels, res_channels, skip_channels):
        super(Wavenet, self).__init__()
        self.causal = CConv1d(in_channels = in_channels, out_channels = res_channels)
        self.res_stack = ResidualStack(layers, stacks, res_channels, res_channels, skip_channels)
        self.receptive_fields = self.calc_receptive_fields(layers, stacks)
        self.clf_net = clf_net(skip_channels, skip_channels, self.calc_receptive_fields)

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields

        #self.check_input_size(x, output_size)

        return output_size
    def forward(self, x):
        #x.shape = [batch_size, in_channels = 1, input_length]
        output = x
        output_size = self.calc_output_size(x)
        output = self.causal(output) #[batch_size, out_channels, input_length]
        #output = self.causal2(output)
        skip_connections = self.res_stack(output, output_size)

        output_ = torch.sum(skip_connections, dim = 0)
        output = self.clf_net(output_)

        return output #[batch_size, 1, 1]


model = Wavenet(7, 2, 1, 64, 64).to(device)
a = torch.rand([16, 1, 302]).to(device)
print(model(a).shape)

total_count = 0
for name, param in model.named_parameters():
    # print(f"Layer: {name} - Size: {param.numel()}")
    total_count += param.numel()
print("#Total Parameters of this model is :", total_count)
early_stopping_count = 0
