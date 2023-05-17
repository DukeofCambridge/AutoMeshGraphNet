import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_add


class MLP(nn.Module):
    """
    基本组件：多层感知机
    1.encode阶段对所有node和edge的feature通过mlp处理
    2.process阶段每步通过mlp对所有nodes,edges的feature进行更新
    3.注意前两个阶段都是分别对每个node和edge用一个mlp处理，改变的是每个边点feature的大小，最后decode要把所有边点的feature综合起来
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 layer_norm=True,
                 activation='relu'):

        super(MLP, self).__init__()
        act_fun = nn.ReLU()
        if activation == 'elu':
            act_fun = nn.ELU()
        # input layer
        layers = [nn.Linear(input_dim, hidden_dim), act_fun]
        # hidden layers
        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fun]
        # output layer
        layers += [nn.Linear(hidden_dim, output_dim)]
        # norm layer
        if layer_norm:
            layers += [nn.LayerNorm(output_dim)]

        self.net = nn.Sequential(*layers)

    # torch.nn.Module 的__call__(self)函数中会返回 forward()函数 的结果，因此PyTorch中的forward()函数可以直接通过类名被调用，而不用实例化对象
    def forward(self, input):
        output = self.net(input)

        return output


class GraphNetBlock(nn.Module):
    """
    用来执行process步骤的组件（更新所有nodes,edges的feature），我们知道process每次更新是通过mlp完成的，构造器传入的参数规定了这个mlp是怎样的
    """
    def __init__(self, hidden_dim, num_layers, acti, agg_fun):
        super(GraphNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.agg_fun = agg_fun
        self.mlp_node = MLP(input_dim=2 * hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activation=acti)  # 3*hidden_dim: [nodes, accumulated_edges]
        self.mlp_edge = MLP(input_dim=3 * hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activation=acti)  # 3*hidden_dim: [sender, edge, receiver]

    def update_edges(self, senders, receivers, node_features, edge_features):
        """
        unsqueeze(i)在第i个维度插入一维，相当于让原数据升一维，比如[4,1].unsqueeze(0)后大小变为[1,4,1]
        expand()和repeat()函数用于对已有维度进行大小扩展，方法是复制原本的数据，expand()只能对大小为1的维度进行扩展，而repeat()可以让任意维度的大小乘一个数
        """
        senders = senders.unsqueeze(2).expand(-1, -1, self.hidden_dim)  # [1,106973]->[1,106973,1]->[1,106973,64]
        receivers = receivers.unsqueeze(2).expand(-1, -1, self.hidden_dim)  # [1,106973]->[1,106973,1]->[1,106973,64]
        sender_features = torch.gather(node_features, dim=1, index=senders)  # gather是scatter的逆操作
        receiver_features = torch.gather(node_features, dim=1, index=receivers)
        features = torch.cat([sender_features, receiver_features, edge_features], dim=-1)

        return self.mlp_edge(features)

    def update_nodes(self, receivers, node_features, edge_features):
        # 将每个点与所有邻接节点的边上的特征求和，与节点特征在最后一个维度上拼接
        accumulate_edges = scatter_add(edge_features, receivers, dim=1)
        if self.agg_fun == 'mlp':
            features = torch.cat([node_features, accumulate_edges], dim=-1)  # size: [1,14844,64+64]
            # print('features.shape: ', features.shape)
            return self.mlp_node(features)
        else:
            features = node_features + accumulate_edges
            return features

    def forward(self, senders, receivers, node_features, edge_features):
        new_edge_features = self.update_edges(senders, receivers, node_features, edge_features)
        new_node_features = self.update_nodes(receivers, node_features, new_edge_features)
        # 自己原本的特征加上从邻居元素收集的特征
        new_node_features += node_features
        new_edge_features += edge_features

        return new_node_features, new_edge_features


class Encoder(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, num_layers, acti):
        super(Encoder, self).__init__()
        self.node_mlp = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activation=acti)
        self.edge_mlp = MLP(input_dim=input_dim_edge, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activation=acti)

    def forward(self, node_features, edge_features):
        node_latents = self.node_mlp(node_features)
        edge_latents = self.edge_mlp(edge_features)

        return node_latents, edge_latents


class Process(nn.Module):
    # 每次更新调用GraphNetBlock.forward(),执行 message_passing_steps 次更新
    def __init__(self, hidden_dim, num_layers, message_passing_steps, acti, agg_fun):
        super(Process, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(message_passing_steps):
            self.blocks.append(GraphNetBlock(hidden_dim, num_layers, acti, agg_fun))

    def forward(self, senders, receivers, node_features, edge_features):
        for graphnetblock in self.blocks:
            node_features, edge_features = graphnetblock(senders, receivers, node_features, edge_features)

        return node_features, edge_features


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, acti):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       layer_norm=False, activation=acti)

    def forward(self, node_features):
        final_node_features = self.mlp(node_features)
        final_node_features = final_node_features.squeeze(2).squeeze(0)
        # final_node_features是每个node的输出结果，这里直接求均值作为整个模型的风阻预测值
        return torch.mean(final_node_features).flatten().float()


class EncodeProcessDecode(nn.Module):
    """
    核心组件，封装了完整的EPD步骤
    """
    def __init__(self,
                 input_dim_node,
                 input_dim_edge,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 message_passing_steps, args):
        super(EncodeProcessDecode, self).__init__()

        self.encoder = Encoder(input_dim_node, input_dim_edge, hidden_dim, num_layers, args['activation'])
        self.process = Process(hidden_dim, num_layers, message_passing_steps, args['activation'], args['agg_fun'])
        self.decoder = Decoder(hidden_dim, output_dim, num_layers, args['activation'])

    def forward(self, senders, receivers, node_features, edge_features):
        # before-encoding
        # torch.Size([1, 15287, 3])
        # torch.Size([1, 106973, 4])
        node_features, edge_features = self.encoder(node_features, edge_features)
        # after-encoding
        # torch.Size([1, 15287, 64])
        # torch.Size([1, 106973, 64])
        node_features, edge_features = self.process(senders, receivers, node_features, edge_features)
        # after-process
        # torch.Size([1, 15287, 64])
        # torch.Size([1, 106973, 64])
        predict = self.decoder(node_features)
        # print("prediction-size")
        # print(predict.shape)
        return predict


class Normalizer(nn.Module):
    def __init__(self, size, std_epsilon=1e-8):
        super(Normalizer, self).__init__()

        self.register_buffer('std_epsilon', torch.tensor(std_epsilon))
        self.register_buffer('count', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('sum', torch.zeros(size, dtype=torch.float32))
        self.register_buffer('sum_squared', torch.zeros(size, dtype=torch.float32))

    def set_accumulated(self, accumulator):
        self.count = accumulator.count
        self.sum = accumulator.sum
        self.sum_squared = accumulator.sum_squared

    def forward(self, data):
        return (data - self.mean()) / self.std()

    def inverse(self, normalized_data):
        return normalized_data * self.std() + self.mean()

    def mean(self):
        return self.sum / self.count

    def std(self):
        std = torch.sqrt(self.sum_squared / self.count - self.mean() ** 2)
        return torch.maximum(std, self.std_epsilon)
