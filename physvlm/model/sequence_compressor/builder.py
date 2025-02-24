import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)



# def build_sequence_compressor(config, **kwargs):        
#     if config.sequence_compressor_strid == 2:
#         kernel_size=4
#         stride=2
#         padding=1
#     elif config.sequence_compressor_strid == 4:
#         kernel_size=8
#         stride=4
#         padding=2
#     else:
#         return nn.Sequential()
#     hidden_size = config.hidden_size
#     return nn.Sequential(
#         Permute(0, 2, 1),
#         nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.ReLU(),  # Optional: Add activation function if needed
#         Permute(0, 2, 1)
#     )
    

def build_sequence_compressor(config, **kwargs):   
    return nn.Sequential(
        Permute(0, 2, 1),
        nn.AdaptiveMaxPool1d(243),
        Permute(0, 2, 1)
    )