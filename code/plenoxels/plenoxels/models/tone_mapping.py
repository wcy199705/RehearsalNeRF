import tinycudann as tcnn
import torch
import torch.nn as nn

class Main_tonemapping_RGB(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.r_mlp = tcnn.Network(
            n_input_dims=1,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 1,
            }
        )
        self.g_mlp = tcnn.Network(
            n_input_dims=1,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 1,
            }
        )
        self.b_mlp = tcnn.Network(
            n_input_dims=1,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 1,
            }
        )
    def forward(self, rgb):
        B,S,C = rgb.shape
        rgb=rgb.view(-1,C)
        r,g,b = torch.split(rgb,[1,1,1],dim=-1)
        r_out = self.r_mlp(r)
        g_out = self.g_mlp(g)
        b_out = self.b_mlp(b)
        rgb_out = torch.concat([r_out,g_out,b_out],dim=-1)
        rgb_out = torch.sigmoid(rgb_out)
        rgb_out = rgb_out.view(B,S,C)
        return rgb_out
    def get_params(self):
        nn_params = [
            self.r_mlp.named_parameters(prefix="r_mlp"),
            self.g_mlp.named_parameters(prefix="g_mlp"),
            self.b_mlp.named_parameters(prefix="b_mlp")
        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}

        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }

class Main_tonemapping_Mono(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.mlp = tcnn.Network(
            n_input_dims=3,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 1,
            }
        )
    def forward(self, rgb):
        B,S,C = rgb.shape
        rgb=rgb.view(-1,C)
        rgb_out = self.mlp(rgb)
        rgb_out = rgb_out.view(B,S,C)
        return rgb_out        
    def get_params(self):
        nn_params = [
            self.mlp.named_parameters(prefix="mlp"),
        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }

class NeRV_tonemapping(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,rgb):
        tone_mapped = rgb/(1.0+rgb)
        return tone_mapped
    
