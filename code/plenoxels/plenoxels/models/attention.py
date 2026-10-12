import tinycudann as tcnn
import torch
import torch.nn as nn
import math
class Attention_base(nn.Module):
    def __init__(self, feature_dim, decoder_dim):
        super().__init__()

        self.query_linear = nn.Linear(3,feature_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "None",
                "n_hidden_layers": 0,
            }
        )
        """
        self.key_linear = nn.Linear(3,feature_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "None",
                "n_hidden_layers": 0,
            }
        )
        """

        self.value_linear = nn.Linear(3,feature_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "None",
                "n_hidden_layers": 0,
            }
        )
        """
        self.attention_decoder = nn.Linear(feature_dim,decoder_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=decoder_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "Sigmoid",
                "n_hidden_layers": 0,
            }
        )
        """
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, obj, light):
        #obj: B * #Sample *3
        #light: B * #Sample *3
        query = self.query_linear(obj)
        key = self.key_linear(light)
        value = self.value_linear(light)
        key_t = key.transpose(1,2)
        score = (query @ key_t)/ math.sqrt(3)
        score = self.softmax(score)
        value = score@value
        output = self.attention_decoder(value)
        output = torch.sigmoid(output)
        return output

    def test_similarity(self, obj, light):
        # Same diagnostic as MLP_base.test_similarity; Attention_base.forward has the
        # matching (obj, light) signature. Used by app_test_no_light at validation.
        one_out = self.forward(obj, obj)
        one_out = torch.ones_like(one_out) - one_out
        one_out = torch.concat([one_out, torch.ones_like(one_out) - self.forward(light, light)], dim=0)
        zero_out = self.forward(obj, torch.zeros_like(obj))
        zero_out = torch.concat([zero_out, self.forward(light, torch.zeros_like(obj))], dim=0)
        zero_out = zero_out.view(-1, zero_out.shape[-1])
        one_out = one_out.view(-1, one_out.shape[-1])
        return one_out, zero_out

    def get_params(self):
        nn_params = [
            self.query_linear.named_parameters(prefix="query_linear"),
            self.key_linear.named_parameters(prefix="key_linear"),
            self.value_linear.named_parameters(prefix="value_linear"),
            self.attention_decoder.named_parameters(prefix="attention_decoder")

        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}

        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }
class Attention_base_w_tonemapping(nn.Module):
    def __init__(self, feature_dim, decoder_dim, tonemapping):
        super().__init__()

        self.query_linear = nn.Linear(3,feature_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "None",
                "n_hidden_layers": 0,
            }
        )
        """
        self.key_linear = nn.Linear(3,feature_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "None",
                "n_hidden_layers": 0,
            }
        )
        """

        self.value_linear = nn.Linear(3,feature_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "None",
                "n_hidden_layers": 0,
            }
        )
        """
        self.attention_decoder = nn.Linear(feature_dim,decoder_dim,bias=False) 
        """
        tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=decoder_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "output_activation": "Sigmoid",
                "n_hidden_layers": 0,
            }
        )
        """
        self.softmax = nn.Softmax(dim=-1)
        self.tonemapping = tonemapping

    def forward(self, obj, light):
        #obj: B * #Sample *3
        #light: B * #Sample *3
        obj = self.tonemapping(obj)
        light = self.tonemapping(light)

        query = self.query_linear(obj)
        key = self.key_linear(light)
        value = self.value_linear(light)
        key_t = key.transpose(1,2)
        score = (query @ key_t)/ math.sqrt(3)
        score = self.softmax(score)
        value = score@value
        output = self.attention_decoder(value)
        output = torch.sigmoid(output)
        return output
        
    def get_params(self):
        nn_params = [
            self.query_linear.named_parameters(prefix="query_linear"),
            self.key_linear.named_parameters(prefix="key_linear"),
            self.value_linear.named_parameters(prefix="value_linear"),
            self.attention_decoder.named_parameters(prefix="attention_decoder")

        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}

        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }

class MLP_base(nn.Module):
    def __init__(self, feature_dim, decoder_dim):
        super().__init__()
        self.decoder_dim = decoder_dim

        self.mapping = tcnn.Network(
            n_input_dims=2,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": feature_dim,
                "n_hidden_layers": 1,
            },
        )


    def forward(self, obj, light):
        #obj: B * #Sample *3
        #light: B * #Sample *3
        r_obj, g_obj, b_obj = torch.split(obj,[1,1,1],dim=-1)
        r_lgt, g_lgt, b_lgt = torch.split(light,[1,1,1],dim=-1)
        r_rad = torch.concat([r_obj,r_lgt],dim=-1)
        g_rad = torch.concat([g_obj,g_lgt],dim=-1)
        b_rad = torch.concat([b_obj,b_lgt],dim=-1)

        B,S,C = r_rad.shape
        r_rad = r_rad.view(-1,C)
        g_rad = g_rad.view(-1,C)
        b_rad = b_rad.view(-1,C)

        r_out = self.mapping(r_rad)
        g_out = self.mapping(g_rad)
        b_out = self.mapping(b_rad)
        output = torch.concat([r_out,g_out,b_out],dim=-1)
        output = output.view(B,S,self.decoder_dim)

        return output
        
    def get_params(self):
        nn_params = [
            self.mapping.named_parameters(prefix="mapping"),
        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}

        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }

    @torch.no_grad()
    def test_similarity(self,obj,light):

        one_out = self.forward(obj,obj)
        one_out = torch.ones_like(one_out)-one_out
        one_out = torch.concat([one_out,torch.ones_like(one_out)-self.forward(light,light)],dim=0)
        zero_out = self.forward(obj,torch.zeros_like(obj))
        zero_out = torch.concat([zero_out,self.forward(light,torch.zeros_like(obj))],dim=0)
        zero_out = zero_out.view(-1,zero_out.shape[-1])
        one_out = one_out.view(-1,one_out.shape[-1])

        return one_out, zero_out
    
class MLP_base_w_full(nn.Module):
    def __init__(self, feature_dim, decoder_dim):
        super().__init__()
        self.decoder_dim = decoder_dim

        self.mapping = tcnn.Network(
            n_input_dims=2,
            n_output_dims=8,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": feature_dim,
                "n_hidden_layers": 1,
            },
        )
        self.mapping2 = tcnn.Network(
            n_input_dims=24,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": feature_dim,
                "n_hidden_layers": 1,
            },
        )

    def forward(self, obj, light):
        #obj: B * #Sample *3
        #light: B * #Sample *3
        r_obj, g_obj, b_obj = torch.split(obj,[1,1,1],dim=-1)
        r_lgt, g_lgt, b_lgt = torch.split(light,[1,1,1],dim=-1)
        r_rad = torch.concat([r_obj,r_lgt],dim=-1)
        g_rad = torch.concat([g_obj,g_lgt],dim=-1)
        b_rad = torch.concat([b_obj,b_lgt],dim=-1)

        B,S,C = r_rad.shape
        r_rad = r_rad.view(-1,C)
        g_rad = g_rad.view(-1,C)
        b_rad = b_rad.view(-1,C)

        r_out = self.mapping(r_rad)
        g_out = self.mapping(g_rad)
        b_out = self.mapping(b_rad)
        output = torch.concat([r_out,g_out,b_out],dim=-1)
        output = self.mapping2(output)
        output = output.view(B,S,self.decoder_dim)

        return output
        
    def get_params(self):
        nn_params = [
            self.mapping.named_parameters(prefix="mapping"),
            self.mapping.named_parameters(prefix="mapping2"),

        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}

        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }

    @torch.no_grad()
    def test_similarity(self,obj,light):

        one_out = self.forward(obj,obj)
        one_out = torch.ones_like(one_out)-one_out
        one_out = torch.concat([one_out,torch.ones_like(one_out)-self.forward(light,light)],dim=0)
        zero_out = self.forward(obj,torch.zeros_like(obj))
        zero_out = torch.concat([zero_out,self.forward(light,torch.zeros_like(obj))],dim=0)
        zero_out = zero_out.view(-1,zero_out.shape[-1])
        one_out = one_out.view(-1,one_out.shape[-1])

        return one_out, zero_out
    
class MLP_base_w_brdf(nn.Module):
    def __init__(self, feature_dim, decoder_dim, tonemapping):
        super().__init__()
        #Note that! this 'tonemapping' function should be not trainable(because we define it in the initialize function)
        self.decoder_dim = decoder_dim
        self.MLP_base = tcnn.Network(
            n_input_dims=6,
            n_output_dims=decoder_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": feature_dim,
                "n_hidden_layers": 1,
            },
        )
        self.tonemapping = tonemapping
    def forward(self, obj, light):
        #obj: B * #Sample *3
        #light: B * #Sample *3
        obj_tonemapped = self.tonemapping(obj)
        light_tonemapped = self.tonemapping(light)

        input = torch.concat([obj_tonemapped,light_tonemapped],dim=-1)
        B,S,C = input.shape
        input = input.view(-1,C)
        
        output = self.MLP_base(input)
        output = output.view(B,S,self.decoder_dim)

        return output
        
    def get_params(self):
        nn_params = [
            self.MLP_base.named_parameters(prefix="MLP_base"),
        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys()
        )}

        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values())
        }
    @torch.no_grad()
    def test_similarity(self,obj,light):
        one_out = self.forward(obj,obj)
        one_out = torch.ones_like(one_out)-one_out
        one_out = torch.concat([one_out,torch.ones_like(one_out)-self.forward(light,light)],dim=0)
        zero_out = self.forward(obj,torch.zeros_like(obj))
        zero_out = torch.concat([zero_out,self.forward(light,torch.zeros_like(obj))],dim=0)
        zero_out = zero_out.view(-1,zero_out.shape[-1])
        one_out = one_out.view(-1,one_out.shape[-1])

        return one_out, zero_out