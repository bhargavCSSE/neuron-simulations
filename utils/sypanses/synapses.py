import torch


class Synapse:
    def __init__(self, pre_neuron, post_neuron, weight=1.0):
        self.pre = pre_neuron
        self.post = post_neuron
        self.weight = weight

    def update(self):
        if self.pre.output.item() == 1:
            current = torch.tensor([self.weight], dtype=torch.float32, device=self.post.device)
            self.post.receive_input(current)
            
class FeedForwardSynapses:
    def __init__(self, pre, post, weight_scale=1.0):
        self.pre = pre
        self.post = post
        shape = (post.size, pre.size)
        self.weights = torch.rand(shape, device=pre.device) * weight_scale

    def propagate(self):
        input_spikes = self.pre.spiked.float()
        input_current = torch.matmul(self.weights, input_spikes)
        self.post.receive_input(input_current)