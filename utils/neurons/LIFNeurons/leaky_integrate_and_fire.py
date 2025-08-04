import torch

"""
Single Neuron Class
"""
class LIFNeuron:
    def __init__(self, v_rest=0.0, v_thresh=1.0, v_reset=0.0, tau=20.0, r=1.0, device=None):
        self.device = device

        # Neuron parameters
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau = tau
        self.r = r

        # Internal state
        self.v = torch.tensor([v_rest], dtype=torch.float32, device=self.device)  # Membrane potential
        self.input_current = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.spike = torch.zeros(1, dtype=torch.uint8, device=self.device)  # 0 or 1

    def receive_input(self, current: torch.Tensor):
        """Accumulate input current from other neurons or stimuli."""
        self.input_current += current.to(self.device)

    def update(self, global_time=None, global_wave=None, global_pulse=None):
        """Update neuron state based on LIF dynamics."""
        # LIF equation: dv = (-(v - v_rest) + I*R) / tau
        dv = (-(self.v - self.v_rest) + self.r*self.input_current) / self.tau
        self.v += dv

        # Check if neuron spikes
        if self.v >= self.v_thresh:
            self.spike[...] = 1
            self.v[...] = self.v_reset  # Reset after spike
        else:
            self.spike[...] = 0

        # Clear input current for next time step
        self.input_current.zero_()

    @property
    def output(self) -> torch.Tensor:
        """Return current spike output (1 if spiked, else 0)."""
        return self.spike


"""
A Group of LIF Neurons Class (Vectorized for GPU acceleration)
"""
class LIFNeuronGroup:
    def __init__(self, size, v_rest=0.0, v_thresh=1.0, v_reset=0.0, tau=20.0, r=1.0, device=None):
        self.size = size
        self.device = device
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau = tau
        self.r = r

        self.v_trace = []
        self.spike_trace = []

        self.v = torch.full((size,), v_rest, device=device)
        self.input_current = torch.zeros(size, device=device)
        self.spiked = torch.zeros(size, dtype=torch.uint8, device=device)

    def receive_input(self, current):
        self.input_current += current

    def update(self):
        dv = (-(self.v - self.v_rest) + self.r*self.input_current) / self.tau
        self.v += dv

        self.spiked = (self.v >= self.v_thresh).to(torch.uint8)
        self.v = torch.where(self.spiked.bool(), torch.tensor(self.v_reset, device=self.device), self.v)

        self.v_trace.append(self.v)
        self.spike_trace.append(self.spiked)

        self.input_current.zero_()