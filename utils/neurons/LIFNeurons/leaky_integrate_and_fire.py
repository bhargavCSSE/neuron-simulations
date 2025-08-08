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
    def __init__(
        self,
        size,
        v_rest=0.0,
        v_thresh=1.0,
        v_reset=0.0,
        tau=20.0,
        r=1.0,
        refractory_period=5.0,
        device=None,
        log_traces=False
    ):
        self.size = size
        self.device = device
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset_val = v_reset
        self.tau = tau
        self.r = r
        self.refractory_period_val = refractory_period
        self.log_traces = log_traces

        # State tensors
        self.v = torch.full((size,), v_rest, device=device)
        self.input_current = torch.zeros(size, device=device)
        self.spiked = torch.zeros(size, dtype=torch.uint8, device=device)
        self.refractory_counter = torch.zeros(size, device=device)

        # Preallocated constants
        self.v_reset = torch.full((size,), v_reset, device=device)
        self.refractory_period = torch.full((size,), refractory_period, device=device)

        # Traces
        if log_traces:
            self.spike_trace = []
            self.full_v_trace = []
            self.full_spike_trace = []

    def reset_trace(self):
        if self.log_traces:
            self.spike_trace.clear()
            self.full_v_trace.clear()
            self.full_spike_trace.clear()

    def receive_input(self, current: torch.Tensor):
        self.input_current += current

    def update(self, dt=1.0):
        # Decay refractory counters
        self.refractory_counter -= 1
        self.refractory_counter.clamp_min_(0)

        not_refractory = self.refractory_counter == 0

        dv = (-(self.v - self.v_rest) + self.r * self.input_current) / self.tau
        self.v = torch.where(not_refractory, self.v + dv * dt, self.v)

        spiked_now = (self.v >= self.v_thresh) & not_refractory
        self.spiked = spiked_now.to(torch.uint8)

        self.v = torch.where(spiked_now, self.v_reset, self.v)
        self.refractory_counter = torch.where(spiked_now, self.refractory_period, self.refractory_counter)

        if self.log_traces:
            self.spike_trace.append(self.spiked.clone())
            self.full_v_trace.append(self.v.clone())
            self.full_spike_trace.append(self.spiked.clone())

        self.input_current.zero_()
