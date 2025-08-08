import torch

"""
Single synapse between LIF neurons
"""
class Synapse:
    def __init__(self, pre_neuron, post_neuron, weight=1.0):
        self.pre = pre_neuron
        self.post = post_neuron
        self.weight = weight

    def update(self):
        if self.pre.output.item() == 1:
            current = torch.tensor([self.weight], dtype=torch.float32, device=self.post.device)
            self.post.receive_input(current)

"""
Feed forward Synpases Between LIF neuron groups (Vectorized for GPU acceleration)
"""
class FeedForwardSynapses:
    """
    Feed-forward synapses between a presynaptic group (pre) and postsynaptic group (post).
    Plasticity uses a 3-factor rule:
        Δw ∝ dopamine * eligibility
    where eligibility E is accumulated each timestep using an STDP kernel (pre/post traces),
    and dopamine is delivered later (e.g., at end of a trial) to convert E into weight updates.

    We also model a **1-timestep conduction delay** so that pre spikes at time t
    only affect the postsynaptic current at time t (via last step's stored spikes).
    This creates clean pre→post causality for LTP.
    """

    def __init__(
        self,
        pre,
        post,
        weight_scale=1.0,       # initial weight scale (uniform random [0, weight_scale))
        stdp_lr=0.01,           # learning rate for plasticity
        tau_pre=20.0,           # ms, pre trace time constant
        tau_post=20.0,          # ms, post trace time constant
        tau_e=200.0,            # ms, eligibility time constant (how long 'credit' persists)
        w_min=0.0,
        w_max=5.0,
    ):
        self.pre = pre
        self.post = post

        # Weight matrix shape: [post_neurons, pre_neurons]
        shape = (post.size, pre.size)
        self.weights = torch.rand(shape, device=pre.device) * weight_scale

        # Plasticity hyperparams
        self.stdp_lr = stdp_lr
        self.w_min, self.w_max = w_min, w_max

        # Precompute scalar decay factors once (applied each timestep)
        # Using exp(-dt/tau) with dt=1 ms (you can treat each update() call as 1 ms)
        self.decay_pre  = torch.exp(torch.tensor(-1.0 / tau_pre,  device=pre.device))
        self.decay_post = torch.exp(torch.tensor(-1.0 / tau_post, device=pre.device))
        self.decay_E    = torch.exp(torch.tensor(-1.0 / tau_e,    device=pre.device))

        # Activity traces (eligibility ingredients)
        # pre_trace[i]  ~ recent history of presynaptic neuron i
        # post_trace[j] ~ recent history of postsynaptic neuron j
        self.pre_trace  = torch.zeros(pre.size,  device=pre.device)
        self.post_trace = torch.zeros(post.size, device=pre.device)

        # Eligibility matrix E[j, i] accumulates the STDP "credit" for synapse i→j
        self.E = torch.zeros(*shape, device=pre.device)

        # One‑timestep conduction delay buffer:
        # we store current pre spikes and use them for propagation next timestep.
        self.pre_spike_buffer = torch.zeros(pre.size, device=pre.device)

    # ---------- Forward propagation ----------

    def propagate(self):
        """
        Compute postsynaptic input current from *last step’s* presynaptic spikes.
        This implements a one‑timestep synaptic delay, enforcing causal pre→post timing.
        """
        # Convert buffered (delayed) presynaptic spikes into current
        # (weights @ spikes) -> current for each postsyn neuron
        input_current = torch.matmul(self.weights, self.pre_spike_buffer)
        # Deliver current to the postsynaptic group (will be integrated in its update())
        self.post.receive_input(input_current)

    def tick_end(self):
        """
        Call at the END of each simulation step:
        shift current presynaptic spikes into the buffer so they will affect
        the next step's propagation.
        """
        self.pre_spike_buffer = self.pre.spiked.float().clone()

    # ---------- Plasticity (2-factor: STDP → eligibility) ----------

    def update_eligibility(self):
        """
        Update pre/post traces and accumulate eligibility E for every synapse.
        This step does **not** change weights yet — it only updates E.
        Later, dopamine will scale E to change weights (see reinforce()).
        """
        # 1) Exponential decay of pre/post traces (recent activity memory)
        self.pre_trace  *= self.decay_pre
        self.post_trace *= self.decay_post

        # 2) Read this step's spikes as floats (0/1)
        pre_spk  = self.pre.spiked.float()   # shape: [pre_neurons]
        post_spk = self.post.spiked.float()  # shape: [post_neurons]

        # 3) Compute STDP contributions using *previous* traces (causal):
        #    LTP: post fires now, given recent pre activity -> strengthens w
        ltp = torch.outer(post_spk, self.pre_trace)    # shape: [post, pre]
        #    LTD: pre fires now, given recent post activity -> weakens w
        ltd = torch.outer(self.post_trace, pre_spk)    # shape: [post, pre]

        # 4) Accumulate eligibility (no dopamine yet)
        self.E *= self.decay_E             # eligibility decays over time
        self.E += (ltp - ltd)              # add current STDP credit

        # 5) Update traces AFTER using them
        self.pre_trace  += pre_spk
        self.post_trace += post_spk

    # ---------- Dopamine (3rd factor) ----------

    def reinforce(self, da: float):
        """
        Convert accumulated eligibility E into actual weight updates, scaled by dopamine.
        Positive da -> reward -> reinforce recent causal pre→post pairings.
        Negative da -> punishment -> depress them.
        """
        if da == 0.0:
            return

        # Weight update: Δw = η * da * E
        self.weights += self.stdp_lr * da * self.E

        # Keep weights in a reasonable range
        self.weights.clamp_(self.w_min, self.w_max)

    def zero_eligibility(self):
        """Reset eligibility (e.g., between episodes if desired)."""
        self.E.zero_()


    