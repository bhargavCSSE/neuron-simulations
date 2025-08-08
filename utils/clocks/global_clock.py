import torch
import time
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global simulation state
GLOBAL_TIME = torch.zeros(1, device=device)
GLOBAL_WAVE = torch.zeros(1, device=device)
GLOBAL_PULSE = torch.zeros(1, dtype=torch.uint8, device=device)
GLOBAL_SPIKE = torch.zeros(1, dtype=torch.uint8, device=device)

class Clock:
    """
    GPU sine wave clock with binary pulse output.
    Can run in real-time or simulation-step mode.
    Updates: GLOBAL_TIME, GLOBAL_WAVE, GLOBAL_PULSE, GLOBAL_SPIKE
    """

    def __init__(self, freq: float, sample_rate: int, amplitude: float, real_time: bool = True):
        self.freq = freq
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.dt = 1.0 / sample_rate
        self.real_time = real_time
        self.running = False
        self.thread = None

        # Preallocate tensor for time to avoid reallocs
        self._t_tensor = torch.zeros(1, device=device)

    def start(self):
        self.running = True
        if self.real_time:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def run(self):
        """Real-time mode loop."""
        self.start_time = time.perf_counter()
        next_time = self.start_time
        prev_pulse_state = GLOBAL_PULSE.clone()

        while self.running:
            now = time.perf_counter()
            if now >= next_time:
                t = now - self.start_time
                self._update_step(t, prev_pulse_state)
                next_time += self.dt
            else:
                time.sleep(0.0001)

    def step(self, sim_time: float):
        """Simulation mode: call manually with simulation time."""
        prev_pulse_state = GLOBAL_PULSE.clone()
        self._update_step(sim_time, prev_pulse_state)

    def _update_step(self, t: float, prev_pulse_state: torch.Tensor):
        self._t_tensor.fill_(t)

        wave = self.amplitude * torch.sin(2 * torch.pi * self.freq * self._t_tensor)
        pulse = (wave > 0).to(torch.uint8)
        spike = (pulse ^ prev_pulse_state).to(torch.uint8)

        prev_pulse_state.copy_(pulse)

        GLOBAL_TIME.copy_(self._t_tensor)
        GLOBAL_WAVE.copy_(wave)
        GLOBAL_PULSE.copy_(pulse)
        GLOBAL_SPIKE.copy_(spike)
