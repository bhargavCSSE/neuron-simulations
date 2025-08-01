import torch
import time
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_TIME = torch.zeros(1, device=device)
GLOBAL_WAVE = torch.zeros(1, device=device)
GLOBAL_PULSE = torch.zeros(1, dtype=torch.uint8, device=device)
GLOBAL_SPIKE = torch.zeros(1, dtype=torch.uint8, device=device)

class clock:
    """
    Real-time GPU sine wave clock with binary pulse output.
    Updates global tensors: GLOBAL_TIME, GLOBAL_WAVE, GLOBAL_PULSE
    """
        
    def __init__(self, freq:float, sample_rate:int, amplitude:float):
        self.freq = freq
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def run(self):
        self.start_time = time.perf_counter()
        next_time = self.start_time
        prev_pulse_state = GLOBAL_PULSE.clone()
        dt = 1.0/self.sample_rate

        while self.running:
            now = time.perf_counter()
            
            if now >= next_time:
                t = now - self.start_time
                
                t_tensor = torch.tensor([t], device=device)
                
                # time = A*sin(2*pi*freq*time + displacement)
                wave = self.amplitude*torch.sin(2*torch.pi*self.freq*t_tensor + 0.0)
                pulse = (wave > 0).to(torch.uint8)
                spike = (pulse ^ prev_pulse_state).to(torch.uint8)

                prev_pulse_state.copy_(pulse)

                GLOBAL_TIME.copy_(t_tensor)
                GLOBAL_WAVE.copy_(wave)
                GLOBAL_PULSE.copy_(pulse)
                GLOBAL_SPIKE.copy_(spike)
                
                next_time += dt
            else:
                # Yield CPU briefly to avoid burning 100%
                time.sleep(0.0001)