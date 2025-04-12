# saccadic_gaze_predictor.py

import numpy as np
import cv2
import mss
import matplotlib.pyplot as plt
import time
from brian2 import *

# =============================
# 1. Screen Capture with MSS
# =============================
def capture_screen_mss(resize_to=(192, 108)):
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]  # Drop alpha
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize_to:
            img = cv2.resize(img, resize_to)
        return img

# =============================
# 2. OpenCV Saliency Map
# =============================
def compute_opencv_saliency(image):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        raise Exception("Saliency computation failed")
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    saliency_map = saliency_map / np.max(saliency_map)
    return saliency_map

# =============================
# 3. Resize to Neural Input
# =============================
def saliency_to_exc_input(saliency_map, grid_shape=(20, 20)):
    resized = cv2.resize(saliency_map, grid_shape)
    return resized.flatten()

# =============================
# 4. Brian2 Neuron Model Setup
# =============================
nx, ny = 20, 20
N = nx * ny
Ee = 0*mV
Ei = -80*mV
v_rest = -70*mV
v_thresh = -50*mV
v_reset = -70*mV
tau = 10*ms
tau_e = 5*ms
tau_i = 10*ms

# LIF model with conductance
eqs = '''
dv/dt = (ge*(Ee - v) + gi*(Ei - v) + (v_rest - v)) / tau : volt
dge/dt = -ge / tau_e : 1
dgi/dt = -gi / tau_i : 1
'''

G = NeuronGroup(N, eqs, threshold='v > v_thresh', reset='v = v_reset', method='euler')
G.v = v_rest + (5 * mV) * np.random.rand(N)

# Lateral inhibition + self-excitation
S_inhib = Synapses(G, G, on_pre='gi_post += 0.2')
S_inhib.connect(condition='i != j')

S_self = Synapses(G, G, on_pre='ge_post += 0.02')
S_self.connect(condition='i == j')

# =============================
# 5. Wait and Run Repeatedly Over 5 Seconds
# =============================
print("⏳ Waiting 2 seconds before starting capture loop...")
time.sleep(2)

start_time = time.time()
end_time = start_time + 5  # run for 5 seconds
iteration = 0

all_spikes = []
INPUT_GAINS = [0.5, 1.0, 1.5, 2.0, 3.0]  # Auto-tune input scaling

while time.time() < end_time:
    iteration += 1
    print(f"\n▶️ Iteration {iteration}")

    screen_img = capture_screen_mss()
    sal_map = compute_opencv_saliency(screen_img)
    input_vector = saliency_to_exc_input(sal_map, grid_shape=(nx, ny))

    print(f"Saliency Input — Max: {np.max(input_vector):.2f}, Mean: {np.mean(input_vector):.2f}")

    best_gain = None
    for gain in INPUT_GAINS:
        G.v = v_rest + (5 * mV) * np.random.rand(N)
        G.ge = 0
        G.gi = 0

        spikemon = SpikeMonitor(G)
        G.ge = input_vector * gain
        run(200*ms)

        if len(spikemon.i) > 0:
            spikes = np.unique(spikemon.i)
            all_spikes.extend(spikes)
            best_gain = gain
            print(f"🔥 Gain {gain}: Neurons activated → {list(spikes)}")
            break  # Stop at first gain that triggers spikes
        else:
            print(f"❌ Gain {gain}: No saccades triggered.")

    if best_gain is None:
        print("⚠️ All gain levels failed to activate neurons.")

# =============================
# 6. Plot All Activated Neurons
# =============================
plt.figure(figsize=(6, 6))
spike_counts = np.zeros((ny, nx))

for idx in all_spikes:
    x = idx % nx
    y = idx // nx
    spike_counts[y, x] += 1

plt.imshow(spike_counts, cmap='hot', interpolation='nearest')
plt.title("All Neurons Activated Over 5 Seconds")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.colorbar(label='Spike Count')
plt.tight_layout()
plt.show()

# =============================
# 7. Unit Tests with Image Output
# =============================
def test_capture_screen():
    img = capture_screen_mss(resize_to=(100, 100))
    assert img.shape == (100, 100, 3), "Screen capture should return a 100x100 RGB image"
    plt.imshow(img)
    plt.title("Test: Screen Capture")
    plt.axis("off")
    plt.show()

def test_saliency_map():
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    sal = compute_opencv_saliency(dummy_img)
    assert sal.shape == (100, 100), "Saliency map should match input resolution"
    assert np.max(sal) <= 1.0 and np.min(sal) >= 0.0, "Saliency map should be normalized"
    plt.imshow(sal, cmap='hot')
    plt.title("Test: Saliency Map")
    plt.axis("off")
    plt.show()

def test_saliency_to_input():
    dummy_map = np.random.rand(100, 100)
    input_vec = saliency_to_exc_input(dummy_map, grid_shape=(20, 20))
    assert input_vec.shape == (400,), "Flattened saliency input should have 400 elements"
    plt.imshow(dummy_map, cmap='hot')
    plt.title("Test: Original Saliency Map")
    plt.axis("off")
    plt.show()
    plt.imshow(input_vec.reshape(20, 20), cmap='hot')
    plt.title("Test: Resized Saliency Map (Neuron Input)")
    plt.axis("off")
    plt.show()

# Run tests manually
if __name__ == '__main__':
    test_capture_screen()
    test_saliency_map()
    test_saliency_to_input()
    print("All tests passed.")
