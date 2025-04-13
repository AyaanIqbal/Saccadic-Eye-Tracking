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
# 7. Optional: Unit Tests
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

# =============================
# 4. Brian2 Neuron Model Setup
# =============================
nx, ny = 20, 20
N = nx * ny
Ee = 0 * mV
Ei = -80 * mV
v_rest = -70 * mV
v_thresh = -50 * mV
v_reset = -70 * mV
tau = 10 * ms
tau_e = 5 * ms
tau_i = 10 * ms

# LIF model with conductance-based synapses
eqs = '''
dv/dt = (ge*(Ee - v) + gi*(Ei - v) + (v_rest - v)) / tau : volt
dge/dt = -ge / tau_e : 1
dgi/dt = -gi / tau_i : 1
'''

G = NeuronGroup(N, eqs, threshold='v > v_thresh', reset='v = v_reset', method='euler')
G.v = v_rest + (5 * mV) * np.random.rand(N)

# Lateral inhibition and self-excitation
S_inhib = Synapses(G, G, on_pre='gi_post += 0.2')
S_inhib.connect(condition='i != j')

S_self = Synapses(G, G, on_pre='ge_post += 0.02')
S_self.connect(condition='i == j')

# Spike monitor
spikemon = SpikeMonitor(G)

# =============================
# 5. Capture + Saliency Loop
# =============================
print("⏳ Waiting 2 seconds before starting capture loop...")
time.sleep(2)

start_time = time.time()
end_time = start_time + 5  # Run for 5 seconds
iteration = 0

while time.time() < end_time:
    iteration += 1
    print(f"\n▶️ Iteration {iteration}")

    G.v = v_rest + (5 * mV) * np.random.rand(N)  # Reset voltages
    G.ge = 0
    G.gi = 0

    screen_img = capture_screen_mss()
    sal_map = compute_opencv_saliency(screen_img)
    input_vector = saliency_to_exc_input(sal_map, grid_shape=(nx, ny))
    G.ge = input_vector * 2.0

    run(200 * ms)

    winner = spikemon.i[-1] if len(spikemon.i) > 0 else None
    if winner is not None:
        x = winner % nx
        y = winner // nx
        print(f"🔥 Predicted saccade: neuron ({x}, {y})")
    else:
        print("⚠️ No saccade triggered.")

# =============================
# 6. Combined Plot: Screen Capture, Neuron Spike Heatmap, Saliency Input Map
# =============================

# Count how many times each neuron fired
firing_counts = np.zeros(N, dtype=int)
for neuron_id in spikemon.i:
    firing_counts[neuron_id] += 1

# Reshape to 2D grid
firing_grid = firing_counts.reshape((ny, nx))

# Use the final screen_img and sal_map from the last iteration (already defined)
# Reuse the saliency input sent to neurons
resized_saliency_input = input_vector.reshape((ny, nx))

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Screen capture
axs[0].imshow(screen_img)
axs[0].set_title("Test: Screen Capture")
axs[0].axis("off")

# Plot 2: Neuron spike heatmap
heatmap = axs[1].imshow(firing_grid, cmap='hot', interpolation='nearest')
axs[1].set_title("Neuron Spike Heatmap")
axs[1].set_xlabel("Neuron X Position")
axs[1].set_ylabel("Neuron Y Position")
cbar1 = fig.colorbar(heatmap, ax=axs[1], fraction=0.046, pad=0.04)
cbar1.set_label("Spike Count")

# Plot 3: Resized saliency neuron input
saliency_plot = axs[2].imshow(resized_saliency_input, cmap='hot', interpolation='nearest')
axs[2].set_title("Resized Saliency Input (Used by Neurons)")
axs[2].set_xlabel("Neuron X Position")
axs[2].set_ylabel("Neuron Y Position")
cbar2 = fig.colorbar(saliency_plot, ax=axs[2], fraction=0.046, pad=0.04)
cbar2.set_label("Saliency Intensity")

plt.tight_layout()
plt.show()