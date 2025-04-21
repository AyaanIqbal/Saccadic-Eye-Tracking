import numpy as np
import cv2
import mss
import matplotlib.pyplot as plt
import time
from brian2 import *

# 1. Capture + Saliency

def capture_screen_mss(resize_to=(192, 108)):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize_to:
            img = cv2.resize(img, resize_to)
        return img

def compute_opencv_saliency(image):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        raise Exception("Saliency computation failed")
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    return saliency_map / np.max(saliency_map)

def filter_saliency_map(saliency_map, threshold=0.2):
    blurred = cv2.GaussianBlur(saliency_map, (3, 3), 0)
    _, filtered = cv2.threshold(blurred, threshold, 1.0, cv2.THRESH_TOZERO)
    return filtered

def saliency_to_exc_input(saliency_map, grid_shape=(20, 20)):
    resized = cv2.resize(saliency_map, grid_shape)
    return resized.flatten()

# 2. Brian2 Setup

nx, ny = 20, 20
N = nx * ny
Ee = 0 * mV
Ei = -80 * mV
v_rest = -70 * mV
v_thresh = -55 * mV
v_reset = -70 * mV
tau = 10 * ms
tau_e = 5 * ms
tau_i = 10 * ms

eqs = '''
dv/dt = (ge*(Ee - v) + gi*(Ei - v) + (v_rest - v)) / tau : volt
dge/dt = -ge / tau_e : 1
dgi/dt = -gi / tau_i : 1
'''

G = NeuronGroup(N, eqs, threshold='v > v_thresh', reset='v = v_reset', method='euler')
G.v = v_rest + (5 * mV) * np.random.rand(N)

S_inhib = Synapses(G, G, 'w : 1', on_pre='gi_post += w')
S_inhib.connect(condition='i != j')

S_self = Synapses(G, G, on_pre='ge_post += 0.02')
S_self.connect(condition='i == j')

# 3. Preprocess Input

screen_img = capture_screen_mss()
sal_map = compute_opencv_saliency(screen_img)
filtered_sal_map = filter_saliency_map(sal_map, threshold=0.2)
input_vector_base = saliency_to_exc_input(filtered_sal_map, grid_shape=(nx, ny))

# Mask specific neurons (0,0), (19,19), (18,19)
input_vector_masked = input_vector_base.copy()
omit_indices = [0 * nx + 0, 19 * nx + 19, 19 * nx + 18 ]
for idx in omit_indices:
    input_vector_masked[idx] = 0

resized_input_base = input_vector_masked.reshape((ny, nx))

# 4. Sweep Parameters

excitation_values = np.arange(1.5, 3.51, 0.25)
inhibition_values = np.linspace(0.05, 0.25, 8)
combined_heatmap = np.zeros((ny, nx), dtype=int)
trigger_counts = np.zeros((len(inhibition_values), len(excitation_values)))

for inhib_idx, inhib in enumerate(inhibition_values):
    S_inhib.w[:] = inhib

    for exc_idx, excitation in enumerate(excitation_values):
        print(f"\nInhibition: {inhib:.2f}, Excitation: {excitation:.2f}")

        G.v = v_rest + (5 * mV) * np.random.rand(N)
        G.ge = 0
        G.gi = 0
        G.ge = input_vector_masked * excitation  # <== Using masked input

        spikemon = SpikeMonitor(G)
        run(300 * ms)

        spike_counts = np.bincount(spikemon.i, minlength=N)
        trigger_counts[inhib_idx, exc_idx] = np.sum(spike_counts)

        if np.sum(spike_counts) > 0:
            winner = np.argmax(spike_counts)
            x, y = winner % nx, winner // nx
            combined_heatmap[y, x] += 1
            print(f"  Winner neuron: ({x}, {y}) | Spikes: {spike_counts[winner]}")
        else:
            print("  No neurons fired.")

# 5. Plot Figure 1 (Neuron Input & Heatmaps)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0][0].imshow(screen_img)
axs[0][0].set_title("Screen Capture")
axs[0][0].axis("off")

axs[0][1].imshow(resized_input_base, cmap='hot', interpolation='nearest')
axs[0][1].set_title("Neuron Input (Filtered Saliency)")
axs[0][1].set_xlabel("Neuron X")
axs[0][1].set_ylabel("Neuron Y")
fig.colorbar(axs[0][1].images[0], ax=axs[0][1], fraction=0.046, pad=0.04).set_label("Intensity")

axs[1][0].imshow(filtered_sal_map, cmap='hot', interpolation='nearest')
axs[1][0].set_title("Filtered Saliency Map")
axs[1][0].set_xlabel("Pixel X")
axs[1][0].set_ylabel("Pixel Y")
fig.colorbar(axs[1][0].images[0], ax=axs[1][0], fraction=0.046, pad=0.04).set_label("Intensity")

axs[1][1].imshow(combined_heatmap, cmap='hot', interpolation='nearest')
axs[1][1].set_title("Cumulative Winner Neuron Heatmap")
axs[1][1].set_xlabel("Neuron X")
axs[1][1].set_ylabel("Neuron Y")
fig.colorbar(axs[1][1].images[0], ax=axs[1][1], fraction=0.046, pad=0.04).set_label("Wins")

plt.tight_layout()
plt.show()

# 6. Plot Figure 2 (Parameter Heatmap)

plt.figure(figsize=(10, 8))
im = plt.imshow(trigger_counts, cmap='hot', interpolation='nearest',
                extent=[excitation_values[0], excitation_values[-1],
                        inhibition_values[-1], inhibition_values[0]],
                aspect='auto')

plt.title("Total Spikes by Excitation and Inhibition")
plt.xlabel("Excitation Scaling")
plt.ylabel("Inhibition Strength")
plt.xticks(excitation_values, rotation=45)
plt.yticks(inhibition_values)
cbar = plt.colorbar(im, fraction=0.025, pad=0.04)
cbar.set_label("Total Spikes")

plt.tight_layout()
plt.show()