import numpy as np
import cv2
import matplotlib.pyplot as plt
from brian2 import *
import os
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score

# ========== 1. File Setup ==========

stimuli_dir = 'C:/Users/ayaan/Downloads/ALLSTIMULI/ALLSTIMULI'
fixmap_dir = 'C:/Users/ayaan/Downloads/ALLFIXATIONMAPS/ALLFIXATIONMAPS'

# Use the correct filename for the stimulus (which ends in .jpeg)
stim_filename = 'i05june05_static_street_boston_p1010764.jpeg'

# Generate the corresponding fixation map name (which ends in _fixMap.jpg)
base_name = os.path.splitext(stim_filename)[0]
fix_filename = f"{base_name}_fixMap.jpg"

# Full paths
stim_path = os.path.join(stimuli_dir, stim_filename)
fix_path = os.path.join(fixmap_dir, fix_filename)

# Debug checks
print("Stimulus Image Path:", stim_path)
print("Fixation Map Path:", fix_path)
print("Stimulus exists?", os.path.exists(stim_path))
print("Fixation map exists?", os.path.exists(fix_path))

# ========== 2. Load Stimulus Image ==========

screen_img = cv2.imread(stim_path)
if screen_img is None:
    raise FileNotFoundError(f"Failed to load image: {stim_path}")

screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
screen_img = cv2.resize(screen_img, (192, 108))

# ========== 3. Compute Saliency ==========
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

sal_map = compute_opencv_saliency(screen_img)
filtered_sal_map = filter_saliency_map(sal_map)
input_vector_base = saliency_to_exc_input(filtered_sal_map, grid_shape=(20, 20))

# ========== 4. Brian2 Neuron Grid ==========
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

# Mask corners
input_vector_masked = input_vector_base.copy()
omit_indices = [0 * nx + 0, 19 * nx + 19, 19 * nx + 18]
for idx in omit_indices:
    input_vector_masked[idx] = 0

# ========== 5. Run Neuron Simulation ==========
excitation = 2.5
inhibition = 0.1

S_inhib.w[:] = inhibition
G.v = v_rest + (5 * mV) * np.random.rand(N)
G.ge = 0
G.gi = 0
G.ge = input_vector_masked * excitation

spikemon = SpikeMonitor(G)
run(300 * ms)

spike_counts = np.bincount(spikemon.i, minlength=N)
combined_heatmap = spike_counts.reshape((ny, nx))

# ========== 6. Load Fixation Map ==========
fixation_map = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
fixation_map = cv2.resize(fixation_map, (nx, ny)) / 255.0

# ========== 7. Evaluate (AUC) ==========
if combined_heatmap.max() > 0:
    model_output = combined_heatmap / np.max(combined_heatmap)
else:
    model_output = combined_heatmap.astype(np.float32)

# Flatten both
flat_model = model_output.flatten()
flat_fix = fixation_map.flatten()
binary_fix = (flat_fix > 0.5).astype(np.uint8)

auc = roc_auc_score(binary_fix, flat_model)
print(f"AUC Score: {auc:.4f}")

# ========== 8. Visualize ==========
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(screen_img)
axs[0].set_title("Stimulus")

axs[1].imshow(fixation_map, cmap='hot')
axs[1].set_title("Ground Truth Fixation Map")

axs[2].imshow(model_output, cmap='hot')
axs[2].set_title("Model Output Heatmap")

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.show()
