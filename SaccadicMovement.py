import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from brian2 import *
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv

# Dataset paths
load_dotenv()
stimuli_dir = os.getenv("STIMULI_DIR")
fixmap_dir = os.getenv("FIXMAP_DIR")

# Pick 5 image filenames
all_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith('.jpeg')])
selected_files = all_files[:5]

# Constants
nx, ny = 20, 20
excitation_values = np.linspace(1.5, 3.5, 5)
inhibition_values = np.linspace(0.3, 0.5, 5)
auc_accumulator = np.zeros((len(inhibition_values), len(excitation_values)))
image_count = 0

# Compute NSS
def compute_nss(pred_map, fixation_map):
    mean = np.mean(pred_map)
    std = np.std(pred_map)
    if std == 0:
        return 0
    norm_map = (pred_map - mean) / std
    return np.mean(norm_map[fixation_map > 0.5])

# Neuron parameters
Ee = 60 * mV
Ei = -80 * mV
v_rest = -70 * mV
v_thresh = -55 * mV
v_reset = -70 * mV
tau = 10 * ms
tau_e = 5 * ms
tau_i = 10 * ms

for stim_filename in selected_files:
    base_name = os.path.splitext(stim_filename)[0]
    fix_filename = f"{base_name}_fixMap.jpg"
    stim_path = os.path.join(stimuli_dir, stim_filename)
    fix_path = os.path.join(fixmap_dir, fix_filename)

    if not os.path.exists(stim_path) or not os.path.exists(fix_path):
        continue

    screen_img = cv2.imread(stim_path)
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
    screen_img = cv2.resize(screen_img, (192, 108))

    fixation_map = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
    fixation_map = cv2.resize(fixation_map, (nx, ny)) / 255.0
    flat_fix = fixation_map.flatten()
    binary_fix = (flat_fix > 0.5).astype(np.uint8)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = saliency.computeSaliency(screen_img)
    if not success:
        continue
    sal_map = cv2.GaussianBlur(sal_map, (5, 5), 0)
    sal_map = sal_map / np.max(sal_map)
    blurred = cv2.GaussianBlur(sal_map, (3, 3), 0)
    _, filtered = cv2.threshold(blurred, 0.2, 1.0, cv2.THRESH_TOZERO)
    input_vector_base = cv2.resize(filtered, (nx, ny)).flatten()

    omit_indices = [0 * nx + 0, 19 * nx + 19, 19 * nx + 18]
    for idx in omit_indices:
        input_vector_base[idx] = 0

    best_auc = -1
    best_output = None
    best_exc = 0
    best_inh = 0
    best_nss = -1

    # Parameter Sweep
    for inh_idx, inhibition in enumerate(inhibition_values):
        for exc_idx, excitation in enumerate(excitation_values):
            N = nx * ny
            eqs = '''
            dv/dt = (ge*(Ee - v) + gi*(Ei - v) + (v_rest - v)) / tau : volt
            dge/dt = -ge / tau_e : 1
            dgi/dt = -gi / tau_i : 1
            '''
            G = NeuronGroup(N, eqs, threshold='v > v_thresh', reset='v = v_reset', method='euler')
            G.v = v_rest + (5 * mV) * np.random.rand(N)

            S_inhib = Synapses(G, G, 'w : 1', on_pre='gi_post += w')
            S_inhib.connect(condition='i != j')
            S_inhib.w[:] = inhibition

            S_self = Synapses(G, G, on_pre='ge_post += 0.02')
            S_self.connect(condition='i == j')

            G.ge = input_vector_base * excitation
            G.gi = 0

            spikemon = SpikeMonitor(G)
            run(300 * ms, report=None)

            spike_counts = np.bincount(spikemon.i, minlength=N).reshape((ny, nx))
            model_output = spike_counts / spike_counts.max() if spike_counts.max() > 0 else spike_counts
            flat_model = model_output.flatten()

            try:
                # Find best AUC and NSS
                auc = roc_auc_score(binary_fix, flat_model)
                nss = compute_nss(model_output, fixation_map)
                auc_accumulator[inh_idx, exc_idx] += auc
                if auc > best_auc:
                    best_auc = auc
                    best_output = model_output.copy()
                    best_exc = excitation
                    best_inh = inhibition
                    best_nss = nss
            except:
                continue

    # Plot Figures
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(screen_img)
    axs[0].set_title(f"Stimulus: {stim_filename}")

    axs[1].imshow(fixation_map, cmap='hot')
    axs[1].set_title("Fixation Map (Dataset)")

    axs[2].imshow(best_output, cmap='hot')
    axs[2].set_title(f"Model Output\nExc={best_exc}, Inh={best_inh}\nAUC={best_auc:.4f}, NSS={best_nss:.2f}")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    image_count += 1

# AUC Heatmap
average_auc = auc_accumulator / image_count
fig, ax = plt.subplots(figsize=(10, 7))
cax = ax.imshow(average_auc, cmap='hot', interpolation='nearest', aspect='auto',
                extent=[excitation_values[0], excitation_values[-1],
                        inhibition_values[-1], inhibition_values[0]])
ax.set_title("Average AUC Heatmap across 5 images")
ax.set_xlabel("Excitation Scaling")
ax.set_ylabel("Inhibition Strength")
ax.set_xticks(excitation_values)
ax.set_yticks(inhibition_values)
fig.colorbar(cax, label="Average AUC Score")
plt.tight_layout()
plt.show()
