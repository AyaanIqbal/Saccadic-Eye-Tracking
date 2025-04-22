import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from brian2 import *
from sklearn.metrics import roc_auc_score

# Dataset paths
stimuli_dir = r'C:\Users\ayaan\Downloads\ALLSTIMULI\ALLSTIMULI'
fixmap_dir = r'C:\Users\ayaan\Downloads\ALLFIXATIONMAPS\ALLFIXATIONMAPS'

# Pick 5 image filenames
all_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith('.jpeg')])
selected_files = all_files[:5]

# Constants
nx, ny = 20, 20
excitation_values = np.linspace(1.5, 3.5, 5)
inhibition_values = np.linspace(0.3, 0.5, 5)

for stim_filename in selected_files:
    base_name = os.path.splitext(stim_filename)[0]
    fix_filename = f"{base_name}_fixMap.jpg"
    stim_path = os.path.join(stimuli_dir, stim_filename)
    fix_path = os.path.join(fixmap_dir, fix_filename)

    # Skip if either file doesn't exist
    if not os.path.exists(stim_path) or not os.path.exists(fix_path):
        continue

    # Load stimulus image
    screen_img = cv2.imread(stim_path)
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
    screen_img = cv2.resize(screen_img, (192, 108))

    # Load fixation map
    fixation_map = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
    fixation_map = cv2.resize(fixation_map, (nx, ny)) / 255.0
    flat_fix = fixation_map.flatten()
    binary_fix = (flat_fix > 0.5).astype(np.uint8)

    # Compute saliency
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = saliency.computeSaliency(screen_img)
    if not success:
        continue
    sal_map = cv2.GaussianBlur(sal_map, (5, 5), 0)
    sal_map = sal_map / np.max(sal_map)
    blurred = cv2.GaussianBlur(sal_map, (3, 3), 0)
    _, filtered = cv2.threshold(blurred, 0.2, 1.0, cv2.THRESH_TOZERO)
    input_vector_base = cv2.resize(filtered, (nx, ny)).flatten()

    # Mask corners
    omit_indices = [0 * nx + 0, 19 * nx + 19, 19 * nx + 18]
    for idx in omit_indices:
        input_vector_base[idx] = 0

    best_auc = -1
    best_output = None
    best_exc = 0
    best_inh = 0

    for inhibition in inhibition_values:
        for excitation in excitation_values:
            # Setup neurons
            N = nx * ny
            eqs = '''
            dv/dt = (ge*(0*mV - v) + gi*(-80*mV - v) + (-70*mV - v)) / (10*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            dgi/dt = -gi / (10*ms) : 1
            '''
            G = NeuronGroup(N, eqs, threshold='v > -55*mV', reset='v = -70*mV', method='euler')
            G.v = -70*mV + (5 * mV) * np.random.rand(N)

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
                auc = roc_auc_score(binary_fix, flat_model)
                if auc > best_auc:
                    best_auc = auc
                    best_output = model_output.copy()
                    best_exc = excitation
                    best_inh = inhibition
            except:
                continue

    # Plot all three: stimulus, ground truth, and model output
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(screen_img)
    axs[0].set_title(f"Stimulus: {stim_filename}")

    axs[1].imshow(fixation_map, cmap='hot')
    axs[1].set_title("Fixation Map (Dataset)")

    axs[2].imshow(best_output, cmap='hot')
    axs[2].set_title(f"Model Output\nExc={best_exc}, Inh={best_inh}, AUC={best_auc:.4f}")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
