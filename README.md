# Saccadic Eye Tracking  
Developed by Ayaan Iqbal & Mekhael Thaha

**Saccadic Eye Tracking** is a biologically inspired system designed to simulate bottom-up visual attention using spiking neural networks. The project models how saliency-driven stimuli activate neurons in a 2D grid to predict fixation targets, using real-world scenes from the MIT1003 dataset.

This project was developed for **SYDE 552 (Computational Neuroscience)** at the University of Waterloo, and explores how excitation-inhibition balance influences visual salience.

---

## Table of Contents  
- [Key Features and Tools](#key-features-and-tools)  
- [System Architecture](#system-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  

---

## Key Features and Tools  

| Feature                      | Description                                                                 | Technology/Tools                     |
|-----------------------------|-----------------------------------------------------------------------------|--------------------------------------|
| Visual Saliency Map Input   | Generates saliency maps using spectral residual method from natural images | OpenCV, NumPy                        |
| Spiking Neuron Simulation   | Models 400 LIF neurons with lateral inhibition in a 20×20 grid              | Brian2                               |
| Gaze Prediction Evaluation  | Evaluates accuracy using AUC and NSS against human fixation data            | Matplotlib, Custom Metric Functions  |

---

## System Architecture  

The model consists of three main modules:

### Preprocessing (OpenCV):  
- Generates static saliency maps using low-level visual features  
- Applies Gaussian blur and thresholding for contrast enhancement  

### Spiking Neural Network (Brian2):  
- Simulates LIF neurons with parameterized excitation and inhibition  
- Lateral inhibition ensures winner-take-all dynamics  
- Spike activity is recorded and normalized into saliency predictions  

### Evaluation (AUC/NSS Metrics):  
- Compares predicted saliency maps with ground truth fixation data from MIT1003  
- Outputs heatmaps for performance across excitation/inhibition values  

---

## Installation  

### Prerequisites  
- Python 3.8 or higher  
- `pip install brian2 opencv-python numpy matplotlib`

### Clone the Repository  
```bash
git clone https://github.com/AyaanIqbal/Saccadic-Eye-Tracking.git
cd Saccadic-Eye-Tracking
```

## Usage

### Run the Demo  
```bash
python demo.py
```
This script runs the full pipeline on a small sample of 5 random images from the MIT1003 dataset. It will:
- Generate saliency maps using OpenCV's Spectral Residual method
- Preprocess and normalize input for the spiking neural network
- Simulate spiking activity in a 20×20 grid of Leaky Integrate-and-Fire (LIF) neurons using Brian2
- Produce predicted saliency (fixation) maps
- Evaluate results against ground truth fixation data using AUC and NSS metrics
- Display all images, spike heatmaps, and metric scores for inspection

## Full Parameter Sweep
```bash
python heatmap_plotter.py
```
This script performs a parameter sweep to analyze model performance across different excitation and inhibition values. It will:
- Iterate through a predefined grid of excitation/inhibition parameter pairs
- Simulate the spiking neural network for each configuration
- Compute AUC scores for each combination using fixation data
- Generate and save a heatmap to visualize performance trends and identify optimal values

## Citation & Acknowledgments

This project was developed as part of a research initiative for the SYDE 552 – Computational Neuroscience course at the University of Waterloo. The research paper details the model design, biological motivation, and evaluation results.

> **SYDE 552 – Computational Neuroscience**  
> Instructor: **Prof. Terrence C. Stewart**, University of Waterloo  
> Project Title: _Modeling Visual Salience with Excitation-Inhibition Dynamics: A Bottom-Up Approach to Gaze Prediction_  
> Authors: Ayaan Iqbal, Mekhael Thaha

**Dataset**

- MIT1003 Eye-Tracking Dataset  
- Ground truth fixation maps from [MIT Saliency Benchmark](http://saliency.mit.edu/datasets.html)

