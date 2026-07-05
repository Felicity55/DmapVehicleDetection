# DmapVehicleDetection

A robust, PyTorch-based framework for vehicle detection, density estimation, and semantic segmentation in congested traffic environments. By combining regression-based density maps (utilizing both MCNN and CSRNet backbones) with explicit foreground semantic segmentation masks, this project accurately localizes, tracks, and counts vehicles even under heavy occlusion and highly dynamic lighting conditions.

* **Citation Link**
Soumi Das, Palaiahnakote Shivakumara, Umapada Pal, and Raghavendra Ramachandra. "Gaussian 
Kernels Based Network for Multiple License Plate Number Detection in Day-Night Images." International 
Conference on Document Analysis and Recognition (ICDAR), pp. 70–87. Springer, 2023.

---

## ## System Architecture & Key Features
<img width="462" height="173" alt="Proposed Architecture" src="https://github.com/user-attachments/assets/3fe5032c-a8ab-4ae1-99f2-ed7a8eb28ab1" />

* **Multi-Column CNN (MCNN)**: Utilizes varying filter sizes ($9 \times 9$, $7 \times 7$, $5 \times 5$) across multiple columns to capture multi-scale vehicle features, adapting smoothly to variations in vehicle sizes caused by perspective distortion.
* **Dilated CNN (CSRNet)**: Deploys dilated convolutional layers to expand the network's receptive field without losing spatial resolution, making it highly effective for dense and highly congested urban traffic zones.
* **Semantic Segmentation Masking**: Runs a parallel path generating foreground vs. background segmentation maps to classify pixel boundaries and suppress environmental noise.
* **Dual-Stream Feature Fusion**: Implements a dedicated post-processing pipeline (`Fusion.py` and `ORoperation.py`) that refines density estimates using binary segmentation boundaries to eliminate false positives.
* **Full-Resolution Image Training**: Processes raw, full-resolution traffic imagery directly without arbitrary patch cropping, preserving crucial global context and continuous structural geometry.

---

## ## Repository Directory Structure

```text
├── .vscode/                     # Editor-specific runtime configurations
├── logs/                        # Saved model checkpoints and Tensorboard logs
├── csrnet_model.py              # CSRNet backbone architecture implementation
├── mcnn_model.py                # MCNN backbone architecture implementation
├── my_dataloader.py            # Main dataset pipeline loader 
├── v_dataloader.py             # DataLoader variant tailored for video frame processing
├── gt_dmap.py                   # Continuous ground-truth density map generator
├── gt_segmentmap.py             # Ground-truth binary segmentation mask generator
├── k_nearest_gaussian_kernel.py # Geometry-adaptive Gaussian kernel calculator
├── KMeans.py                    # Unsupervised clustering utilities for vehicle groupings
├── segmentmap.py                # Core segmentation map building properties
├── Fusion.py                    # Dual-stream map fusion engine
├── ORoperation.py               # Bitwise mask operations for post-processing boundaries
├── train.py                     # Baseline model training execution script
├── vehicle_train.py             # Specialized vehicle network training pipeline
├── vehicle_test.py              # Evaluation script for calculating tracking errors
├── test.py / Dtest.py           # Quick testing and debugging verification scripts
├── vTest.py / compact_test.py   # Full video inference and tracking pipelines
├── hyper_param_conf.py          # Centralized configuration file for hyperparameters
├── counting_metrics.py          # Math evaluation metrics (MAE, MSE)
├── results.png                  # Visual output verification placeholder
└── filename.avi / output.avi    # Sample exported inference video files

```

---

## ## Installation & Environment Setup

### 1. Prerequisites

* Python $\ge$ 3.6
* PyTorch $\ge$ 1.0.0
* CUDA Capability (Highly recommended for deep learning training execution)

### 2. Clone the Repository

Clone this repository to your local workspace and navigate into the project directory:

```bash
git clone https://github.com/Felicity55/DmapVehicleDetection.git
cd DmapVehicleDetection

```

### 3. Dependencies

Install all required computational, visualization, and image processing libraries:

```bash
pip install torch torchvision visdom opencv-python scikit-learn matplotlib numpy

```

---

## ## Workflow Execution Pipeline

### Step 1: Ground-Truth Generation & Data Setup

Your vehicle dataset must contain raw images paired with point annotations ($x, y$ coordinate matrices marking vehicle centers).

1. Open `k_nearest_gaussian_kernel.py` or `gt_dmap.py` and modify the `root_path` variable to point to your local dataset directory.
2. Run the adaptive kernel script to generate ground-truth density maps:
```bash
python k_nearest_gaussian_kernel.py

```


3. Generate parallel foreground tracking boundaries by running the segmentation script:
```bash
python gt_segmentmap.py

```



### Step 2: Real-Time Visualization

This repository integrates with **Visdom** to log training curves, loss drops, and target-to-prediction map side-by-sides. Launch the monitoring server *before* training:

```bash
python -m visdom.server

```

Once active, open your web browser and navigate to `http://localhost:8097`.

### Step 3: Train the Model

Adjust parameters such as backbones, target paths, learning rates, and batch counts inside `hyper_param_conf.py`. Launch the primary vehicle training loop:

```bash
python vehicle_train.py

```

### Step 4: Run Quantitative Evaluation

Evaluate performance against your validation split. This tracks precise counting discrepancies via Mean Absolute Error (MAE) and Mean Squared Error (MSE):

```bash
python vehicle_test.py

```

### Step 5: Full Video Inference Tracking

To pass a raw frame-by-frame traffic video through the model, overlay density bounding maps dynamically, and export an annotated output clip:

```bash
python vTest.py

```

The evaluated results will be compiled and saved inside the project root as `output.avi` or `filename.avi`.

---

## ## Mathematical Overview

Ground-truth density maps are modeled by blurring each discrete vehicle center delta function $\delta(x - x_i)$ via a normalized Gaussian kernel:

$$G_{\sigma}(x) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

In highly packed configurations, the variance $\sigma$ is computed adaptively relative to the average distance $\bar{d}_i$ of its $k$-nearest vehicle neighbors:

$$\sigma_i = \beta \bar{d}_i$$

The total inferred vehicle count $N$ inside any target matrix is extracted seamlessly by integrating over the continuous surface plane of the predicted density map $P$:

$$N = \sum_{x=1}^{W} \sum_{y=1}^{H} P(x, y)$$

---

## ## Outputs
<img width="359" height="272" alt="Sample1_Output" src="https://github.com/user-attachments/assets/6b004bd2-e7cc-4526-90ef-958c9c608401" />
<img width="356" height="310" alt="Sample2_Output" src="https://github.com/user-attachments/assets/d3b3b717-c9e7-4db8-b8d6-8f2a7fbcb46f" />
<img width="227" height="285" alt="Sample3_Output" src="https://github.com/user-attachments/assets/c48c54f4-ec0c-4ec7-af4d-504678778a38" />

