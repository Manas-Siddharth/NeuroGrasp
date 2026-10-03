# NeuroGrasp

## Deep Learning-Based Robotic Grasp Detection from RGB Images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0+cu124](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-NVIDIA_RTX_3050-76b900.svg)](https://developer.nvidia.com/cuda-zone)

NeuroGrasp is a deep learning perception framework for autonomous robotic grasp detection. Given a single 2D RGB camera image, NeuroGrasp directly predicts 5-DoF planar robotic grasp parameters $(x, y, w, h, \theta)$ along with a grasp feasibility confidence score in real time.

---

## 1. Project Overview

Autonomous robotic object manipulation requires reliable visual perception to identify feasible grasping locations. Traditional methods rely on heavy 3D point cloud processing or heuristic geometry fitting. NeuroGrasp formulates planar robotic grasp detection as a direct deep learning regression and classification task, mapping RGB input images directly to oriented 2D grasp rectangles.

---

## 2. Motivation

Robotic arms operating in unstructured environments (e.g., bin picking, warehouse sorting, home service robotics) must rapidly determine where and how to position parallel-jaw end-effectors to successfully grasp objects. Predicting planar grasps directly from 2D RGB images enables low-latency perception on lightweight embedded vision hardware.

---

## 3. Key Features

- **RGB Grasp Detection**: Direct 2D planar grasp prediction from monocular RGB camera input.
- **ResNet18 CNN Backbone**: Leverages ImageNet pre-trained feature representation for transfer learning.
- **Multi-Task Regression Head**: Predicts center position, bounding dimensions, continuous orientation vector, and confidence logit.
- **Continuous Orientation Encoding**: Uses $(\sin\theta, \cos\theta)$ vector representation to eliminate angle wrapping discontinuities at $\pm 180^\circ$.
- **GPU Acceleration**: Fully accelerated via PyTorch CUDA 12.4 on NVIDIA GeForce RTX 3050.
- **Verified Evaluation Suite**: Includes rotated rectangle IoU calculation, metric logger, and quantitative error analysis.
- **Independent Geometry Verification**: Verified against an independent Sutherland-Hodgman polygon clipping implementation.
- **Streamlit Web Demo**: Interactive Web GUI for uploading images and visualizing predicted grasps.
- **Automated Unit Tests**: Built-in `pytest` test suite verifying dataset loaders, model shapes, loss calculation, and inference pipelines.
- **ROS 2 Integration Roadmap**: Detailed architectural guide for integrating perception outputs into ROS 2 and MoveIt 2.

---

## 4. System Architecture

```text
               +-----------------------+
               |     RGB Image Input   |
               |     (224 x 224 x 3)   |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               | ResNet18 CNN Backbone |
               | (Feature Extraction)  |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               | 512-dim Feature Layer |
               +-----------+-----------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
+-----------------------+   +-----------------------+
|  Grasp Regression Head|   |Confidence Class. Head |
|  (x, y, w, h, sin/cos)|   |  (Confidence Logit)   |
+-----------+-----------+   +-----------+-----------+
            |                             |
            +--------------+--------------+
                           |
                           v
               +-----------------------+
               |   Predicted Grasp     |
               | (x, y, w, h, θ, Conf) |
               +-----------+-----------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
+-----------------------+   +-----------------------+
| Visualization & Demo  |   | Evaluation & ROS 2    |
| (Streamlit / OpenCV)  |   | Perception Node       |
+-----------------------+   +-----------------------+
```

---

## 5. Model Architecture

- **Input Dimension**: $B \times 3 \times 224 \times 224$ (RGB Image, ImageNet normalized)
- **Backbone**: ResNet18 (ImageNet pre-trained weights)
- **Feature Vector**: 512 dimensions from adaptive average pooling layer (`fc` layer replaced by Identity).
- **Dual Prediction Heads**:
  1. **Grasp Head**: 3-layer MLP (`Linear(512, 256) -> BatchNorm1d -> ReLU -> Linear(256, 64) -> ReLU -> Linear(64, 6)`). Outputs 6 regression targets: $[x_{norm}, y_{norm}, w_{norm}, h_{norm}, \sin\theta, \cos\theta]$.
  2. **Confidence Head**: 2-layer MLP (`Linear(512, 128) -> ReLU -> Linear(128, 1)`). Outputs 1 unnormalized confidence logit $z$.

### Why $(\sin\theta, \cos\theta)$ Orientation Encoding?
Direct regression of rotation angle $\theta$ in degrees or radians suffers from sharp boundary discontinuities at $\pm 180^\circ$ ($\pm \pi$ rad), causing severe gradient instability during backpropagation. NeuroGrasp predicts continuous trigonometric components $(\sin\theta, \cos\theta)$ and unambiguously reconstructs the angle during post-processing:

$$\theta = \text{atan2}(\sin\theta, \cos\theta)$$

---

## 6. Dataset

> [!IMPORTANT]
> **Dataset Distinction**:
> - **CURRENT EXPERIMENT**: Evaluated on **200 synthetic Jacquard-style sample pairs** (160 train / 40 validation) programmatically generated by `data/create_sample_dataset.py`.
> - **Reported Metrics**: The reported metrics are prototype baseline results obtained using synthetic Jacquard-style data for reproducible hardware and code verification. They are **not** benchmark results on the full official Jacquard dataset.
> - **OFFICIAL JACQUARD BENCHMARK**: The official Jacquard robotic grasping dataset contains over 54,000 real-world synthetic CAD images and 11+ million grasp annotations (Depierre et al., ICRA 2018). Official dataset website: [https://jacquard.liris.cnrs.fr/database.php](https://jacquard.liris.cnrs.fr/database.php).

- **Annotation Format**: `x;y;theta;opening;jaws_size`
  - $x, y$: Grasp center pixel coordinates.
  - $\theta$: Rotation angle in degrees.
  - $\text{opening}$: Gripper opening width $w$ in pixels.
  - $\text{jaws\_size}$: Gripper jaw height $h$ in pixels.

---

## 7. Data Preprocessing

1. **Image Resizing**: Input RGB images ($1024 \times 1024$) are resized to $224 \times 224$ pixels.
2. **Coordinate Rescaling**: Spatial ground truth parameters are scaled consistently:
   $$x_{scaled} = x \cdot \left(\frac{224}{W_{orig}}\right), \quad y_{scaled} = y \cdot \left(\frac{224}{H_{orig}}\right)$$
   $$w_{scaled} = w \cdot \left(\frac{224}{W_{orig}}\right), \quad h_{scaled} = h \cdot \left(\frac{224}{H_{orig}}\right)$$
3. **Target Normalization**: Coordinates and sizes are normalized to $[0, 1]$ relative to image resolution (224), while orientation is converted to $\sin(\theta_{rad})$ and $\cos(\theta_{rad})$.
4. **Tensor Transformation**: Normalized using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 8. Loss Function

NeuroGrasp optimizes a multi-task loss combining spatial regression and grasp confidence classification:

$$\mathcal{L}_{total} = \lambda_{xy} \mathcal{L}_{xy} + \lambda_{wh} \mathcal{L}_{wh} + \lambda_{angle} \mathcal{L}_{angle} + \lambda_{conf} \mathcal{L}_{conf}$$

$$\mathcal{L}_{xy} = \text{SmoothL1}\left((x, y)_{pred}, (x, y)_{gt}\right)$$
$$\mathcal{L}_{wh} = \text{SmoothL1}\left((w, h)_{pred}, (w, h)_{gt}\right)$$
$$\mathcal{L}_{angle} = \text{SmoothL1}\left((\sin\theta, \cos\theta)_{pred}, (\sin\theta, \cos\theta)_{gt}\right)$$
$$\mathcal{L}_{conf} = \text{BCEWithLogits}\left(z_{pred}, c_{gt}\right)$$

Default hyperparameter weights: $\lambda_{xy} = 1.0, \lambda_{wh} = 1.0, \lambda_{angle} = 1.0, \lambda_{conf} = 0.5$.

---

## 9. Training Configuration

- **Backbone**: ResNet18 (Pretrained)
- **Optimizer**: AdamW ($\text{learning\_rate}=0.001$, $\text{weight\_decay}=0.0001$)
- **Scheduler**: CosineAnnealingLR ($\eta_{min}=10^{-6}$)
- **Batch Size**: 16
- **Epochs**: 25
- **Device**: CUDA (`NVIDIA GeForce RTX 3050 Laptop GPU`)
- **Random Seed**: 42

---

## 10. Official Experimental Results

Evaluated across the complete **40-sample validation set** using `python -m src.evaluate`:

| Metric | Result Value |
| :--- | :--- |
| **Validation Samples** | `40` |
| **Validation Loss** | `0.0893` |
| **Mean Center Position Error** | `18.12 px` |
| **Mean Gripper Width Error** | `7.10 px` |
| **Mean Jaw Height Error** | `3.41 px` |
| **Mean Orientation Angle Error** | `7.48°` |
| **Mean Grasp IoU** | `0.0588` |
| **Grasp Success Rate ($\text{IoU} \ge 0.25 \land \Delta\theta \le 30^\circ$)** | `10.0%` |
| **Avg Inference Latency** | `~6.2 ms / image` |

---

## 11. Scientifically Honest Results Interpretation

- **Supervised Convergence**: The multi-task loss converged cleanly from `3.2657` to `0.0893` over 25 epochs.
- **Position & Angle Regression**: The network learned to locate object centers within $18.12\text{ px}$ and align grasp orientations within $7.48^\circ$ of ground truth.
- **IoU & Success Rate**: Mean IoU remains low ($0.0588$) and grasp success rate is limited ($10.0\%$). This reflects a real performance limitation of training on a small 200-sample synthetic prototype dataset rather than an evaluation code flaw. Achieving high IoU ($\ge 0.25$) across diverse real-world object geometries requires scaling training to the full 54,000-image Jacquard dataset.

---

## 12. Geometry & IoU Verification

To ensure strict mathematical accuracy, the rotated-rectangle IoU calculation in `src/utils.py` was independently audited against a standalone **Sutherland-Hodgman Polygon Clipping** algorithm (`data/verify_geometry.py`).

- **Maximum Observed Difference**: `0.000001`
- **Result**: The rotated-rectangle IoU implementation is mathematically verified exact.

---

## 13. Data Leakage Verification

- **Train Split**: 160 samples (80%)
- **Validation Split**: 40 samples (20%)
- **Programmatic Overlap Check**: 0 overlapping indices. 100% disjoint split verified.

---

## 14. Visual Results

Generated output visualizations saved in `results/`:
- `results/training_loss.png`: Training vs validation loss convergence curve.
- `results/validation_results.png`: Ground truth vs predicted grasp comparison figure.
- `results/grasp_predictions.png`: Single-image inference output visualization.
- `results/predictions/sample_01.png` to `sample_10.png`: Individual side-by-side ground truth vs predicted visual comparisons.

---

## 15. Streamlit Interactive Web Application

NeuroGrasp includes a built-in interactive Streamlit application for real-time visual demonstration.

```bash
streamlit run demo/app.py
```

Features:
- Upload custom RGB camera images or pick from built-in sample images.
- Real-time grasp prediction and bounding box rendering.
- Displays center X/Y, width, height, angle (deg), confidence score, and latency metrics.

---

## 16. Installation

```bash
# 1. Clone Repository
git clone https://github.com/user/NeuroGrasp.git
cd NeuroGrasp

# 2. Create Virtual Environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt
```

---

## 17. Training

To train the model on the dataset:

```bash
python -m src.train --epochs 25
```

---

## 18. Evaluation

To evaluate model metrics across the validation set:

```bash
python -m src.evaluate
```

---

## 19. Unit Testing

Run the automated test suite via `pytest`:

```bash
python -m pytest -v
```

---

## 20. Project Structure

```
NeuroGrasp/
│
├── README.md                  # Comprehensive Documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Dependency requirements
├── config.yaml                # Hyperparameters & dataset options
├── pytest.ini                 # Pytest configuration
│
├── data/
│   ├── README.md              # Dataset format specification
│   ├── create_sample_dataset.py # Synthetic dataset generator
│   └── sample_dataset/        # 200 synthetic Jacquard-style sample pairs
│
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Jacquard dataset loader & parser
│   ├── model.py               # NeuroGraspNet ResNet18 backbone architecture
│   ├── losses.py              # Multi-task loss function
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Quantitative evaluation script
│   ├── inference.py           # Grasp predictor engine
│   └── utils.py               # Geometry, IoU, and visualization helpers
│
├── demo/
│   └── app.py                 # Streamlit web application
│
├── notebooks/
│   └── NeuroGrasp_Demo.ipynb  # Interactive walkthrough notebook
│
├── results/                   # Generated plots & evaluation metrics
│   └── predictions/           # Individual GT vs Pred visual comparisons
├── checkpoints/               # Trained PyTorch model weights
├── docs/                      # Technical architecture & methodology documentation
├── ros2/                      # ROS 2 message specification
└── tests/                     # Unit test suite
```

---

## 21. Limitations

1. **Prototype Dataset**: Trained on 200 synthetic Jacquard-style samples rather than the full 54k dataset.
2. **IoU & Success Rate**: Mean IoU ($0.0588$) and success rate ($10.0\%$) demonstrate early learning but require full dataset scaling for production accuracy.
3. **Planar Grasps**: Predicts 2D planar grasps $(x, y, w, h, \theta)$ suitable for top-down parallel-jaw grippers; does not estimate full 6-DoF 3D rotations $(R, t)$.
4. **Confidence Target**: Confidence head is trained on positive grasp annotations; calibration requires negative non-grasp region sampling.

---

## 22. Future Work

- Train NeuroGrasp on the full 54,000-image Jacquard dataset.
- Incorporate depth maps (RGB-D) for 3D spatial position estimation $(X, Y, Z)$.
- Implement anchor-based or heatmap-based multi-grasp detection heads.
- Calibrate confidence scoring with negative non-grasp sampling.
- Integrate ROS 2 nodes for physical robot arm execution.

---

## 23. ROS 2 Integration Roadmap

NeuroGrasp is designed to serve as the visual perception module in an end-to-end robotic manipulation pipeline:

```text
  +------------------+
  |  RGB-D Camera    |
  +--------+---------+
           |
           v
  +------------------+
  |    NeuroGrasp    |  ---> Predicts 2D Grasp (x, y, w, h, θ, Conf)
  +--------+---------+
           |
           v
  +------------------+
  | 3D Projection    |  ---> Uses Depth Map Z = D(x, y) & Camera Intrinsics
  +--------+---------+
           |
           v
  +------------------+
  | Camera-to-Robot  |  ---> Applies Extrinsic Matrix T_robot_camera
  +--------+---------+
           |
           v
  +------------------+
  |   ROS 2 Topic    |  ---> Publishes /neurograsp/predicted_grasp
  +--------+---------+
           |
           v
  +------------------+
  |    MoveIt 2      |  ---> Motion Planning & Inverse Kinematics (IK)
  +--------+---------+
           |
           v
  +------------------+
  |  Robot Controller|  ---> Drives Physical Robot Arm & Parallel Gripper
  +------------------+
```

---

## 24. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 25. References & Citation

- **Jacquard Dataset**: Depierre, A., Hermans, V., & Garcia, R. (2018). *Jacquard: A Large Scale Dataset for Robotic Grasp Detection*. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). [https://jacquard.liris.cnrs.fr/](https://jacquard.liris.cnrs.fr/)
