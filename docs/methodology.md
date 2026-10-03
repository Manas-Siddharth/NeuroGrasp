# NeuroGrasp Methodology & Mathematical Principles

## 1. Planar Grasp Representation

A 2D planar robotic grasp is defined by 5 geometric parameters:

$$G = (x, y, w, h, \theta)$$

Where:
- $(x, y)$: Pixel coordinates of the grasp center point.
- $w$: Gripper opening width (distance between parallel jaws in pixels).
- $h$: Gripper jaw height / depth (thickness of jaws in pixels).
- $\theta$: Rotation angle of the grasp relative to the horizontal image axis.

---

## 2. Continuous Orientation Representation

Direct regression of rotation angle $\theta$ suffers from discontinuities at $\pm \pi$ radians ($\pm 180^\circ$). To prevent angle wrapping issues during gradient backpropagation, NeuroGrasp decomposes angle $\theta$ into continuous trigonometric components:

$$\mathbf{v}_{\theta} = \begin{bmatrix} \sin\theta \\ \cos\theta \end{bmatrix}$$

The angle is recovered during post-processing via:

$$\theta = \text{atan2}(\sin\theta, \cos\theta)$$

---

## 3. Multi-Task Loss Formulation

The loss function optimizes center location, dimensions, orientation, and grasp validity simultaneously:

$$\mathcal{L}_{total} = \lambda_{xy} \mathcal{L}_{xy} + \lambda_{wh} \mathcal{L}_{wh} + \lambda_{angle} \mathcal{L}_{angle} + \lambda_{conf} \mathcal{L}_{conf}$$

Where:
- $\mathcal{L}_{xy} = \text{SmoothL1}((x, y)_{pred}, (x, y)_{gt})$
- $\mathcal{L}_{wh} = \text{SmoothL1}((w, h)_{pred}, (w, h)_{gt})$
- $\mathcal{L}_{angle} = \text{SmoothL1}((\sin\theta, \cos\theta)_{pred}, (\sin\theta, \cos\theta)_{gt})$
- $\mathcal{L}_{conf} = \text{BCEWithLogits}(z_{pred}, c_{gt})$

Default hyperparameter loss weights: $\lambda_{xy} = 1.0, \lambda_{wh} = 1.0, \lambda_{angle} = 1.0, \lambda_{conf} = 0.5$.

---

## 4. Evaluation Criteria & Grasp Success Metric

A predicted grasp $\hat{G}$ is evaluated against ground truth $G^*$ using Intersection over Union (IoU) and Angular Error $\Delta \theta$:

1. **Jaccard Index / IoU**:
   $$\text{IoU}(\hat{G}, G^*) = \frac{\text{Area}(\hat{G} \cap G^*)}{\text{Area}(\hat{G} \cup G^*)}$$

2. **Angular Error**:
   $$\Delta \theta = |\text{atan2}(\sin(\hat{\theta} - \theta^*), \cos(\hat{\theta} - \theta^*))|$$

3. **Grasp Success Criterion**:
   A grasp prediction is classified as **Successful** if and only if:
   $$\text{IoU}(\hat{G}, G^*) \ge 0.25 \quad \text{AND} \quad \Delta \theta \le 30^\circ$$
