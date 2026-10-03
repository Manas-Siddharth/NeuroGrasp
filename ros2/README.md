# ROS 2 Integration & Future Robotic Execution Architecture

## 1. System Pipeline Overview

NeuroGrasp functions as an autonomous **robotic perception module**.

### Current Perception Pipeline (Implemented):
```mermaid
flowchart LR
    RGB["RGB Camera"] --> IMG["RGB Image"]
    IMG --> PP["Preprocessing"]
    PP --> CNN["NeuroGrasp CNN"]
    CNN --> PARAMS["Grasp Parameters (x,y,w,h,θ,Conf)"]
    PARAMS --> VIS["Visualization / Demo App"]
```

### Future Hardware & ROS 2 Physical Execution Pipeline:
```mermaid
flowchart LR
    RGBD["RGB-D Camera"] --> NG["NeuroGrasp perception"]
    NG --> 3D["3D Grasp Pose Transformation\n(Pin-hole camera model + Depth)"]
    3D --> ROS2["ROS 2 Node\n(/neurograsp/predicted_grasp)"]
    ROS2 --> MOVEIT["MoveIt 2 / Motion Planner"]
    MOVEIT --> ARM["Robot Arm Controller\n(e.g., Franka / UR5)"]
    ARM --> PICK["Physical Gripper Execution"]
```

---

## 2. ROS 2 Custom Message Definition

To pass predicted grasp parameters across ROS 2 nodes, define `neurograsp_msgs/msg/GraspPrediction.msg`:

```text
# GraspPrediction.msg
std_msgs/Header header

float32 x              # Pixel center X
float32 y              # Pixel center Y
float32 width          # Gripper opening width (pixels or meters)
float32 height         # Jaw thickness (pixels or meters)
float32 angle_rad      # Grasp angle (radians)
float32 confidence     # Confidence score [0.0 - 1.0]

# Projected 3D Camera Frame Coordinates (optional when depth available)
geometry_msgs/Point position_3d
geometry_msgs/Quaternion orientation_3d
```

---

## 3. 2D to 3D Coordinate Projection Math

Given depth value $Z = D(x, y)$ at the predicted grasp center $(x, y)$ and camera intrinsic parameters $(f_x, f_y, c_x, c_y)$:

$$X_c = \frac{(x - c_x) \cdot Z}{f_x}$$
$$Y_c = \frac{(y - c_y) \cdot Z}{f_y}$$
$$Z_c = Z$$

The 3D grasp pose in camera frame $\mathbf{P}_c = [X_c, Y_c, Z_c]^T$ is transformed into the robot arm base coordinate frame $\mathbf{P}_r$ via extrinsic matrix $T_{robot}^{camera}$:

$$\mathbf{P}_r = T_{robot}^{camera} \cdot \mathbf{P}_c$$
