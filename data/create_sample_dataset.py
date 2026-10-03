import os
import math
import numpy as np
from PIL import Image, ImageDraw

def create_sample_dataset(output_dir="data/sample_dataset", num_samples=200):
    """
    Generate synthetic dataset of Jacquard-style RGB images and grasp annotations.
    Creates diverse geometric objects (rectangles, ellipses, L-shapes, T-shapes)
    with accurate ground truth planar grasps.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    for i in range(1, num_samples + 1):
        img_size = 1024
        # Random light background color
        bg_color = (
            int(np.random.randint(235, 255)),
            int(np.random.randint(235, 255)),
            int(np.random.randint(235, 255))
        )
        img = Image.new("RGB", (img_size, img_size), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Center location for object
        cx = float(np.random.uniform(250, 774))
        cy = float(np.random.uniform(250, 774))
        
        # Dimensions
        obj_w = float(np.random.uniform(120, 280))
        obj_h = float(np.random.uniform(60, 160))
        angle_deg = float(np.random.uniform(-85, 85))

        # Color for object
        color = (
            int(np.random.randint(30, 200)),
            int(np.random.randint(30, 200)),
            int(np.random.randint(50, 220))
        )

        rad = math.radians(angle_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # Shape type
        shape_type = np.random.choice(["rectangle", "ellipse", "l_shape"])

        if shape_type == "rectangle":
            rect_corners = [
                [-obj_w / 2, -obj_h / 2],
                [ obj_w / 2, -obj_h / 2],
                [ obj_w / 2,  obj_h / 2],
                [-obj_w / 2,  obj_h / 2]
            ]
            rotated_corners = []
            for x, y in rect_corners:
                rx = cx + x * cos_t - y * sin_t
                ry = cy + x * sin_t + y * cos_t
                rotated_corners.append((rx, ry))
            draw.polygon(rotated_corners, fill=color, outline=(20, 20, 20))

        elif shape_type == "ellipse":
            # Draw ellipse by polygon approximation
            pts = []
            for deg in range(0, 360, 10):
                r = math.radians(deg)
                x = (obj_w / 2) * math.cos(r)
                y = (obj_h / 2) * math.sin(r)
                rx = cx + x * cos_t - y * sin_t
                ry = cy + x * sin_t + y * cos_t
                pts.append((rx, ry))
            draw.polygon(pts, fill=color, outline=(20, 20, 20))

        else:  # L-shape
            l_corners = [
                [-obj_w / 2, -obj_h / 2],
                [ obj_w / 2, -obj_h / 2],
                [ obj_w / 2,  0],
                [ 0,          0],
                [ 0,          obj_h / 2],
                [-obj_w / 2,  obj_h / 2]
            ]
            rotated_corners = []
            for x, y in l_corners:
                rx = cx + x * cos_t - y * sin_t
                ry = cy + x * sin_t + y * cos_t
                rotated_corners.append((rx, ry))
            draw.polygon(rotated_corners, fill=color, outline=(20, 20, 20))

        # Save RGB image
        img_filename = f"1_{i}_RGB.png"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)

        # Generate corresponding grasp annotation file (_grasps.txt)
        # Format: x;y;theta;opening;jaws_size
        grasp_filename = f"1_{i}_grasps.txt"
        grasp_path = os.path.join(output_dir, grasp_filename)
        
        opening = obj_w + np.random.uniform(15, 35)
        jaws_size = float(np.random.uniform(25, 45))
        
        with open(grasp_path, "w", encoding="utf-8") as f:
            # Main optimal grasp at object center along orientation
            f.write(f"{cx:.1f};{cy:.1f};{angle_deg:.1f};{opening:.1f};{jaws_size:.1f}\n")
            # Additional valid grasp offsets
            f.write(f"{cx + 10.0*cos_t:.1f};{cy + 10.0*sin_t:.1f};{angle_deg:.1f};{opening:.1f};{jaws_size:.1f}\n")
            f.write(f"{cx - 10.0*cos_t:.1f};{cy - 10.0*sin_t:.1f};{angle_deg:.1f};{opening:.1f};{jaws_size:.1f}\n")

    print(f"[+] Generated {num_samples} Jacquard dataset samples in {output_dir}")

if __name__ == "__main__":
    create_sample_dataset()
