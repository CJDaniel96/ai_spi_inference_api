import os
import cv2
import argparse
from ultralytics import YOLO
from itertools import combinations

def calculate_distance(box1, box2):
    """Calculates the distance between the closest borders of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate horizontal and vertical distances
    dx = max(0, x1_min - x2_max, x2_min - x1_max)
    dy = max(0, y1_min - y2_max, y2_min - y1_max)

    # Determine points for drawing the line
    if dx > dy:  # Horizontal distance is greater
        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)
        
        if overlap_y_min < overlap_y_max:
            y_coord = (overlap_y_min + overlap_y_max) / 2
        else:
            y_coord = (y1_min + y1_max + y2_min + y2_max) / 4

        if x1_min > x2_max:
            start_point = (int(x1_min), int(y_coord))
            end_point = (int(x2_max), int(y_coord))
        else:
            start_point = (int(x1_max), int(y_coord))
            end_point = (int(x2_min), int(y_coord))
    else:  # Vertical distance is greater
        overlap_x_min = max(x1_min, x2_min)
        overlap_x_max = min(x1_max, x2_max)

        if overlap_x_min < overlap_x_max:
            x_coord = (overlap_x_min + overlap_x_max) / 2
        else:
            x_coord = (x1_min + x1_max + x2_min + x2_max) / 4

        if y1_min > y2_max:
            start_point = (int(x_coord), int(y1_min))
            end_point = (int(x_coord), int(y2_max))
        else:
            start_point = (int(x_coord), int(y1_max))
            end_point = (int(x_coord), int(y2_min))
            
    return dx, dy, start_point, end_point

def calculate_pads_distance(image_path, center_model, pad_model, conf_threshold, closeness_threshold=2, show_image=False, save_image=True):
    distance_list = []
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape
    img_center_x, img_center_y = img_w / 2, img_h / 2

    # Perform inference with center_model to find the center paste
    center_results = center_model(image, conf=conf_threshold)
    center_boxes = center_results[0].boxes.xyxy.cpu().numpy()

    if len(center_boxes) == 0:
        print("No center paste found.")
        if show_image:
            # show image and wait
            image = cv2.resize(image, (256, 256))
            cv2.imshow("Pads Distance", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None

    # Find the box closest to the image center as the center paste
    center_box = min(center_boxes, key=lambda box: (
        abs((box[0] + box[2]) / 2 - img_center_x) +
        abs((box[1] + box[3]) / 2 - img_center_y)
    ))
    
    # Draw center box
    x1, y1, x2, y2 = map(int, center_box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red for center paste

    # Perform inference with pad_model to find surrounding pads
    pad_results = pad_model(image, conf=conf_threshold)
    pad_boxes = pad_results[0].boxes.xyxy.cpu().numpy()

    # Calculate and draw distances
    for box in pad_boxes:
        dx, dy, start_point, end_point = calculate_distance(center_box, box)
        distance = (dx**2 + dy**2)**0.5

        # If the distance is less than the threshold, skip
        if distance < closeness_threshold:
            continue

        # Draw pad boxes
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1) # Blue for surrounding pads
        
        # Only draw a line if the boxes do not overlap
        if dx > 0 or dy > 0:
            distance_list.append(distance)
            cv2.line(image, start_point, end_point, (0, 255, 0), 1)
            text_pos = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            cv2.putText(image, f"{distance:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if save_image:
        output_dir = r"D:\Dre\SPI_pad_paste\output_results\infer_distance"
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, image_name)
        cv2.imwrite(save_path, image)

    if show_image:
        # Show the image with bounding boxes and distances
        image = cv2.resize(image, (256, 256))
        cv2.imshow("Pads Distance", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not distance_list:
        return None
    try:
        min_distance = min(distance_list)
        return float(f"{min_distance:.2f}")
    except Exception:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distances between solder pastes.")
    parser.add_argument("--image_path", type=str, default=r"D:\Dre\SPI_pad_paste\output_results\for_ME\20250523135357\19_3663.jpg", help="Path to the image file.")
    # parser.add_argument("--center_model_path", type=str, default=r"D:\Dre\SPI_pad_paste\yolo_model\scripts\runs\detect\paste_model\weights\best.pt", help="Path to the center model weights.")
    # parser.add_argument("--pad_model_path", type=str, default=r"D:\Dre\SPI_pad_paste\yolo_model\models\gap_model\0714\weights\best.pt", help="Path to the pad model weights.")
    parser.add_argument("--center_model_path", type=str, default=r"D:\Dre\JQ_SPI_02_AI_API\models\distance\center.pt", help="Path to the center model weights.")
    parser.add_argument("--pad_model_path", type=str, default=r"D:\Dre\JQ_SPI_02_AI_API\models\distance\pad.pt", help="Path to the pad model weights.")
    parser.add_argument("--conf_threshold", type=float, default=0.6, help="Confidence threshold for YOLO models.")
    parser.add_argument("--show_image", action="store_true", help="Show image with distances using OpenCV.")
    parser.add_argument("--save_image", action="store_true", help="Save image with distances to output folder.")
    args = parser.parse_args()
    
    center_model = YOLO(args.center_model_path)
    pad_model = YOLO(args.pad_model_path)
    
    print(calculate_pads_distance(
        image_path=args.image_path, 
        center_model=center_model, 
        pad_model=pad_model, 
        conf_threshold=args.conf_threshold, 
        show_image=args.show_image,
        save_image=args.save_image
    ))