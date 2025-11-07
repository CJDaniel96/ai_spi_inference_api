import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import distance_detection_server
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from distance_detection_server import DistanceDetectionInference
except Exception as e:
    print("Failed to import DistanceDetectionInference from distance_detection_server.py:\n", e)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Use DistanceDetectionInference.process_image() to compute min center→pad distance for one image.")
    parser.add_argument("--image", type=str, default=r"D:\Dre\JQ_SPI_02_AI_API\data\20250523135357\6_2736.jpg", help="Path to input image")
    parser.add_argument("--center-model", type=str, default="D:/Dre/JQ_SPI_02_AI_API/models/distance/center.pt", help="Path to center model .pt")
    parser.add_argument("--pad-model", type=str, default="D:/Dre/JQ_SPI_02_AI_API/models/distance/pad.pt", help="Path to pad model .pt")
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Confidence threshold for YOLO models")
    parser.add_argument("--closeness-threshold", type=float, default=2.0, help="Minimum edge distance to draw/consider (px)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save annotated image")
    parser.add_argument("--no-show", action="store_true", help="Do not display the window; useful on headless runs")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if cv2 is None and (args.save or not args.no_show):
        print("OpenCV (cv2) is needed for visualization. Install opencv-python or use --no-show without --save.")
        sys.exit(1)

    # Initialize inference engine and load models
    engine = DistanceDetectionInference(center_model_path=args.center_model, pad_model_path=args.pad_model,
                                       conf_threshold=args.conf_threshold, closeness_threshold=args.closeness_threshold)
    try:
        engine.load()
    except Exception as e:
        print("Failed to load models:", e)
        sys.exit(1)

    # Use the server engine's process_image to compute min distance
    result = engine.process_image(image_path)
    min_d = result.get("min_center_to_pad_distance")
    print(f"Min center->pad distance: {min_d if min_d is not None else 'None'}")

    # Optional visualization: replicate drawing style from center_paste_dis_img_infer.py
    if (args.save or not args.no_show) and cv2 is not None:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to read image for visualization: {image_path}")
        else:
            img_h, img_w, _ = img.shape
            img_center_x, img_center_y = img_w / 2, img_h / 2

            # Run detections for visualization
            try:
                center_results = engine.center_model(img, conf=args.conf_threshold)  # type: ignore[attr-defined]
                c_boxes_tensor = center_results[0].boxes
                center_boxes = c_boxes_tensor.xyxy.cpu().numpy() if (c_boxes_tensor is not None and c_boxes_tensor.xyxy is not None) else []
            except Exception as e:
                print("Failed to run center model (viz):", e)
                center_boxes = []

            if len(center_boxes) == 0:
                print("No center paste found for visualization.")
            else:
                # Choose the bbox closest to image center
                center_box = min(center_boxes, key=lambda box: (
                    abs((box[0] + box[2]) / 2 - img_center_x) +
                    abs((box[1] + box[3]) / 2 - img_center_y)
                ))
                # Draw center box (red)
                x1, y1, x2, y2 = map(int, center_box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # Pads
                try:
                    pad_results = engine.pad_model(img, conf=args.conf_threshold)  # type: ignore[attr-defined]
                    p_boxes_tensor = pad_results[0].boxes
                    pad_boxes = p_boxes_tensor.xyxy.cpu().numpy() if (p_boxes_tensor is not None and p_boxes_tensor.xyxy is not None) else []
                except Exception as e:
                    print("Failed to run pad model (viz):", e)
                    pad_boxes = []

                # Draw pads and distance lines like reference script
                for box in pad_boxes:
                    dx, dy, start_point, end_point = DistanceDetectionInference.calculate_distance(center_box, box)
                    distance = (dx**2 + dy**2) ** 0.5

                    # Skip too-close distances
                    if distance < args.closeness_threshold:
                        continue

                    # Draw pad box (blue)
                    px1, py1, px2, py2 = map(int, box)
                    cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 1)

                    # Draw line only if non-overlapping on at least one axis
                    if dx > 0 or dy > 0:
                        cv2.line(img, start_point, end_point, (0, 255, 0), 1)
                        text_pos = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                        cv2.putText(img, f"{distance:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Save/show
            if args.save:
                out_path = Path(args.save)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(out_path), img)
                if not ok:
                    print(f"Failed to save annotated image to {out_path}")
                else:
                    print(f"Saved annotated image to {out_path}")

            if not args.no_show:
                disp = cv2.resize(img, (256, 256))
                cv2.imshow("Pads Distance", disp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
