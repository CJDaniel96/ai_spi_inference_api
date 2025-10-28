import requests
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd


class PasteDetectionTrigger:
    def __init__(self, server_url: str = "http://127.0.0.1:8001"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()

    def check_server_health(self) -> bool:
        """Check if the paste detection server is healthy and ready"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("model_ready", False)
            return False
        except requests.RequestException as e:
            print(f"Server health check failed: {e}")
            return False

    def wait_for_server(self, max_wait_time: int = 60):
        """Wait for server to be ready"""
        print("Checking paste detection server health...")
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            if self.check_server_health():
                print("Paste detection server is ready!")
                return True
            print("Server not ready, waiting...")
            time.sleep(2)

        print(f"Server did not become ready within {max_wait_time} seconds")
        return False

    def trigger_inference(self, job_folder: str, image_extensions: Optional[List[str]] = None,
                         save_results: bool = True, output_file: Optional[str] = None) -> dict:
        """Trigger paste detection inference on the specified job folder"""

        # Validate job folder exists
        job_path = Path(job_folder)
        if not job_path.exists():
            raise FileNotFoundError(f"Job folder not found: {job_folder}")

        # Prepare request payload
        payload = {
            "job_folder": str(job_path.absolute()),
            "image_extensions": image_extensions
        }

        print(f"Triggering paste detection inference for: {job_folder}")
        print(f"Server: {self.server_url}")

        try:
            # Send inference request
            start_time = time.time()
            response = self.session.post(
                f"{self.server_url}/inference",
                json=payload,
                timeout=600  # 10 minute timeout for large batches
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()

                print(f"\nPaste detection inference completed successfully!")
                print(f"Total images processed: {result['total_images']}")
                print(f"Total inference time: {result['total_inference_time']:.2f} seconds")
                print(f"Request time: {end_time - start_time:.2f} seconds")

                # Calculate statistics
                if result['results']:
                    total_paste_pixels = sum(r['paste_pixels'] for r in result['results'])
                    avg_paste_pixels = total_paste_pixels / len(result['results'])
                    images_with_paste = sum(1 for r in result['results'] if r['paste_pixels'] > 0)

                    print(f"Images with paste detected: {images_with_paste}/{result['total_images']}")
                    print(f"Total paste pixels: {total_paste_pixels}")
                    print(f"Average paste pixels per image: {avg_paste_pixels:.1f}")

                # Save results if requested
                if save_results and result['results']:
                    if output_file is None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_file = f"paste_detection_results_{timestamp}.csv"

                    # Flatten bbox data for CSV
                    csv_data = []
                    for r in result['results']:
                        csv_data.append({
                            'image_path': r['image_path'],
                            'paste_pixels': r['paste_pixels'],
                            'bbox_x1': r['bbox'][0],
                            'bbox_y1': r['bbox'][1],
                            'bbox_x2': r['bbox'][2],
                            'bbox_y2': r['bbox'][3],
                            'inference_time': r['inference_time']
                        })

                    df = pd.DataFrame(csv_data)
                    df.to_csv(output_file, index=False)
                    print(f"Results saved to: {output_file}")

                return result

            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type') == 'application/json' else response.text
                raise Exception(f"Server error ({response.status_code}): {error_detail}")

        except requests.Timeout:
            raise Exception("Request timed out - inference may be taking too long")
        except requests.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def get_server_info(self) -> dict:
        """Get paste detection server information"""
        try:
            response = self.session.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Server responded with status {response.status_code}"}
        except requests.RequestException as e:
            return {"error": f"Failed to connect to server: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Trigger Paste Detection Inference")
    parser.add_argument("job_folder", help="Path to job folder containing images")
    parser.add_argument("--server-url", default="http://127.0.0.1:8001",
                       help="URL of the paste detection server (default: http://127.0.0.1:8001)")
    parser.add_argument("--extensions", nargs="+",
                       help="Image file extensions to process (e.g., .jpg .png)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to CSV file")
    parser.add_argument("--output-file",
                       help="Output CSV file path (default: auto-generated)")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for server to be ready before sending request")
    parser.add_argument("--max-wait", type=int, default=60,
                       help="Maximum time to wait for server (seconds)")

    args = parser.parse_args()

    # Create trigger client
    trigger = PasteDetectionTrigger(args.server_url)

    try:
        # Wait for server if requested
        if args.wait:
            if not trigger.wait_for_server(args.max_wait):
                print("Exiting: Paste detection server not ready")
                return 1

        # Trigger inference
        result = trigger.trigger_inference(
            job_folder=args.job_folder,
            image_extensions=args.extensions,
            save_results=not args.no_save,
            output_file=args.output_file
        )

        print("\nPaste detection inference completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())