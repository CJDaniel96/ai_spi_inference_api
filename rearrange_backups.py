import os
import shutil
from pathlib import Path

def rearrange_backups():
    # Base directory containing the timestamped folders
    base_path = Path(r"D:\spi_ai\backup\02\csv")

    if not base_path.exists():
        print(f"Error: {base_path} does not exist.")
        return

    print(f"Scanning {base_path}...")

    # Iterate through all items in the base path
    for folder in base_path.iterdir():
        # Only process directories that look like timestamps (14 digits)
        # and are not the year folders we might have already created (4 digits)
        if folder.is_dir() and len(folder.name) == 14 and folder.name.isdigit():
            year = folder.name[:4]
            month = folder.name[4:6]
            
            # Define target path: base/year/month/folder_name
            target_parent = base_path / year / month
            target_path = target_parent / folder.name
            
            # Create year/month structure if it doesn't exist
            target_parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Moving {folder.name} -> {year}\{month}")
            
            try:
                # Move the folder
                shutil.move(str(folder), str(target_path))
            except Exception as e:
                print(f"  Error moving {folder.name}: {e}")

    print("Rearrangement complete.")

if __name__ == "__main__":
    rearrange_backups()
