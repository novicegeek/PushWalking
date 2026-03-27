import os

folder_path = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\22.Measurement Data\S05_20260210\Qualisys"  # Replace with your actual folder path

for filename in os.listdir(folder_path):
    if "_Processed" in filename:
        new_name = filename.replace("_Processed", "")
        
        # Build full paths for the rename operation
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_name}")