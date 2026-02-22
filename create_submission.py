import zipfile
import os

def create_submission():
    base_dir = r'c:\Users\Sadak\Desktop\dobaseline'
    files_to_zip = [
        'solution.py',
        'model.joblib',
        'requirements.txt'
    ]
    
    zip_name = 'submission.zip'
    
    print(f"Creating {zip_name}...")
    
    with zipfile.ZipFile(os.path.join(base_dir, zip_name), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in files_to_zip:
            file_path = os.path.join(base_dir, file_name)
            if os.path.exists(file_path):
                # The arcname parameter ensures the file is at the root of the zip archive
                zipf.write(file_path, arcname=file_name)
                print(f"Added {file_name} to archive.")
            else:
                print(f"ERROR: {file_name} not found! Cannot add to archive.")
                
    # Check the final size
    final_size_mb = os.path.getsize(os.path.join(base_dir, zip_name)) / (1024 * 1024)
    print(f"\nSuccessfully created {zip_name}!")
    print(f"Final Size: {final_size_mb:.2f} MB")

if __name__ == "__main__":
    create_submission()
