import os
import shutil
from pathlib import Path
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import imghdr
import magic

def get_file_type(file_path):
    """
    Get actual file type using magic numbers
    """
    try:
        mime = magic.Magic()
        file_type = mime.from_file(file_path)
        return file_type.lower()
    except Exception:
        return ""

def is_valid_image(file_path):
    """
    Check if file is a valid image by performing multiple checks:
    1. Valid image extension
    2. Magic number check for image type
    3. Can be opened by PIL
    4. Meets minimum size requirements
    5. Has valid image data
    Returns: 
        - (True, None) if valid image
        - (False, reason) if invalid image
    """
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        return False, "invalid_extension"

    # Check if file is actually an image using magic numbers
    file_type = get_file_type(file_path)
    if not any(img_type in file_type for img_type in ['image', 'jpeg', 'png', 'gif']):
        return False, "not_image_type"

    try:
        with Image.open(file_path) as img:
            # Verify the image
            img.verify()
            
            # Try to load image data
            img = Image.open(file_path)
            img.load()
            
            # Check image size
            width, height = img.size
            if width < 10 or height < 10:  # Minimum size check
                return False, "too_small"
                
            # Check if image has valid data
            try:
                img.getdata()[0]  # Try to access pixel data
            except Exception:
                return False, "invalid_data"
                
            return True, None
            
    except (IOError, SyntaxError, ValueError) as e:
        return False, "corrupted"
    except Exception as e:
        return False, f"unknown_error: {str(e)}"

def check_image(args):
    """
    Check single image and move if invalid
    Returns tuple of (filename, status, error_type)
    """
    src_path, dst_dir = args
    filename = os.path.basename(src_path)
    
    # Check if file is empty
    if os.path.getsize(src_path) == 0:
        dst_path = os.path.join(dst_dir, "empty", filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)
        return (filename, False, "empty")
    
    # Check if file is too small (less than 1KB)
    if os.path.getsize(src_path) < 1024:
        dst_path = os.path.join(dst_dir, "too_small", filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)
        return (filename, False, "too_small")
        
    # Perform image validation
    is_valid, error_reason = is_valid_image(src_path)
    if not is_valid:
        dst_path = os.path.join(dst_dir, error_reason, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)
        return (filename, False, error_reason)
        
    return (filename, True, "valid")

def main():
    # Define directories
    base_dir = "FEC_dataset"
    images_dir = os.path.join(base_dir, "all_images")
    bad_images_dir = os.path.join(base_dir, "bad_images")
    
    # Create bad images directory
    os.makedirs(bad_images_dir, exist_ok=True)
    
    # Get list of all files
    image_files = list(Path(images_dir).glob("*"))
    print(f"Found {len(image_files)} files to check")
    
    # Prepare arguments for parallel processing
    args = [(str(f), bad_images_dir) for f in image_files]
    
    # Process files in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_image, arg) for arg in args]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures),
                         desc="Checking images"):
            results.append(future.result())
    
    # Count results by error type
    error_counts = {}
    valid_count = 0
    
    for _, status, error_type in results:
        if status:
            valid_count += 1
        else:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    # Print summary
    print("\nResults Summary:")
    print(f"Total files checked: {len(results)}")
    print(f"Valid images: {valid_count}")
    print("\nInvalid files by type:")
    for error_type, count in error_counts.items():
        print(f"- {error_type}: {count}")
        print(f"  Location: {os.path.join(bad_images_dir, error_type)}")

if __name__ == "__main__":
    main() 