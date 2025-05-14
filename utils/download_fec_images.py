import pandas as pd
import requests
import os
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def process_chunk(chunk, output_dir, max_workers=10):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare download tasks
    tasks = []
    urls_to_remove = []
    
    # Process each row to handle reference and test images
    for idx, row in chunk.iterrows():
        # Handle reference image
        if not pd.isna(row['ref_path']):
            parsed_url = urlparse(row['ref_path'])
            filename = os.path.basename(parsed_url.path)
            save_path = output_dir / filename
            
            if not save_path.exists():
                tasks.append((row['ref_path'], str(save_path)))
                urls_to_remove.append((idx, 'ref_path'))
        
        # Handle test image 1
        if not pd.isna(row['test_path1']):
            parsed_url = urlparse(row['test_path1'])
            filename = os.path.basename(parsed_url.path)
            save_path = output_dir / filename
            
            if not save_path.exists():
                tasks.append((row['test_path1'], str(save_path)))
                urls_to_remove.append((idx, 'test_path1'))
        
        # Handle test image 2
        if not pd.isna(row['test_path2']):
            parsed_url = urlparse(row['test_path2'])
            filename = os.path.basename(parsed_url.path)
            save_path = output_dir / filename
            
            if not save_path.exists():
                tasks.append((row['test_path2'], str(save_path)))
                urls_to_remove.append((idx, 'test_path2'))

    # Download images in parallel
    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_image, url, path) 
                      for url, path in tasks]
            
            # Track successful downloads without progress bar
            successful_downloads = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if future.result():
                    successful_downloads.append(i)
            
            # Remove URLs for successfully downloaded images
            for i in successful_downloads:
                idx, col = urls_to_remove[i]
                chunk.at[idx, col] = None

    return chunk

def process_csv_file(csv_path, output_dir, max_chunk_workers=4, max_download_workers=10):
    # Define column names based on the CSV structure from documentation
    columns = [
        'ref_path', 'x1', 'y1', 'w1', 'h1',  # Reference image
        'test_path1', 'x2', 'y2', 'w2', 'h2',  # Test image 1
        'test_path2', 'x3', 'y3', 'w3', 'h3',  # Test image 2
        'triplet_type'  # Type of triplet
    ]
    
    # Read CSV file in chunks to handle large files
    chunk_size = 1000
    chunks = list(pd.read_csv(csv_path, header=None, usecols=range(16), names=columns, chunksize=chunk_size))
    
    print(f"Processing {len(chunks)} chunks from {os.path.basename(csv_path)}")
    
    # Process chunks in parallel
    updated_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_chunk_workers) as chunk_executor:
        futures = [
            chunk_executor.submit(process_chunk, chunk, output_dir, max_download_workers) 
            for chunk in chunks
        ]
        
        # Wait for all chunks to complete and collect updated chunks
        for future in tqdm(concurrent.futures.as_completed(futures), 
                     total=len(futures), 
                     desc=f"Processing chunks from {os.path.basename(csv_path)}"):
            updated_chunks.append(future.result())
    
    # Combine all chunks and save updated CSV
    updated_df = pd.concat(updated_chunks, ignore_index=True)
    output_csv = csv_path.replace('.csv', '_updated.csv')
    updated_df.to_csv(output_csv, index=False, header=False)
    print(f"Saved updated CSV to {output_csv}")

def main():
    # Define paths
    base_dir = "FEC_dataset"
    train_csv = os.path.join(base_dir, "faceexp-comparison-data-train-public.csv")
    test_csv = os.path.join(base_dir, "faceexp-comparison-data-test-public.csv")
    
    # Create single output directory for all images
    output_dir = os.path.join(base_dir, "all_images")
    
    # Process training data
    print("Processing training data...")
    process_csv_file(train_csv, output_dir, max_chunk_workers=4, max_download_workers=10)
    
    # Process test data
    print("Processing test data...")
    process_csv_file(test_csv, output_dir, max_chunk_workers=4, max_download_workers=10)

if __name__ == "__main__":
    main() 