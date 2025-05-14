import pandas as pd
import requests
from urllib.parse import urlparse
import time
from tqdm import tqdm

def is_url_dead(url):
    try:
        # Remove quotes if present
        url = url.strip('"')
        response = requests.head(url, timeout=10)
        return response.status_code >= 400
    except:
        return True

def process_csv(input_file, output_file):
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(input_file, header=None)
    
    # Get all unique URLs from the first three columns
    urls = set()
    for col in range(0, 3):
        urls.update(df[col].unique())
    
    # Check each URL
    print("Checking URLs...")
    dead_urls = set()
    for url in tqdm(urls):
        if is_url_dead(url):
            dead_urls.add(url)
        time.sleep(0.5)  # Be nice to servers
    
    # Filter out rows with dead URLs
    print("Removing rows with dead URLs...")
    mask = ~df[[0, 1, 2]].isin(dead_urls).any(axis=1)
    df_filtered = df[mask]
    
    # Save to new CSV
    print(f"Saving filtered data to {output_file}...")
    df_filtered.to_csv(output_file, index=False, header=False)
    
    print(f"Original rows: {len(df)}")
    print(f"Rows after removing dead links: {len(df_filtered)}")
    print(f"Number of dead URLs found: {len(dead_urls)}")

if __name__ == "__main__":
    input_file = "FEC_dataset/faceexp-comparison-data-test-public.csv"
    output_file = "FEC_dataset/faceexp-comparison-data-test-public-filtered.csv"
    process_csv(input_file, output_file) 