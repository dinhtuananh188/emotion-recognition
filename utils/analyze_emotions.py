import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np

# Define paths
input_folder = "FEC_dataset/all_images"
csv_file = "emotion_distribution.csv"

def create_confidence_ranges():
    # Create ranges for confidence scores (0-10, 10-20, etc)
    ranges = []
    for i in range(0, 100, 10):
        ranges.append((i, i + 10))
    return ranges

def save_current_stats(disgust_counts, surprise_counts):
    # Create DataFrame
    df = pd.DataFrame({
        '%': [f"{start}-{end}" for start, end in create_confidence_ranges()],
        'Surprise': [surprise_counts[f"{start}-{end}"] for start, end in create_confidence_ranges()],
        'Disgust': [disgust_counts[f"{start}-{end}"] for start, end in create_confidence_ranges()]
    })
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    print("\nCurrent Results:")
    print(df.to_string(index=False))
    return df

def analyze_images():
    confidence_ranges = create_confidence_ranges()
    
    # Initialize counters for each range
    disgust_counts = {f"{start}-{end}": 0 for start, end in confidence_ranges}
    surprise_counts = {f"{start}-{end}": 0 for start, end in confidence_ranges}
    
    # Process each image
    total_processed = 0
    total_images = len([f for f in os.listdir(input_folder) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))])
    
    print(f"Found {total_images} images to process")
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue
            
        file_path = os.path.join(input_folder, filename)
        
        try:
            # Analyze image with DeepFace
            result = DeepFace.analyze(file_path, actions=['emotion'], enforce_detection=False, silent=True)
            
            if not result:
                continue
                
            # Process each face in the image
            for face_data in result:
                emotions = face_data['emotion']
                
                # Get confidence scores for disgust and surprise
                disgust_conf = emotions.get('disgust', 0)
                surprise_conf = emotions.get('surprise', 0)
                
                # Add to appropriate range bucket
                for start, end in confidence_ranges:
                    range_key = f"{start}-{end}"
                    if start <= disgust_conf < end:
                        disgust_counts[range_key] += 1
                    if start <= surprise_conf < end:
                        surprise_counts[range_key] += 1
            
            total_processed += 1
            if total_processed % 100 == 0:
                progress = (total_processed / total_images) * 100
                print(f"\nProcessed {total_processed}/{total_images} images ({progress:.1f}%)...")
                # Update CSV and display current stats
                save_current_stats(disgust_counts, surprise_counts)
                # Update plot
                df = pd.DataFrame({
                    '%': [f"{start}-{end}" for start, end in create_confidence_ranges()],
                    'Surprise': [surprise_counts[f"{start}-{end}"] for start, end in create_confidence_ranges()],
                    'Disgust': [disgust_counts[f"{start}-{end}"] for start, end in create_confidence_ranges()]
                })
                plot_distribution(df)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return disgust_counts, surprise_counts

def plot_distribution(df):
    plt.figure(figsize=(12, 6))
    
    # Plot lines
    plt.plot(range(len(df)), df['Disgust'], 'r-', label='Disgust', marker='o')
    plt.plot(range(len(df)), df['Surprise'], 'b-', label='Surprise', marker='o')
    
    # Customize plot
    plt.xticks(range(len(df)), df['%'], rotation=45)
    plt.xlabel('Confidence Range (%)')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Emotion Detection Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig('emotion_distribution.png')
    plt.close()

if __name__ == "__main__":
    print("Starting emotion analysis...")
    disgust_counts, surprise_counts = analyze_images()
    
    print("\nFinal Results:")
    df = save_current_stats(disgust_counts, surprise_counts)
    
    print("\nCreating final visualization...")
    plot_distribution(df)
    
    print(f"\nAnalysis complete. Final results saved to {csv_file} and emotion_distribution.png") 