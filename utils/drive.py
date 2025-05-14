import gdown
import zipfile
import os

# Google Drive file ID
file_id = "1pmiCKcHxoFB1-V0qgfhXQJqMPYUeISml"

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Output filename
output = "all_images.zip"

# Download the file
gdown.download(url=url, output=output, fuzzy=True)
# Thư mục đích để giải nén
extract_to = "../all_images"

# Giải nén
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Đã giải nén vào thư mục: {extract_to}")