import os
import random



# Set the source and destination directories
src_dir = r'C:\Users\jere-\FinalYearProject\Animal Dataset\Train\Sheep'
dst_dir = r'C:\Users\jere-\FinalYearProject\Animal Dataset\Test\Sheep'

# Get a list of all the files in the source directory
files = os.listdir(src_dir)

# Randomly select 50 files from the list
selected_files = random.sample(files, 50)

# Move the selected files to the destination directory
for file in selected_files:
    src_path = os.path.join(src_dir, file)
    dst_path = os.path.join(dst_dir, file)
    os.rename(src_path, dst_path)

 