import os

# Set the directory containing the files to be renamed
directory = r'C:\Users\jere-\FinalYearProject\adult opossum'

# Loop through the files in the directory
for filename in os.listdir(directory):
    # Construct the new filename with the original base name and the ".jpg" extension
    new_name = f"{filename}.jpg"
    # Rename the file
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

print("Done!")