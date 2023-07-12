import os
from PIL import Image
import shutil

# Specify your base directory here
base_directory = "./dataset/output_frames_clear/"

# Get the list of all subdirectories
subdirectories = [os.path.join(base_directory, name) for name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, name))]

for subdirectory in subdirectories:
    # Get the subdirectory name
    subdir_name = os.path.basename(subdirectory)

    # Get list of all image files in the subdirectory
    image_files = [f for f in os.listdir(subdirectory) if os.path.isfile(os.path.join(subdirectory, f))]

    for image_file in image_files:
        # Define the new name for the image
        new_image_name = subdir_name + "_" + image_file

        # Load the image
        img = Image.open(os.path.join(subdirectory, image_file))

        # Save the image to the parent directory with the new name
        img.save(os.path.join(base_directory, new_image_name))
