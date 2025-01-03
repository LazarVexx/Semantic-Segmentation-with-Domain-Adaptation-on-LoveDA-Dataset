import os

def generate_lst(image_folder, mask_folder, output_file):
    # Get all image files (with .png extension) sorted by name (assuming they are named numerically)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    
    # Open the output .lst file for writing
    with open(output_file, 'w') as f:
        # Loop through each image file and find its corresponding mask
        for image_file in image_files:
            # Construct full paths for image and mask
            image_path = os.path.join(image_folder, image_file)
            
            # Assuming mask filenames match image filenames (e.g., 0.png -> 0.png)
            mask_path = os.path.join(mask_folder, image_file)
            
            # Check if the corresponding mask file exists
            if os.path.exists(mask_path):
                # Write the image and mask paths with a tab between them to the .lst file
                f.write(f'{image_path}\t{mask_path}\n')
            else:
                print(f"Warning: Mask file for {image_file} not found!")

    print(f".lst file generated successfully: {output_file}")
    
# Example usage
image_folder = 'PIDNet/data/loveDa/Train/Rural/images_png'  # Folder containing the image PNG files
mask_folder = 'PIDNet/data/loveDa/Train/Rural/masks_png'    # Folder containing the mask PNG files
output_file = 'PIDNet/data/loveDa/Train.lst'  # Output .lst file path

generate_lst(image_folder, mask_folder, output_file)
