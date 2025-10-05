from PIL import Image
import os

# Print the current working directory to help with debugging paths
print("Current working directory:", os.getcwd())

def resize_images_in_directory(root_dir, target_size=(64, 64)):
    """
    Resize all images inside 'train', 'val', and 'test' subfolders 
    of the root directory to the given target_size (default: 64x64).
    """
    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(root_dir, subset)
        
        # Check if the subset directory exists before proceeding
        if not os.path.exists(subset_path):
            print(f"⚠️ Skipping {subset_path}, directory does not exist.")
            continue

        # Iterate through each class folder inside the subset (e.g., Class0, Class1)
        for class_dir in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_dir)
            if not os.path.isdir(class_path):
                continue

            # Iterate through each image file in the class directory
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    # Open, resize, and save the image
                    with Image.open(image_path) as img:
                        img = img.resize(target_size)
                        img.save(image_path)
                        print(f"✅ Resized: {image_path}")
                except Exception as e:
                    # Handle non-image files (e.g., .DS_Store) or corrupted images
                    print(f"⚠️ Error processing {image_path}: {e}")

if __name__ == "__main__":
    # Run the function using '../data' as root and 64x64 as the target size
    resize_images_in_directory("../data", target_size=(64, 64))
