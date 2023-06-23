from PIL import Image
import os

def resize_images(folder_path, target_width):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        # Open the image using PIL
        image = Image.open(image_path)

        # Calculate the new height while preserving the aspect ratio
        width, height = image.size
        aspect_ratio = width / float(height)
        target_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image = image.resize((target_width, target_height), Image.ANTIALIAS)

        # Save the resized image back to the original path
        resized_image.save(image_path)

        print(f"Resized {image_file} to {target_width} width.")

# Example usage
target_width = 256
for folder_path in range(10):
    print("Folder : ", 0)
    resize_images(str(folder_path) + "/", target_width)

