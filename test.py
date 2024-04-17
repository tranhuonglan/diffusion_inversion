from datasets import load_dataset
# first_examples now contains the first example of each class
from PIL import Image
import numpy as np
from src.split_class import subset_tiny_img
import os


def create_image_grid(image_paths, grid_size=(5, 4), image_size=(64, 64)):
    """
    Create a grid of images from a list of image paths.

    Args:
    - image_paths: List of paths to images.
    - grid_size: Tuple specifying the grid size (columns, rows).
    - image_size: Tuple specifying the size to resize images.

    Returns:
    - A PIL Image object representing the grid.
    """
    # Create a new image for the grid
    grid_img = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]))
    
    for index, img_path in enumerate(image_paths):
        # Open and resize the image
        img = Image.open(img_path).resize(image_size)
        # Calculate the position of the image in the grid
        x = (index % grid_size[0]) * image_size[0]
        y = (index // grid_size[0]) * image_size[1]
        # Paste the image into the grid
        grid_img.paste(img, (x, y))
        
    return grid_img

if __name__ == "__main__":


    # # Assuming 'dataset_path' is the path to your dataset directory
    # dataset_path = 'results/tiny-imagenet-syn-baseline'
    # folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    # # Sort folders to maintain consistency
    # folders.sort()

    # # Collect the first image from each folder
    # first_images = []
    # for folder in folders[:20]:  # Adjust this if you have more or fewer classes to fit into the 5x4 grid
    #     for file in os.listdir(folder):
    #         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             first_images.append(os.path.join(folder, file))
    #             break  # Stop after adding the first image

    # # Create and display the image grid
    # image_grid = create_image_grid(first_images, grid_size=(5, 4))
    # image_grid.show()

    # # Optionally, save the grid to a file
    # image_grid.save('syn_tiny_imagenet_grid.jpg')


    # Load the 'train' split of the 'Maysee/tiny-imagenet' dataset
    # dataset = load_dataset('Maysee/tiny-imagenet', split='train')

    # # Assuming each example in the dataset has a 'label' field indicating its class
    # # And assuming images are stored under a field, e.g., 'image'

    # # Initialize a dictionary to keep track of the first example of each class
    # first_examples = {}

    # for example in dataset:
    #     label = example['label']
    #     if (label in subset_tiny_img) and (label not in first_examples):
    #         first_examples[label] = example['image']


    # # Assuming first_examples is a list of PIL Image objects for simplicity
    # # If your data is not in this form, you will need to load and possibly resize your images accordingly

    # # Resize all images to the same size, e.g., 64x64 pixels
    # size = (64, 64)
    # resized_images = [img.resize(size) for img in first_examples.values()]

    # # Create a new image with a size to hold the 5x4 grid of images
    # grid_size = (size[0] * 5, size[1] * 4)
    # grid_img = Image.new('RGB', grid_size)

    # # Populate the grid with images
    # for i, img in enumerate(resized_images):
    #     # Calculate the position of the current image in the grid
    #     x = (i % 5) * size[0]
    #     y = (i // 5) * size[1]
    #     grid_img.paste(img, (x, y))

    # Save the grid image
    # grid_img.save('real_tiny_imagenet_grid.jpg')

    def split_tiny_imagenet(ds, selected_class):
        class_to_new_label = {old_label: new_label for new_label, old_label in enumerate(selected_class)}
        print('>>>>> Map class to new label', class_to_new_label)
        ds = ds.filter(lambda x: x['label'] in selected_class)

        def map_labels(sample):
            sample['label'] = class_to_new_label[sample['label']]
            return sample

        ds = ds.map(map_labels)
        return ds
    
    # from src.split_class import subset_tiny_img
    # from src.classes import i2d
  
    # ds_test = load_dataset('Maysee/tiny-imagenet', split='valid')
    # real_test = TinyImagenet(split_tiny_imagenet(ds_test, subset_tiny_img))
    # num_classes = len(subset_tiny_img)

    base_path = 'results/tiny-imagenet-syn-baseline'
    match_path = 'results/tiny-imagenet-syn-baseline-group2'
    # for file in os.listdir(base_path):
    #     # Define the new folder name with the class_id as a prefix
    #     new_folder_name = f"{class_id}_{folder_name}"
        
    #     # Define the original and new folder paths
    #     original_folder_path = os.path.join(base_path, folder_name)
    #     new_folder_path = os.path.join(base_path, new_folder_name)
        
    #     # Rename the folder
    #     os.rename(original_folder_path, new_folder_path)
    #     print(f"Renamed '{folder_name}' to '{new_folder_name}'")

    # pattern = re.compile(r"(\d{4})_\['(\d{4})'\]")
    import re
    pattern = re.compile(r"(\d{4})_\['(\d{4})'\]")
    for match_folder in os.listdir(match_path):
        match_name = match_folder.split('_')[-1]
        for folder in os.listdir(base_path):
            name = folder.split('_')[-1]
            if (name == match_name):
                print(name, match_name)
                original_full_path = os.path.join(match_path, match_folder)
                new_full_path = os.path.join(match_path, folder)
                os.rename(original_full_path, new_full_path)
                print(f"Folder renamed from {match_folder} to {folder}")
              
        # Match the pattern to each folder name
        # match = pattern.match(folder)
        # if match:
        #     # Extract the parts of the folder name
        #     prefix, suffix = match.groups()
        #     # Define the new folder name
        #     new_folder_name = f"{prefix}_class_{suffix}"
        #     # Full paths for original and new folder names
        #     original_full_path = os.path.join(base_path, folder)
        #     new_full_path = os.path.join(base_path, new_folder_name)
            
        #     # Rename the folder
        #     os.rename(original_full_path, new_full_path)
        #     print(f"Folder renamed from {folder} to {new_folder_name}")
        # else:
        #     print(f"No match found for {folder}, not renaming.")
