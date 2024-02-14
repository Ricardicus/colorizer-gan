import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import pickle
import random
import sys
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import numpy as np
import torch



def list_images(basedirectory, shuffle=True):
    # Go through all folders under basedirectory
    count = 0
    image_paths = []
    for folder in os.listdir(basedirectory):
        folder_path = os.path.join(basedirectory, folder)

        # Check if it's a folder
        if os.path.isdir(folder_path):
            # Go through all images in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Check if it's an image file
                if image_path.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".avif")
                ):
                    count += 1
                    image_paths.append(image_path)

    random.shuffle(image_paths)
    return image_paths

class ImageLoader:
    def __init__(self, image_paths, device, dimension=(128,128), loaded_in_memory=1000):
        self.device = device
        self.image_paths = image_paths
        self.loaded_in_memory = loaded_in_memory
        self.data = []
        self.dim = dimension
        self.data_range = (0, 0)  # Start and end indices of images currently loaded in memory

    def image_to_lab_vector(self, image_path):
        image = Image.open(image_path).convert("RGB")
        resize_transform = transforms.Resize(self.dim)
        image = resize_transform(image)

        img = np.array(image)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b

        img_lab[:, :, 0] = img_lab[:, :, 0] / 50 - 1 # Between -1 and 1 in L
        img_lab[:, :, 1] = img_lab[:, :, 1] / 127 # Between -1 and 1 in a
        img_lab[:, :, 2] = img_lab[:, :, 2] / 127 # Between -1 and 1 in b

        img_lab = img_lab.transpose((2, 0, 1))

        return img_lab

    def filter_out_grey_scale_images(self):
        new_images_paths = []
        count = 0
        found_greys = 0
        max_count = len(self.image_paths)
        for image_path in self.image_paths:
            if not self.is_grayscale(image_path):
                new_images_paths.append(image_path)
            else:
                found_greys += 1
            print(f"\rProcessing ... {count*100/max_count:.2f} %, found {found_greys} grey scale images",end="")
            count += 1
        reduced = len(self.image_paths) - len(new_images_paths)
        self.image_path = new_images_paths

        return reduced

    def is_grayscale(self, img_path):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)

        # Check if all colors are the same (R == G == B) across all pixels
        if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
            return True
        else:
            return False

    def grey_scale_vector_to_file(self, image_greyscale_vector, file):
        image_rgb = ( image_greyscale_vector + 1 ) * 50
        l_channel = image_rgb[0, :, :].astype(np.uint8)

        # The l_channel now represents the grayscale image
        # If needed, you can convert this to a 3-channel grayscale image
        greyscale_image = np.stack((l_channel, l_channel, l_channel), axis=-1)
        image = Image.fromarray(greyscale_image)
        image.save(file)

    def l_and_ab_to_lab(self, l, ab):
        lab = torch.cat((l, ab), dim=1)
        return lab

    def image_to_greyscale_vector(self, image_path):
        image = Image.open(image_path).convert("RGB")
        resize_transform = transforms.Resize(self.dim)
        image = resize_transform(image)

        img = np.array(image)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b

        img_lab[:, :, 0] = img_lab[:, :, 0] / 50 - 1 # Between -1 and 1 in L
        img_lab[:, :, 1] = img_lab[:, :, 1] / 127 # Between -1 and 1 in a
        img_lab[:, :, 2] = img_lab[:, :, 2] / 127 # Between -1 and 1 in b

        img_lab = img_lab.transpose((2, 0, 1))

        # Convert image to vector
        image_vector = img_lab

        # Create a greyscale image version
        image_greyscale = img_lab[0, :, :]

        image_greyscale_vector = np.array(image_greyscale)[np.newaxis, ...]

        return image_greyscale_vector

    def image_vector_to_file(self, image_vector_lab, file):
        if isinstance(image_vector_lab, torch.Tensor):
            image_vector_lab = image_vector_lab.detach().numpy()

        image_vector_lab[0, :, :] = (image_vector_lab[0, :, :] + 1) * 50

        image_vector_lab[1, :, :] = image_vector_lab[1, :, :] * 127
        image_vector_lab[2, :, :] = image_vector_lab[2, :, :] * 127

        # Clipping the LAB values to a valid range
        #image_vector_lab[0, :, :] = np.clip(image_vector_lab[0, :, :], 0, 100)
        #image_vector_lab[1, :, :] = np.clip(image_vector_lab[1, :, :], -128, 127)
        #image_vector_lab[2, :, :] = np.clip(image_vector_lab[2, :, :], -128, 127)

        image_lab = image_vector_lab.transpose((1, 2, 0))
        image_rgb = lab2rgb(image_lab) * 255

        image = Image.fromarray(image_rgb.astype(np.uint8))
        image.save(file)

    def load_images(self, start_index, end_index):
        self.data = []
        for i in range(start_index, min(end_index, len(self.image_paths))):
            image_path = self.image_paths[i]
            image = Image.open(self.image_paths[i]).convert("RGB")
            resize_transform = transforms.Resize(self.dim)
            image = resize_transform(image)

            img = np.array(image)
            img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b

            img_lab[:, :, 0] = img_lab[:, :, 0] / 50 - 1 # Between -1 and 1 in L
            img_lab[:, :, 1] = img_lab[:, :, 1] / 127 # Between -1 and 1 in a
            img_lab[:, :, 2] = img_lab[:, :, 2] / 127 # Between -1 and 1 in b

            img_lab = img_lab.transpose((2, 0, 1))

            # Convert image to vector
            image_vector = img_lab

            # Create a greyscale image version
            image_greyscale = img_lab[0, :, :]

            image_greyscale_vector = np.array(image_greyscale)[np.newaxis, ...]

            self.data.append((image_greyscale_vector, image_vector, image_path))

        self.data_range = (start_index, min(end_index, len(self.image_paths)))

    def poll(self, batch_index, batch_size):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size

        # Check if batch is within the range of currently loaded images
        if not (self.data_range[0] <= start_index < self.data_range[1] and self.data_range[0] <= end_index <= self.data_range[1]):
            self.load_images(start_index, end_index)

        selected_data = self.data[start_index - self.data_range[0] : end_index - self.data_range[0]]

        nbr_images = len(selected_data)
        if nbr_images == 0:
            return None

        train_x = np.zeros((nbr_images, 1, self.dim[1], self.dim[0]))
        train_y = np.zeros((nbr_images, 2, self.dim[1], self.dim[0]))

        for i, (example_x, example_y, image) in enumerate(selected_data):
            train_x[i] = example_x
            train_y[i] = example_y[1:, :, :]

        return (
            torch.from_numpy(train_x).float().to(self.device),
            torch.from_numpy(train_y).float().to(self.device),
            nbr_images,
        )

    def store_cool_cielab_image(self):
        self.data = []
        resize_transform = transforms.Resize(self.dim)
        # Pick a random image from the self.image_paths list
        random_image = random.choice(self.image_paths)
        image_grey = Image.open(random_image).convert("L")  # Convert to grayscale
        image_grey = resize_transform(image_grey)
        #image_grey.save("grey_image.jpg")  # Save grey image to file

        image_rgb = Image.open(random_image).convert("RGB")  # Open the same image in RGB mode
        image_rgb = resize_transform(image_rgb)
        image_rgb.save("rgb_image.jpg")  # Save RGB image to file

        img = np.array(image_rgb)  # Use the RGB image for conversion

        img_lab = rgb2lab(img).astype("float32")  # Convert RGB to L*a*b
        img_lab_a_only = img_lab.copy()
        img_lab_a_only[:, :, 1] = 0  # Set B channel to 0
        img_lab_a_only[:, :, 2] = 0  # Set B channel to 0
        img_lab_a_only_rgb = lab2rgb(img_lab_a_only)  # Convert back to RGB
        plt.imsave("grey_image.jpg", img_lab_a_only_rgb)  # Save A channel image to file

        img_lab = rgb2lab(img).astype("float32")  # Convert RGB to L*a*b
        img_lab_a_only = img_lab.copy()
        img_lab_a_only[:, :, 1] = 0  # Set B channel to 0
        img_lab_a_only_rgb = lab2rgb(img_lab_a_only)  # Convert back to RGB
        plt.imsave("a_image.jpg", img_lab_a_only_rgb)  # Save A channel image to file

        img_lab_b_only = img_lab.copy()
        img_lab_b_only[:, :, 2] = 0  # Set A channel to 0
        img_lab_b_only_rgb = lab2rgb(img_lab_b_only)  # Convert back to RGB
        plt.imsave("b_image.jpg", img_lab_b_only_rgb)  # Save B channel image to file
        # Save the final image to file
        final_image_path = "cool_cielab_image.jpg"
        # Read the grayscale image from file
        image_grey = Image.open("grey_image.jpg")

        # Read the LAB image with only A channel from file
        lab_a_only_image = Image.open("a_image.jpg")

        # Read the LAB image with only B channel from file
        lab_b_only_image = Image.open("b_image.jpg")

        # Create a new image with grey to the left, a in the middle, and b to the right
        combined_image = Image.new("RGB", (image_rgb.width + image_grey.width + lab_a_only_image.width + lab_b_only_image.width, max(image_grey.height, lab_a_only_image.height, lab_b_only_image.height)))

        # Paste the grayscale image to the left
        combined_image.paste(image_rgb, (0, 0))

        # Paste the grayscale image to the left
        combined_image.paste(image_grey, (image_grey.width, 0))

        # Paste the LAB image with only A channel in the middle
        combined_image.paste(lab_a_only_image, (2*image_grey.width, 0))

        # Paste the LAB image with only B channel to the right
        combined_image.paste(lab_b_only_image, (3*image_grey.width, 0))

        # Save the combined image to file
        combined_image.save("combined_image.jpg")


    def shuffle(self):
        random.shuffle(self.image_paths)
        self.data = []
        self.data_range = (0, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the pickle file")
    parser.add_argument("--output", default="out.png", help="Output file name")
    parser.add_argument("--image", type=int, default=0, help="Image index")
    args = parser.parse_args()

    traindataset, validationdataset = prepare_dataset(args.file)

    image, label = traindataset[args.image]
    # print(result)
    img = vector_to_image(image)
    img.save(args.output)
    print(
        f"Stored image nbr {args.image} representing '{label_to_string(label)}', as {args.output}"
    )
