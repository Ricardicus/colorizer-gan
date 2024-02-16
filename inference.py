import argparse
import random
import csv
import sys
import numpy as np
import torchvision
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_inspect import list_images, ImageLoader
from model import UNet, Discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to the pickle file")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--batch-load", type=int, default=5, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.003, help="Learning rate for training generator"
    )
    parser.add_argument(
        "--lr-disc", type=float, default=0.000009, help="Learning rate for training discriminator"
    )
    parser.add_argument(
        "--loss-mov-avg", type=float, default=0.9, help="Noice filtering, moving average, factor for loss value"
    )
    parser.add_argument(
        "--disc-hold-out", type=int, default=4, help="Hold out factor for training the discriminator"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use"
    )
    parser.add_argument(
        "--itr",
        type=int,
        default=99999999,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--train-percentage",
        type=float,
        default=100.0,
        help="Percentage of training examples used",
    )
    parser.add_argument(
        "--images", default=False, help="Include images if specified, default is False"
    )
    parser.add_argument(
        "--no-disc", default=False, help="Train only the generator (generator pre training)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Dimension of the images (width=height)",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=30,
        help="Complexity factor for the generator",
    )
    parser.add_argument(
        "--disc-complexity",
        type=int,
        default=32,
        help="Complexity factor for the discriminator",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Examples to generate"
    )
    parser.add_argument(
        "--model-disc",
        type=str,
        default="discriminator_model.pth",
        help="Discriminator model to load",
    )
    parser.add_argument(
        "--model-gen",
        type=str,
        default="generator_model.pth",
        help="Generator model to load",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    device = args.device
    disc_complexity = args.disc_complexity
    loss_mov_avg = args.loss_mov_avg
    no_disc = args.no_disc
    complexity = args.complexity
    dimension = (args.dim, args.dim)
    examples = args.examples

    image_loader = ImageLoader(None, device, dimension=dimension)

    file_gen = args.model_gen
    file_disc = args.model_disc

    a = UNet(1, 2, complexity=complexity)
    d = Discriminator(2, complexity=disc_complexity)

    if torch.backends.mps.is_available() and device == "mps":
        print("mps available")
        a = a.to("mps")
        d = d.to("mps")
    else:
        a = a.to(device)
        d = d.to(device)

    # loading the model from file if exists
    try:
        a.load_state_dict(torch.load(file_gen, map_location=torch.device('cpu')))
        print(f"Loaded model from {file_gen}.")
    except Exception as e:
        print(f"Failed to load model: {file_gen}")
        print(e)
        sys.exit(1)
    try:
        d.load_state_dict(torch.load(file_disc))
        print(f"Loaded model from {file_disc}.")
    except:
        pass

    images = list_images(args.images)
    print(f"Found {len(images)} images under {args.images}.")

    total_params_gen = sum(p.numel() for p in a.parameters())
    print(f"Generator number of parameters:     {total_params_gen}")

    # Generate some images
    random.shuffle(images)

    for i in range(examples):
        if i >= len(images):
            break
        image_path = images[i]
        image = Image.open(image_path)
        image = image.resize(dimension)
        # store original image
        image.save(f"outputs/original_{i}.png")

        #testFile = "outputs/testfile.png"
        #imgvtest = image_loader.image_to_lab_vector(image_path)
        #image_loader.image_vector_to_file(imgvtest, testFile)
        #sys.exit(0)

        image_greyscale_vector = image_loader.image_to_greyscale_vector(image_path)

        # save image greyscale
        image_loader.grey_scale_vector_to_file(image_greyscale_vector, f"outputs/greyscale_{i}.png")

        # turn it into a batch of size 1
        image_greyscale_vector = torch.unsqueeze(torch.from_numpy(image_greyscale_vector).float().to(device), 0)
        
        # run inference
        output = a(image_greyscale_vector)
        # Check what the discriminator says
        #disc_output = d(output)[0].to("cpu")
        #print(f"Disc image {i}: {disc_output}")

        # extract output from batch
        output = output[0].to("cpu")

        output_with_L = torch.cat([image_greyscale_vector[0], output], dim=0)
        # denormalize output
        image_loader.image_vector_to_file(output_with_L, f"outputs/output_{i}.png")


    if examples >= 9:
        rows, cols = 3, 6  # 3 rows and 6 columns as per your description
        collection_image = Image.new('RGB', (dimension[0] * cols, dimension[1] * rows))
        # Load and place the original images on the left side
        for i in range(9):  # Assuming 9 original images
            image_path = f"outputs/original_{i}.png"
            image = Image.open(image_path)
            # Calculate the position where this image will be placed in the collection
            x = (i % 3) * dimension[0]
            y = (i // 3) * dimension[1]
            collection_image.paste(image, (x, y))

        # Load and place the output images on the right side
        for i in range(9):  # Assuming 9 output images
            image_path = f"outputs/output_{i}.png"
            image = Image.open(image_path)
            # Calculate the position, noting that output images start from the fourth column
            x = (i % 3 + 3) * dimension[0] # +3 shifts the column to the right side
            y = (i // 3) * dimension[1]
            collection_image.paste(image, (x, y))

        # Save the final collection image
        collection_image.save("outputs/collection_image.png")




