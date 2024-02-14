import argparse
import random
import csv
import sys
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_inspect import (denormalize, label_to_string_one_hot, list_images,
                          ImageLoader,
                          prepare_dataset,
                          prepare_dataset_from_dictionary_greyscale_x,
                          vector_to_image)
from model import Autoencoder, UNet, Discriminator


def train(
    generator, discriminator, batch, optimizer_g, optimizer_d, criterion_generator, criterion_discriminator, iterations_acc=0, learningrate=0.003, 
    train_generator = True, train_discriminator = True, adversarial_factor = 0.1, image_loader = None
):
    train_x, train_y = batch

    xy_combined = torch.cat([train_x, train_y], dim=1)

    device = next(generator.parameters()).device

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    # Train discriminator on real image
    real_labels = torch.zeros(train_y.size(0)).to(device)
    real_labels += 1#random.uniform(0.8, 1.0)
    output = discriminator(xy_combined).view(-1)
    loss_discriminator_real = criterion_discriminator(output, real_labels)
    if train_discriminator:
        loss_discriminator_real.backward()
        optimizer_d.step()
    # Train generator, store output for fake image
    fake_image = generator(train_x).detach()
    
    # Train discriminator on fake image, noisy labels
    fake_labels = torch.zeros(train_y.size(0)).to(device)
    fake_labels += 0#random.uniform(0.0, 0.2)
    x_fake_combined = torch.cat([train_x, fake_image], dim=1)
    d_output = discriminator(x_fake_combined).view(-1)
    loss_discriminator_fake = criterion_discriminator(d_output, fake_labels)
    if train_discriminator:
        loss_discriminator_fake.backward()
        optimizer_d.step()

    if image_loader:
        labv_fake = x_fake_combined
        labv_real = xy_combined

        # store the images
        #image_loader.image_vector_to_file(labv_fake[0], f"outputs/{d_output[0]}_fake.png")
        #image_loader.image_vector_to_file(labv_real[0], f"outputs/{output[0]}_real.png")
        

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    # Generate output from grey scale image
    output = generator(train_x)
    # How far off is the generator from the original image
    content_loss = criterion_generator(output, train_y)
    # What does the discriminator say about the generated image
    output_combined = torch.cat([train_x, output], dim=1)
    output_disc = discriminator(output_combined)

    # I want to fool the discriminator to say that it is true (real)
    # apply noisy labels
    input_label = torch.zeros(train_x.size(0), 1).to(device)
    input_label += 1#random.uniform(0.8, 1.0)
    advesarial_loss = criterion_discriminator(output_disc, input_label)
    if train_generator:
        # scale the loss of the discriminator
        loss_generator = (1-adversarial_factor) * content_loss + adversarial_factor * advesarial_loss 
        loss_generator.backward()

        optimizer_g.step()

    return content_loss.item(), loss_discriminator_real.item(), advesarial_loss.item(), loss_generator.item(), output_disc


def inference(autoencoder, data, outfile):
    return autoencoder(data)

# Function to read losses from a CSV file
def read_losses(csv_filename):
    losses = []
    with open(csv_filename, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split(",")
            losses.append((float(l[0]), float(l[1]),float(l[2]),float(l[3]), float(l[4]), float(l[5])))
    return losses

def plot_losses(losses):
    # Separate generator and discriminator losses
    loss_generator, loss_discriminator, moving_avg_gen, moving_avg_disc, adversarial_loss, adversarial_loss_mov_avg = zip(*losses)

    # Plotting
    plt.figure(figsize=(10, 15))  # Adjusted figure size to accommodate new subplot
    plt.subplot(3, 1, 1)  # Adjusted subplot index for generator loss
    plt.plot(loss_generator, label='Loss', color='blue', alpha=0.3)  # Made non-moving average plot more transparent
    plt.plot(moving_avg_gen, label='Moving Avg', color='green', linestyle='--')
    plt.title('Generator Reconstruction Loss Over Time')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)  # Adjusted subplot index for discriminator loss
    plt.plot(loss_discriminator, label='Discriminator Loss', color='red', alpha=0.3)  # Made non-moving average plot more transparent
    plt.plot(moving_avg_disc, label='Discriminator Moving Avg', color='orange', linestyle='--')
    plt.title('Discriminator Loss Over Time')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 3)  # Added subplot for adversarial loss
    plt.plot(adversarial_loss, label='Adversarial Loss', color='purple', alpha=0.3)  # Made non-moving average plot more transparent
    plt.plot(adversarial_loss_mov_avg, label='Adversarial Moving Avg', color='pink', linestyle='--')
    plt.title('Adversarial Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


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
        "--lr-disc", type=float, default=0.0007, help="Learning rate for training discriminator"
    )
    parser.add_argument(
        "--adversarial-factor", type=float, default=0.01, help="Loss factor for adversarial loss. loss_generator = content_loss + adversarial_factor * adversarial_loss"
    )
    parser.add_argument(
        "--loss-mov-avg", type=float, default=0.98, help="Noice filtering, moving average, factor for loss value"
    )
    parser.add_argument(
        "--disc-hold-out", type=int, default=4, help="Hold out factor for training the discriminator"
    )
    parser.add_argument(
        "--img-scale", type=int, default=1, help="Reduce the image dimension with this factor"
    )
    parser.add_argument(
        "--losses-file", type=str, default="gan_losses.csv", help="Device to use"
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
        "--store-net",
        type=int,
        default=1000,
        help="Store network every n",
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
        "--step-decay",
        type=int,
        default=500,
        help="Number of iterations before the learning rate is decreased",
    )
    parser.add_argument(
        "--step-decay-factor",
        type=float,
        default=0.5,
        help="The factor with which the learning rate is decreased",
    )
    parser.add_argument(
        "--train-percentage",
        type=float,
        default=100.0,
        help="Percentage of training examples used",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Dimension of the images (width=height)",
    )
    parser.add_argument(
        "--images", default=False, help="Include images if specified, default is False"
    )
    parser.add_argument(
        "--no-disc", default=False, help="Train only the generator (generator pre training)"
    )
    parser.add_argument(
        "--filter-gray",
        action="store_true",
        help="Filter out gray scale images"
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
    loss_mov_avg = args.loss_mov_avg
    no_disc = args.no_disc
    complexity = args.complexity
    disc_complexity = args.disc_complexity
    losses_file = args.losses_file
    store_net = args.store_net
    img_scale = args.img_scale
    step_decay = args.step_decay
    step_decay_factor = args.step_decay_factor
    adversarial_factor = args.adversarial_factor
    dimension = (args.dim, args.dim)
    max_epocs = args.epochs
    filter_gray = args.filter_gray

    file_gen = args.model_gen
    file_disc = args.model_disc

    a = UNet(1, 2, complexity=complexity)
    d = Discriminator(3, complexity=disc_complexity, dimension=dimension[0])

    if torch.backends.mps.is_available() and device == "mps":
        print("mps available")
        a = a.to("mps")
        d = d.to("mps")
    else:
        a = a.to(device)
        d = d.to(device)

    # loading the model from file if exists
    try:
        a.load_state_dict(torch.load(file_gen))
        print(f"Loaded model from {file_gen}.")
    except:
        pass
    try:
        d.load_state_dict(torch.load(file_disc))
        print(f"Loaded model from {file_disc}.")
    except:
        pass

    images = list_images(args.images)
    print(f"Found {len(images)} images under {args.images}.")


    optimizer = optim.Adam(a.parameters(), lr=args.lr)
    criterion_generator = torch.nn.L1Loss()
    criterion_discriminator = torch.nn.BCELoss()
    optimizer_d = optim.Adam(d.parameters(), lr=args.lr_disc)

    batch_load = args.batch_load
    disc_hold_out = args.disc_hold_out
    batch_count = 0
    iterations = 0
    nbr_images_in_data = len(images)

    image_loader = ImageLoader(images, device, dimension=dimension, loaded_in_memory=batch_load)
   
    if filter_gray:
        reduced = image_loader.filter_out_grey_scale_images()
        if reduced > 0:
            print(f"\nRemoved {reduced} grey scaled images from the dataset")

    print(f"Will process {batch_load} images at a time in memory")
    total_params_gen = sum(p.numel() for p in a.parameters())
    total_params_disc = sum(p.numel() for p in d.parameters())
    print(f"Generator number of parameters:     {total_params_gen}")
    print(f"Discriminator number of parameters: {total_params_disc}")
    print(f"Adversarial loss factor: {adversarial_factor}")
    print(f"Discrimator hold out: {disc_hold_out}")

    loss_generator_mov_avg = None
    loss_discriminator_mov_avg = None
    loss_adversarial_mov_avg = None
    loss_total_mov_avg = None
    batch_count = 0
    epochs = 0
    loss_content = ""
    while iterations < args.itr and epochs <= max_epocs:
        loss = 0
        # train discriminator at this rate
        discriminator_train_rate = disc_hold_out # every forth
        batch_walk = 0

        if batch_count * batch_size >= nbr_images_in_data:
            epochs += 1
            batch_count = 0
            image_loader.shuffle()
            continue

        train_x, train_y, nbr_examples = image_loader.poll(batch_count, batch_size)

        if nbr_examples != batch_size:
            epochs += 1
            batch_count = 0
            image_loader.shuffle()
            continue

        train_x = train_x.to(device)
        train_y = train_y.to(device)

        # Create batches for train_x and train_y
        train_x_batches = torch.split(train_x, batch_size)
        train_y_batches = torch.split(train_y, batch_size)

        # Store status to csv
        csv_filename = losses_file
        loss_file = open(csv_filename, mode='a', newline='')

        disc_output = [0]
        while batch_walk < len(train_x_batches) and iterations < args.itr:
            train_discriminator = (iterations % discriminator_train_rate) == 0
            if no_disc:
                train_discriminator = False
            content_loss, loss_discriminator, adversarial_loss, total_loss, disc_output = train(
                a,
                d,
                (train_x_batches[batch_walk], train_y_batches[batch_walk]),
                optimizer,
                optimizer_d,
                criterion_generator,
                criterion_discriminator,
                iterations,
                train_discriminator = train_discriminator,
                train_generator = True,
                adversarial_factor = adversarial_factor,
                image_loader = image_loader
            )
            iterations += 1
            batch_count += 1
            batch_walk += 1

            if loss_discriminator_mov_avg == None:
                loss_discriminator_mov_avg = loss_discriminator
                loss_generator_mov_avg = content_loss
                loss_adversarial_mov_avg = adversarial_loss
                loss_total_mov_avg = total_loss
            else:
                loss_discriminator_mov_avg = loss_discriminator_mov_avg * loss_mov_avg + (1.0 - loss_mov_avg) * loss_discriminator
                loss_generator_mov_avg = loss_generator_mov_avg * loss_mov_avg + (1.0 - loss_mov_avg) * content_loss
                loss_adversarial_mov_avg = loss_adversarial_mov_avg * loss_mov_avg + (1.0 - loss_mov_avg) * adversarial_loss
                loss_total_mov_avg = loss_total_mov_avg * loss_mov_avg + (1.0 - loss_mov_avg) * total_loss

            generator_learning_rate = optimizer.param_groups[0]['lr']
            discriminator_learning_rate = optimizer_d.param_groups[0]['lr']

            if iterations % step_decay == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * step_decay_factor
                for param_group in optimizer_d.param_groups:
                    param_group['lr'] = param_group['lr'] * step_decay_factor

            loss_content += f"{content_loss},{loss_discriminator},{loss_generator_mov_avg},{loss_discriminator_mov_avg},{adversarial_loss},{loss_adversarial_mov_avg}\n"
            if iterations % 200 == 0:
                loss_file.write(loss_content)
                loss_content = ""
        print(f"Iteration {iterations}, epochs: {epochs}, rec. loss mov.avg.: {loss_generator_mov_avg} (lr: {generator_learning_rate}), disc. loss mov.avg. {loss_discriminator_mov_avg} (lr: {discriminator_learning_rate}), adver. loss mov. avg.: {loss_adversarial_mov_avg} max iterations: {args.itr}, (disc[0]: {disc_output[0].item()})")
        loss_file.close()
        if (iterations % store_net) == 0:
            # Save the model
            print(f"Saving models {file_gen} and {file_disc}")
            torch.save(a.state_dict(), file_gen)
            torch.save(d.state_dict(), file_disc)


    # Save the model
    torch.save(a.state_dict(), file_gen)
    torch.save(d.state_dict(), file_disc)
    # Generate some images
    random.shuffle(images)

    examples = 10
    for i in range(examples):
        if i >= len(images):
            break
        image_path = images[i]
        image = Image.open(image_path)
        image = image.resize(dimension)
        # store original image
        image.save(f"outputs/original_{i}.png")

        image_greyscale_vector = image_loader.image_to_greyscale_vector(image_path)

        # save image greyscale
        image_loader.grey_scale_vector_to_file(image_greyscale_vector, f"outputs/greyscale_{i}.png")

        # turn it into a batch of size 1
        image_greyscale_vector = torch.unsqueeze(torch.from_numpy(image_greyscale_vector).float().to(device), 0)
        
        # run inference
        output = a(image_greyscale_vector)
        # Check what the discriminator says

        # extract output from batch
        output = output[0].to("cpu")

        output_with_L = torch.cat([image_greyscale_vector[0].to("cpu"), output], dim=0)
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





