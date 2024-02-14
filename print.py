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

# Function to read losses from a CSV file
def read_losses(csv_filename):
    losses = []
    with open(csv_filename, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split(",")
            losses.append((float(l[0]), float(l[1])))
    return losses

# Function to plot losses
def plot_losses(losses):
    # Separate generator and discriminator losses
    loss_generator, loss_discriminator = zip(*losses)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)  # Create subplot for generator loss
    plt.plot(loss_generator, label='Generator Loss', color='blue')
    plt.title('Generator Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)  # Create subplot for discriminator loss
    plt.plot(loss_discriminator, label='Discriminator Loss', color='red')
    plt.title('Discriminator Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


if __name__ == "__main__":

    file = "gan_losses.csv"
