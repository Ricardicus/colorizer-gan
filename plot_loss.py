import argparse
import random
import csv
import sys
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Function to read losses from a CSV file
def read_losses(csv_filename):
    losses = []
    with open(csv_filename, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split(",")
            losses.append((float(l[0]), float(l[1]),float(l[2]),float(l[3]), float(l[4]), float(l[5])))
    return losses

def plot_losses_plotly(losses):
    # Separate generator and discriminator losses
    loss_generator, loss_discriminator, moving_avg_gen, moving_avg_disc, adversarial_loss, adversarial_loss_mov_avg = zip(*losses)

    # Create subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Generator Reconstruction Loss Over Time", 
                                                        "Discriminator Loss Over Time", 
                                                        "Adversarial Loss Over Time"))

    # Generator Loss
    fig.add_trace(go.Scatter(y=loss_generator, mode='lines', name='Generator Loss', line=dict(color='blue', width=2, dash='solid'), opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(y=moving_avg_gen, mode='lines', name='Generator Moving Avg', line=dict(color='green', width=2, dash='dash')), row=1, col=1)

    # Discriminator Loss
    fig.add_trace(go.Scatter(y=loss_discriminator, mode='lines', name='Discriminator Loss', line=dict(color='red', width=2, dash='solid'), opacity=0.3), row=2, col=1)
    fig.add_trace(go.Scatter(y=moving_avg_disc, mode='lines', name='Discriminator Moving Avg', line=dict(color='yellow', width=2, dash='dash')), row=2, col=1)

    # Adversarial Loss
    fig.add_trace(go.Scatter(y=adversarial_loss, mode='lines', name='Adversarial Loss', line=dict(color='purple', width=2, dash='solid'), opacity=0.3), row=3, col=1)
    fig.add_trace(go.Scatter(y=adversarial_loss_mov_avg, mode='lines', name='Adversarial Moving Avg', line=dict(color='pink', width=2, dash='dash')), row=3, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="Epochs", row=3, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=3, col=1)

    # Update layout and show plot
    fig.update_layout(height=900, width=700, title_text="Loss Over Time", showlegend=True)
    fig.show()

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
    parser.add_argument("--file", default="gan_losses.csv", help="Path to the pickle file")

    args = parser.parse_args()
    file_path = args.file

    # Load losses from CSV file
    losses = read_losses(file_path)

    # plot losses
    plot_losses_plotly(losses)
