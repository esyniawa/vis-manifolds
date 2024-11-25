import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)


class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(MNISTEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits, latent


class LatentSpaceVisualizer:
    def __init__(self, test_loader, update_frequency=50):
        self.test_loader = test_loader
        self.update_frequency = update_frequency
        self.embeddings_history = []
        self.labels_history = []
        self.loss_history = []
        self.batch_numbers = []

        # Initialize UMAP reducer
        self.reducer = umap.UMAP(random_state=42)
        self.scaler = StandardScaler()

        # Get all test data for consistent UMAP transformation
        self.all_latent_vectors = []
        self.all_labels = []
        with torch.no_grad():
            for images, target in test_loader:
                self.all_labels.extend(target.numpy())
        self.all_labels = np.array(self.all_labels)

    def update(self, model, batch_idx, loss):
        if batch_idx % self.update_frequency == 0:
            model.eval()
            latent_vectors = []

            with torch.no_grad():
                for images, _ in self.test_loader:
                    _, latent = model(images)
                    latent_vectors.append(latent.numpy())

            latent_vectors = np.concatenate(latent_vectors, axis=0)
            scaled_vectors = self.scaler.fit_transform(latent_vectors)

            # Compute UMAP embedding
            embedding = self.reducer.fit_transform(scaled_vectors)

            self.embeddings_history.append(embedding)
            self.labels_history.append(self.all_labels)
            self.loss_history.append(loss)
            self.batch_numbers.append(batch_idx)

            model.train()

    def create_animation(self, save_path='latent_space_animation.gif'):
        fig = plt.figure(figsize=(15, 5))

        # Create two subplots: one for UMAP and one for loss
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        def update(frame):
            ax1.clear()
            ax2.clear()

            # Plot UMAP
            scatter = ax1.scatter(self.embeddings_history[frame][:, 0],
                                  self.embeddings_history[frame][:, 1],
                                  c=self.labels_history[frame],
                                  cmap='tab10', s=5, alpha=0.6)
            ax1.set_title(f'Latent Space (Batch {self.batch_numbers[frame]})')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')

            # Plot loss history
            ax2.plot(self.batch_numbers[:frame + 1], self.loss_history[:frame + 1], 'b-')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Batch Number')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')

            plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=len(self.embeddings_history),
                             interval=100, repeat=True)

        # Save animation
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        plt.close()


def train_and_visualize():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model setup
    model = MNISTEncoder(latent_dim=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize visualizer
    visualizer = LatentSpaceVisualizer(test_loader, update_frequency=50)

    # Training loop
    num_epochs = 3  # Reduced epochs since we're saving more frequently

    print("Training and collecting visualizations...")
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            # Update visualization
            visualizer.update(model, batch_idx + epoch * len(train_loader), loss.item())

    print("Creating animation...")
    visualizer.create_animation()
    print("Animation saved as 'latent_space_animation.gif'")


if __name__ == "__main__":
    train_and_visualize()
