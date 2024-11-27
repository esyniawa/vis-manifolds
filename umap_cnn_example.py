import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', message='n_jobs.*random_state')


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_encoder = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_encoder(x)
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
        self.reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        self.scaler = StandardScaler()

        self.all_labels = []
        with torch.no_grad():
            for _, target in test_loader:
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
            embedding = self.reducer.fit_transform(scaled_vectors)

            self.embeddings_history.append(embedding)
            self.labels_history.append(self.all_labels)
            self.loss_history.append(loss)
            self.batch_numbers.append(batch_idx)

            model.train()

    def create_animation(self, save_path='latent_space_animation.gif'):
        fig = plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        def update(frame):
            ax1.clear()
            ax2.clear()

            scatter = ax1.scatter(self.embeddings_history[frame][:, 0],
                                  self.embeddings_history[frame][:, 1],
                                  c=self.labels_history[frame],
                                  cmap='tab10', s=5, alpha=0.6)
            ax1.set_title(f'CNN Latent Space (Batch {self.batch_numbers[frame]})')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')

            ax2.plot(self.batch_numbers[:frame + 1], self.loss_history[:frame + 1], 'b-')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Batch Number')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')

            plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=len(self.embeddings_history),
                             interval=100, repeat=True)
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        plt.close()


def train_and_visualize():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNNEncoder(latent_dim=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    visualizer = LatentSpaceVisualizer(test_loader, update_frequency=1)

    print("Training CNN and collecting visualizations...")
    for epoch in range(3):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            visualizer.update(model, batch_idx + epoch * len(train_loader), loss.item())

    print("Creating animation...")
    visualizer.create_animation('cnn_latent_space_animation.gif')
    print("Animation saved as 'cnn_latent_space_animation.gif'")


if __name__ == "__main__":
    train_and_visualize()