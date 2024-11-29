import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    def __init__(self, latent_dim=32, dropout_rate=0.2):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),

            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),

            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2)
        )

        self.fc_encoder = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )

        # Separate classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        self.accuracy_history = []
        self.batch_numbers = []

        # More conservative UMAP parameters
        self.reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            metric='euclidean',
            random_state=42
        )
        self.scaler = StandardScaler()

        self.all_labels = []
        with torch.no_grad():
            for _, target in test_loader:
                self.all_labels.extend(target.numpy())
        self.all_labels = np.array(self.all_labels)

    def compute_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=1)
        return (predictions == labels).float().mean().item()

    def update(self, model, batch_idx, loss, accuracy):
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
            self.accuracy_history.append(accuracy)
            self.batch_numbers.append(batch_idx)

            model.train()

    def create_animation(self, save_path='latent_space_animation.gif'):
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, width_ratios=[1.5, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        def update(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            # UMAP plot
            scatter = ax1.scatter(self.embeddings_history[frame][:, 0],
                                  self.embeddings_history[frame][:, 1],
                                  c=self.labels_history[frame],
                                  cmap='tab10', s=5, alpha=0.6)
            ax1.set_title(f'CNN Latent Space (Batch {self.batch_numbers[frame]})')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.set_xlim(-25, 25), ax1.set_ylim(-25, 25)

            # Loss plot with smoothing
            current_losses = self.loss_history[:frame + 1]
            current_batches = self.batch_numbers[:frame + 1]

            # Plot raw loss
            ax2.plot(current_batches, current_losses, 'b-', alpha=0.3, label='Raw')

            # Plot smoothed loss only if we have enough data points
            if len(current_losses) > 10:
                window = 10
                smoothed_loss = np.convolve(current_losses,
                                            np.ones(window) / window,
                                            mode='valid')
                smooth_batches = current_batches[window - 1:]
                ax2.plot(smooth_batches, smoothed_loss, 'r-', label='Smoothed')

            ax2.set_title('Training Loss')
            ax2.set_xlabel('Batch Number')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')
            ax2.legend()

            # Accuracy plot
            ax3.plot(current_batches,
                     self.accuracy_history[:frame + 1], 'g-')
            ax3.set_title('Accuracy')
            ax3.set_xlabel('Batch Number')
            ax3.set_ylabel('Accuracy')
            ax3.set_ylim(0, 1)

            plt.tight_layout()

            # Return artist objects (required for animation)
            return scatter,

        # Create and save animation
        anim = FuncAnimation(fig, update,
                             frames=len(self.embeddings_history),
                             interval=100,
                             blit=False)  # Set blit=False for more stable animation

        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        plt.close()

def train_and_visualize():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data loading with augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = CNNEncoder(latent_dim=32, dropout_rate=0.2)
    criterion = nn.CrossEntropyLoss()

    # Use SGD with momentum instead of Adam
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    visualizer = LatentSpaceVisualizer(test_loader, update_frequency=1)

    print("Training CNN and collecting visualizations...")
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            accuracy = visualizer.compute_accuracy(logits, target)
            visualizer.update(model, batch_idx + epoch * len(train_loader),
                              loss.item(), accuracy)
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_epoch_loss)

    print("Creating animation...")
    visualizer.create_animation('stable_cnn_latent_space_animation.gif')
    print("Animation saved as 'stable_cnn_latent_space_animation.gif'")


if __name__ == "__main__":
    train_and_visualize()
