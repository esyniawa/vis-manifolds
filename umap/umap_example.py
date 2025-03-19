import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.xpu import device
from torchvision import datasets, transforms
from tqdm import tqdm

from latent_visualizer import LatentSpaceVisualizer

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


def train_and_visualize(device: torch.device):
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
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Initialize visualizer
    visualizer = LatentSpaceVisualizer(test_loader, update_frequency=1)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            # Add accuracy calculation
            accuracy = visualizer.compute_accuracy(logits, target)
            visualizer.update(model, batch_idx + epoch * len(train_loader), loss.item(), accuracy)

    print("Creating animation...")
    visualizer.create_animation()
    print("Animation saved as 'latent_space_animation.gif'")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_and_visualize(device)
