import torch
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter


class LatentSpaceVisualizer:
    def __init__(self, test_loader, update_frequency=50):
        self.test_loader = test_loader
        self.update_frequency = update_frequency
        self.embeddings_history = []
        self.labels_history = []
        self.loss_history = []
        self.accuracy_history = []
        self.batch_numbers = []

        self.reducer = umap.UMAP(
            n_neighbors=50,
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
            ax1.set_title(f'Latent Space (Batch {self.batch_numbers[frame]})')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.set_xlim(-25, 25), ax1.set_ylim(-25, 25)

            # Loss plot with smoothing
            current_losses = self.loss_history[:frame + 1]
            current_batches = self.batch_numbers[:frame + 1]

            ax2.plot(current_batches, current_losses, 'b-', alpha=0.3, label='Raw')

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

        anim = FuncAnimation(fig, update,
                             frames=len(self.embeddings_history),
                             interval=100,
                             blit=False)

        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        plt.close()