import nengo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


show_animation: bool = True

# Create the network
model = nengo.Network(label="Neural Manifold")
with model:
    # Create 3 neurons with different encoding vectors and improved parameters
    neurons = nengo.Ensemble(
        n_neurons=3,
        dimensions=3,
        neuron_type=nengo.LIF(tau_rc=0.02),
        max_rates=nengo.dists.Uniform(200, 400),
        encoders=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        intercepts=nengo.dists.Uniform(-0.2, 0.2),
        noise=None,
        label="neurons"
    )


    def input_signal(t):
        freq = 0.5
        amplitude = 0.8
        return [
            amplitude * np.sin(2 * np.pi * freq * t),
            amplitude * np.cos(2 * np.pi * freq * t),
            amplitude * np.sin(2 * np.pi * freq * t + np.pi / 4)
        ]


    stim = nengo.Node(input_signal)
    nengo.Connection(stim, neurons, synapse=0.01)

    spike_probe = nengo.Probe(neurons.neurons)
    decoded_probe = nengo.Probe(neurons, synapse=0.05)

# Run simulation
with nengo.Simulator(model) as sim:
    sim.run(10.0)

# Create the animation
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

neuron_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']


def update(frame):
    ax1.clear()
    ax2.clear()

    ax1.set_title("Spike Raster")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron")
    ax2.set_title("Neural Manifold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    window = 1.0
    start_idx = int(max(0, frame - window / sim.dt))
    end_idx = int(frame)

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Neuron 1', 'Neuron 2', 'Neuron 3'])

    for i in range(3):
        spikes = sim.data[spike_probe][start_idx:end_idx, i]
        times = sim.trange()[start_idx:end_idx]
        spike_times = times[spikes > 0]
        spike_y = np.ones_like(spike_times) * i

        ax1.plot(spike_times, spike_y, '|', color=neuron_colors[i],
                 markersize=15, markeredgewidth=2,
                 label=f'Neuron {i + 1}')
        ax1.axhline(y=i, color=neuron_colors[i], alpha=0.2, linestyle='-')

    ax1.set_ylim(-0.5, 2.5)
    ax1.set_xlim(max(0, sim.trange()[frame] - window), sim.trange()[frame])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    trajectory = sim.data[decoded_probe][:end_idx]

    if end_idx > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        for i in range(len(trajectory) - 1):
            ax2.plot3D(trajectory[i:i + 2, 0],
                       trajectory[i:i + 2, 1],
                       trajectory[i:i + 2, 2],
                       color=colors[i],
                       alpha=0.5)

        current_pos = trajectory[end_idx - 1]
        ax2.scatter(*current_pos, c='red', s=100)

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.view_init(elev=20, azim=frame / 2)

    return ax1, ax2


# Create animation with repeat enabled
anim = FuncAnimation(
    fig,
    update,
    frames=np.arange(0, len(sim.trange()), 10),
    interval=20,
    blit=False,
    repeat=True  # Enable repeating
)

# Adjust layout before saving
plt.tight_layout()

# Save as MP4 (high quality)
try:
    print("Saving MP4...")
    writer_mp4 = FFMpegWriter(fps=10, bitrate=2000)
    anim.save('neural_manifold.mp4', writer=writer_mp4)
except Exception as e:
    print(f"Could not save MP4 (requires ffmpeg): {e}")
    print("Try installing ffmpeg with: sudo apt-get install ffmpeg")


# Save as GIF (more compatible)
print("Saving GIF...")
writer_gif = PillowWriter(fps=10)
anim.save('neural_manifold.gif', writer=writer_gif)

if show_animation:
    plt.show()