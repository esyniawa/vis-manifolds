import nengo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys
import argparse

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)

from utils.matplotlib_encoding import AnimationEncoder, EncoderType


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Neural Manifold Visualization with Adaptive Encoder")

    # Simulation parameters
    parser.add_argument('--sim-time', type=float, default=10.0, help='Simulation time in seconds')
    parser.add_argument('--output-dir', type=str, default='./media', help='Output directory')
    parser.add_argument('--show-animation', type=bool, default=False, help='Show animation after saving')

    # Encoder parameters
    parser.add_argument('--encoder', type=str, default='auto',
                        choices=['auto', 'nvidia', 'amd', 'intel', 'vaapi', 'software', 'gif', 'frames'],
                        help='Encoder type')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second')
    parser.add_argument('--quality', type=int, default=23, help='Quality (0-51, lower is better)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower',
                                 'veryslow'],
                        help='Encoding speed preset')
    parser.add_argument('--bitrate', type=int, default=2000, help='Bitrate in Kbps')
    parser.add_argument('--threads', type=int, default=None, help='Number of CPU threads (None=auto)')
    parser.add_argument('--frames', type=int, default=200, help='Number of frames to render')

    return parser.parse_args(), parser


if __name__ == "__main__":
    # Parse arguments
    args, _ = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Map string encoder type to EncoderType enum
    encoder_map = {
        'auto': EncoderType.AUTO,
        'nvidia': EncoderType.NVIDIA,
        'amd': EncoderType.AMD,
        'intel': EncoderType.INTEL,
        'vaapi': EncoderType.VAAPI,
        'software': EncoderType.SOFTWARE,
        'gif': EncoderType.GIF,
        'frames': EncoderType.FRAMES
    }

    # Initialize encoder
    encoder = AnimationEncoder(
        encoder_type=encoder_map[args.encoder],
        output_dir=args.output_dir,
        quality=args.quality,
        fps=args.fps,
        speed_preset=args.preset,
        bitrate=args.bitrate,
        threads=args.threads,
        verbose=False
    )

    # Show available encoders
    if args.encoder == 'auto':
        encoder.list_available_encoders()

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
    print(f"Running neural simulation for {args.sim_time} seconds...")
    with nengo.Simulator(model) as sim:
        sim.run(args.sim_time)

    print("Simulation complete!")

    # Create the animation
    print("Setting up animation...")

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    for ax in [ax1, ax2]:
        ax.patch.set_alpha(0.0)
        ax.set_facecolor('none')

    neuron_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    frames = np.linspace(0, len(sim.trange()) - 1, args.frames, dtype=int)


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
        ax1.set_yticklabels(['X', 'Y', 'Z'])

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
    print("Creating animation...")
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=20,
        blit=False,
        repeat=True
    )

    # Adjust layout
    plt.tight_layout()

    # Save the animation using our encoder
    output_file = encoder.save_animation(anim, "manifold")
    print(f"Animation saved to: {output_file}")

    # Show the animation if requested
    if args.show_animation:
        plt.show()
    else:
        plt.close(fig)