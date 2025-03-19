
## Overview

```python neural_manifolds.py```

The visualization consists of two main components:
1. A spike raster plot showing the firing patterns of three individual neurons
2. A 3D trajectory plot showing the neural manifold (the path traced by the combined neural activity)

## Installation

You need to install the required Python packages if you haven't already imported the conda environment ```env.yml``` .

```bash
pip install numpy==1.26.3
pip install nengo==4.0.0
pip install matplotlib
```

3. Install ffmpeg (for MP4 export):

On Ubuntu/Debian:
```bash
sudo apt-get install ffmpeg
```

On Conda:
```bash
conda install ffmpeg
```

## Technical Details

### Neural Network Configuration
- Uses 3 Leaky Integrate-and-Fire (LIF) neurons
- Each neuron has a different encoding vector aligned with one axis
- Firing rates: 200-400 Hz
- Membrane time constant (tau_rc): 0.02s
- Synaptic filtering: 0.01s on input, 0.05s on output

### Input Signal
- Smooth circular trajectory in 3D space
- Frequency: 0.5 Hz
- Amplitude: 0.8
- Third dimension phase-shifted by π/4

### Visualization Features
- Color-coded spike trains for each neuron
- Gradient-colored trajectory showing time progression
- Rotating 3D view for better spatial perception
- 1-second sliding window for spike raster
- Continuous looping animation

## Customization

Key parameters that can be modified:
- `max_rates`: Adjust neuron firing rates
- `tau_rc`: Change membrane time constant
- `window`: Modify the spike raster window size
- `freq`: Adjust trajectory frequency
- `amplitude`: Change trajectory size
- FPS and bitrate for animation export
