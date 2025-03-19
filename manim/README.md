# Overview

This is a visualization of the meaning of an embedding in a transformer. 
The visualization is created using [Manim Community](https://docs.manim.community/en/stable/index.html), a Python 
library for creating animations and interactive visualizations. All animations are insprired by the fantastic 
youtube channel [3Blue1Brown](https://www.youtube.com/c/3blue1brown).

## Installation

You need to install the required Python packages if you haven't already imported the conda environment ```env.yml``` .

```bash
conda install -c conda-forge manim
```

## Usage

```bash
manim -p -qh transformer_embedding.py TransformerEncodingAnimation
manim -p -qh transformer_embedding.py SelfAttentionMechanism
```

