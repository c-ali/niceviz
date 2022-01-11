# Niceviz

### Functionality

This handy mini-library reads, pre-processes and wraps json data coming from [visdom](https://github.com/fossasia/visdom) into python-dictionaries. It also comes with a plot-functionality for 1D (regular plot) and 2D (heatmap) Data.

### Quickstart

```python
# Read and preprocess
l = read_and_preprocess("link_to_log_folder")

# Use plot function for a single plot
    l["name_of_training_run"]["name_of_visdom_plot"].plot(ylabel="layer", log=True, xlabel="epoch", y_every=2, x_interval=(5, 30))

```

### Dependencies

The required dependencies are separately listed in dependencies.txt and can be installed with pip install -r dependencies.txt
