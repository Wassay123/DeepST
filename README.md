DeepST
======
[DeepST](https://github.com/Wassay123/DeepST): A **Deep Learning** Toolbox for Spatio-Temporal Data

## Installation

DeepST uses the following dependencies: 

* [Keras](https://keras.io/#installation) and its dependencies are required to use DeepST. Please read [Keras Configuration](keras_configuration.md) for the configuration setting. 
* [TensorFlow](https://github.com/tensorflow/tensorflow#download-and-setup), used as the backend for Keras.
* [numpy](https://numpy.org/install/) and [scipy](https://www.scipy.org/install.html)
* HDF5 and [h5py](http://www.h5py.org/)
* [pandas](http://pandas.pydata.org/)
* CUDA 7.5 or the latest version. And **cuDNN** is highly recommended.

## Running the Code

To run the experiments and models provided in this repository, use the `exptBikeNYC.ipynb` or corresponding `.py` files.

## Data path

The default `DATAPATH` variable is `DATAPATH=[path_to_DeepST]/data`. You may set your `DATAPATH` variable using


```bash
# Windows
set DATAPATH=[path_to_your_data]

# Linux
export DATAPATH=[path_to_your_data]
```

## License

DeepST is released under the MIT License (refer to the LICENSE file for details).
