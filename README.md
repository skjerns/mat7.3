# mat 7.3
Load MATLAB 7.3 .mat files into Python.

Starting with MATLAB 7.3, `.mat` files have been changed to store as custom `hdf5` files.
This means they cannot be loaded by `scipy.io.loadmat` any longer and raise.

```Python
NotImplementedError: Please use HDF reader for matlab v7.3 files
```

## Quickstart

This library loads MATLAB 7.3 HDF5 files into a Python dictionary.

```Python
import mat73
data_dict = mat73.loadmat('data.mat')
```

As easy as that!

## Installation

To install, run:
```
pip install mat73
```

Alternatively:
```
pip install git+https://github.com/skjerns/mat7.3
```

## Short-comings

- This library will __only__ load mat 7.3 files. For older versions use `scipy.io.loadmat`
- `cell` objects are loaded as lists. That means they lose their structure and and need to be sorted manually
- Proprietary MATLAB types (e.g `datetime`, `duriation`, etc) are not supported. If someone tells me how to convert them, I'll implement that
- For now, you can't save anything back to the .mat. Let me know if you need this functionality, would be quick to implement.
