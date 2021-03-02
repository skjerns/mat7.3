![Python package](https://github.com/skjerns/mat7.3/workflows/Python%20package/badge.svg)  ![pypi Version](https://img.shields.io/pypi/v/mat73)

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

By enabling `use_attrdict=True` you can even access sub-entries of `structs` as attributes, just like in MATLAB:

```Python
data_dict = mat73.loadmat('data.mat', use_attrdict=True) 
struct = data_dict['structure'] # assuming a structure was saved in the .mat
struct[0].var1 == struct[0]['var1'] # it's the same!
```

## Installation

To install, run:

```
pip install mat73
```

Alternatively for most recent version:

```
pip install git+https://github.com/skjerns/mat7.3
```

## Datatypes

The following MATLAB datatypes can be loaded

| MATLAB                   | Python            |
| ------------------------ | ----------------- |
| logical                  | np.bool_          |
| single                   | np.float32        |
| double                   | np.float64        |
| int8/16/32/64            | np.int8/16/32/64  |
| uint8/16/32/64           | np.uint8/16/32/64 |
| complex                  | np.complex128     |
| char                     | str               |
| struct                   | list of dicts     |
| cell                     | list of lists     |
| canonical empty          | []                |
| missing                  | None              |
| Other (ie Datetime, ...) | Not supported     |

## Short-comings

- This library will __only__ load mat 7.3 files. For older versions use `scipy.io.loadmat`
- Proprietary MATLAB types (e.g `datetime`, `duriation`, etc) are not supported. If someone tells me how to convert them, I'll implement that
- For now, you can't save anything back to the .mat. Let me know if you need this functionality, would be quick to implement.
- See also [hdf5storage](https://github.com/frejanordsiek/hdf5storage), which can indeed be used for saving .mat, but has less features for loading
