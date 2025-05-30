![pypi Version](https://img.shields.io/pypi/v/mat73)

# mat 7.3

Load MATLAB 7.3 .mat files into Python.

Starting with MATLAB 7.3, `.mat` files have been changed to store as custom `hdf5` files.
This means they cannot be loaded by `scipy.io.loadmat` any longer and raise.

```Python
NotImplementedError: Please use HDF reader for matlab v7.3 files, e.g. h5py
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

You can also specifiy to only load a specific variable or variable tree, useful to reduce loading times

```Python
data_dict = mat73.loadmat('data.mat', only_include='structure') 
struct = data_dict['structure'] # now only structure is loaded and nothing else

data_dict = mat73.loadmat('data.mat', only_include=['var/subvar/subsubvar', 'tree1/']) 
tree1 = data_dict['tree1'] # the entire tree has been loaded, so tree1 is a dict with all subvars of tree1
subsubvar = data_dict['var']['subvar']['subsubvar'] # this subvar has been loaded
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
| sparse                   | scipy.sparse.csc  |
| Other (ie Datetime, ...) | Not supported     |

## Short-comings

- This library will __only__ load mat 7.3 files. For older versions use `scipy.io.loadmat`
- Proprietary MATLAB types (e.g `datetime`, `duriation`, etc) are not supported. If someone tells me how to convert them, I'll implement that
- For now, you can't save anything back to the .mat. It's a bit more difficult than expected, so it's not on the roadmap for now
- See also [hdf5storage](https://github.com/frejanordsiek/hdf5storage), which can indeed be used for saving .mat, but has less features for loading
- See also [pymatreader](https://gitlab.com/obob/pymatreader/) which has a (maybe even better) implementation of loading MAT files, even for older ones
