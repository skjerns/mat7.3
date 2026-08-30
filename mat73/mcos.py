# -*- coding: utf-8 -*-
"""
Central reader for MATLAB MCOS objects stored in v7.3 (HDF5) files.

MATLAB serializes instances of classdef classes (table, datetime, string,
categorical, containers.Map, user-defined classes, ...) through MCOS, the
MATLAB Class Object System. In a v7.3 file such a variable is not its data:
it is a small uint32 dataset (an object header) that points into a hidden
subsystem where the property values live. MathWorks has never documented
this layout. What follows was reverse engineered independently by several
projects (mat73-reader, matio, MAT.jl, MatFileHandler) and is verified here
against files written by MATLAB.

Object header
-------------
A uint32 dataset carrying the attribute ``MATLAB_object_decode = 3``::

    [0xDD000000, ndims, dim_1 .. dim_ndims, id_1 .. id_n, class_id]

``0xDD000000`` marks an MCOS header. ``n = prod(dims)`` object ids follow
the dims in column-major order; the last element is the class id. An empty
object array carries no ids.

Subsystem
---------
``#subsystem#/MCOS`` is a cell array (HDF5 object references) whose MATLAB
class is ``FileWrapper__``::

    cell[0]       uint8 metadata blob, parsed below
    cell[1]       canonical empty, unused by this reader
    cell[2:-3]    property values (the "heap"); property entries point here
    cell[-3]      per-class data of unknown purpose
    cell[-2]      int32 class alias indices
    cell[-1]      per-class default property values

Metadata blob (little-endian int32 unless noted)
------------------------------------------------
::

    int32[0]       version (2, 3 and 4 have been seen in the wild)
    int32[1]       number of names
    int32[2:10]    eight byte offsets into the blob; regions used here:
                     offsets[0]  class table
                     offsets[1]  property lists of saveobj-type objects
                     offsets[2]  object table
                     offsets[3]  property lists of regular objects
                     offsets[4]  dynamic-property lists (not needed to read)
    bytes[40:offsets[0]]  null-terminated names (class and property names);
                          references to names are 1-based

    class table    4 x int32 per class id:
                     (namespace_name_idx, class_name_idx, 0, 0)
                   entry 0 is a placeholder
    object table   6 x int32 per object id:
                     (class_id, 0, 0, saveobj_list_idx, regular_list_idx,
                      dependency_id)
                   entry 0 is a placeholder; exactly one of the two list
                   indices is non-zero
    property lists a sequence of blocks; block k is referenced by list index
                   k, and block 0 is an empty placeholder. Each block is
                     (nprops, then nprops x (name_idx, kind, value))
                   padded to an 8-byte boundary. ``kind`` selects how
                   ``value`` is read:
                     0  a name index: the value is that name (enumerations)
                     1  a heap index: the value is ``cell[value + 2]``
                     2  a literal integer
"""
import numpy as np
import h5py

__all__ = ['MCOS_MARKER', 'MCOSObjectHeader', 'MCOSSubsystem',
           'is_mcos_header']

MCOS_MARKER = 0xDD000000

# Where the heap begins inside the FileWrapper__ cell array; see module doc.
_HEAP_OFFSET = 2


def is_mcos_header(dataset):
    """True if ``dataset`` is an MCOS object header (see module doc)."""
    if not isinstance(dataset, h5py.Dataset):
        return False
    if dataset.attrs.get('MATLAB_object_decode', None) != 3:
        return False
    if dataset.dtype != np.uint32 or dataset.size < 3:
        return False
    return int(np.asarray(dataset).flat[0]) == MCOS_MARKER


class MCOSObjectHeader:
    """Decoded object header: array dims, the object ids and the class id."""

    __slots__ = ('dims', 'object_ids', 'class_id')

    def __init__(self, dims, object_ids, class_id):
        self.dims = tuple(int(d) for d in dims)
        self.object_ids = [int(i) for i in object_ids]
        self.class_id = int(class_id)

    @classmethod
    def from_dataset(cls, dataset):
        values = np.asarray(dataset, dtype=np.uint32).ravel().tolist()
        if not values or values[0] != MCOS_MARKER:
            raise ValueError(f'{dataset.name} is not an MCOS object header')
        ndims = values[1]
        dims = values[2:2 + ndims]
        n_objects = int(np.prod(dims)) if dims else 0
        ids = values[2 + ndims:2 + ndims + n_objects]
        class_id = values[-1]
        if len(ids) != n_objects:
            raise ValueError(f'{dataset.name}: header declares {n_objects} '
                             f'objects but carries {len(ids)} ids')
        return cls(dims, ids, class_id)

    def __repr__(self):
        return (f'MCOSObjectHeader(dims={self.dims}, '
                f'object_ids={self.object_ids}, class_id={self.class_id})')


class MCOSSubsystem:
    """Parsed ``#subsystem#/MCOS`` of one file.

    Construction reads and decodes the metadata blob once. Property values
    are handed back as the raw h5py objects they reference (datasets or
    groups), so the caller decides how to convert them; that keeps this
    module free of any conversion policy.
    """

    def __init__(self, hdf5_file):
        if not self.present(hdf5_file):
            raise KeyError('file has no #subsystem#/MCOS group')
        self._file = hdf5_file
        wrapper = hdf5_file['#subsystem#/MCOS']
        self._cells = [ref for ref in np.asarray(wrapper).ravel()]
        blob = np.asarray(hdf5_file[self._cells[0]], dtype=np.uint8)
        self._parse_metadata(blob.ravel().tobytes())

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def present(hdf5_file):
        return ('#subsystem#' in hdf5_file
                and 'MCOS' in hdf5_file['#subsystem#'])

    def _parse_metadata(self, blob):
        header = np.frombuffer(blob[:40], dtype='<i4')
        self.version = int(header[0])
        n_names = int(header[1])
        offsets = [int(o) for o in header[2:10]]

        raw_names = blob[40:offsets[0]].split(b'\x00')
        self.names = [s.decode('utf-8') for s in raw_names if s]
        if len(self.names) != n_names:
            raise ValueError(f'MCOS metadata declares {n_names} names, '
                             f'found {len(self.names)}')

        def region(start, end):
            return np.frombuffer(blob[start:end], dtype='<i4')

        self._classes = region(offsets[0], offsets[1]).reshape(-1, 4)
        self._saveobj_lists = self._parse_property_lists(
            region(offsets[1], offsets[2]))
        self._objects = region(offsets[2], offsets[3]).reshape(-1, 6)
        self._regular_lists = self._parse_property_lists(
            region(offsets[3], offsets[4]))

    @staticmethod
    def _parse_property_lists(values):
        lists = []
        pos = 0
        n = len(values)
        while pos < n:
            nprops = int(values[pos])
            block_len = 1 + 3 * nprops
            entries = values[pos + 1:pos + block_len].reshape(nprops, 3)
            lists.append([(int(a), int(b), int(c)) for a, b, c in entries])
            pos += block_len + (block_len % 2)   # pad to 8 bytes
        return lists

    # -- lookups -------------------------------------------------------------

    def name(self, index):
        """Resolve a 1-based name index; 0 means 'no name'."""
        return self.names[index - 1] if index else ''

    @property
    def n_classes(self):
        return len(self._classes) - 1

    @property
    def n_objects(self):
        return len(self._objects) - 1

    def class_name(self, class_id):
        namespace_idx, name_idx = self._classes[class_id][:2]
        name = self.name(int(name_idx))
        namespace = self.name(int(namespace_idx))
        return f'{namespace}.{name}' if namespace else name

    def class_of(self, object_id):
        return int(self._objects[object_id][0])

    def read_header(self, dataset):
        return MCOSObjectHeader.from_dataset(dataset)

    def heap(self, index):
        """The h5py object a kind-1 property value points to."""
        return self._file[self._cells[index + _HEAP_OFFSET]]

    def properties(self, object_id, convert=None):
        """Properties of one object as an ordered dict.

        Kind-1 values are returned as h5py datasets/groups, or passed through
        ``convert`` when given. Objects written through a custom ``saveobj``
        expose a single property named ``any`` holding whatever ``saveobj``
        returned.
        """
        _, _, _, saveobj_idx, regular_idx, _ = (
            int(v) for v in self._objects[object_id])
        if saveobj_idx:
            entries = self._saveobj_lists[saveobj_idx]
        else:
            entries = self._regular_lists[regular_idx]

        props = {}
        for name_idx, kind, value in entries:
            key = self.name(name_idx)
            if kind == 0:
                props[key] = self.name(value)
            elif kind == 1:
                obj = self.heap(value)
                props[key] = convert(obj) if convert is not None else obj
            elif kind == 2:
                props[key] = value
            else:
                raise ValueError(f'unknown MCOS property kind {kind} '
                                 f'for {key!r} on object {object_id}')
        return props

    def __repr__(self):
        return (f'MCOSSubsystem(version={self.version}, '
                f'classes={[self.class_name(i) for i in range(1, self.n_classes + 1)]}, '
                f'objects={self.n_objects})')
