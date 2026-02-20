% verify_savemat.m
%
% MATLAB-side verification of files produced by mat73.savemat.
%
% Workflow:
%   1. Run:  python tests/create_savemat_testfiles.py
%   2. In MATLAB, cd to the project root and run:  tests/verify_savemat
%
% Each block prints "PASS" or throws an error on failure.

clear; clc;
load(fullfile(fileparts(mfilename('fullpath')), 'savemat_verify.mat'));

passed = 0;
failed = 0;

function check(label, cond)
    if cond
        fprintf('  PASS  %s\n', label);
    else
        fprintf('  FAIL  %s\n', label);
        error('Assertion failed: %s', label);
    end
end

fprintf('\n=== Python scalars ===\n');
check('scalar_int  == 42',             scalar_int == 42);
check('scalar_int  class double',      isa(scalar_int,  'double'));
check('scalar_float == 3.14',          abs(scalar_float - 3.14) < 1e-10);
check('scalar_neg  == -1.5',           scalar_neg == -1.5);
check('scalar_bool_true  == true',     scalar_bool_true  == true);
check('scalar_bool_true  class logical', isa(scalar_bool_true,  'logical'));
check('scalar_bool_false == false',    scalar_bool_false == false);
check('scalar_bool_false class logical', isa(scalar_bool_false, 'logical'));
check('scalar_str  == ''hello''',      strcmp(scalar_str, 'hello'));
check('scalar_nan  is NaN',            isnan(scalar_nan));
check('scalar_inf  is Inf',            isinf(scalar_inf) && scalar_inf > 0);
check('scalar_none is empty',          isempty(scalar_none));

fprintf('\n=== NumPy scalars (dtype preservation) ===\n');
check('np_float64 value',   abs(np_float64 - 1.23) < 1e-10);
check('np_float64 class',   isa(np_float64, 'double'));
check('np_float32 value',   abs(np_float32 - 1.5) < 1e-5);
check('np_float32 class',   isa(np_float32, 'single'));
check('np_int8  value',     np_int8  == 10);
check('np_int8  class',     isa(np_int8,  'int8'));
check('np_int16 value',     np_int16 == 1000);
check('np_int16 class',     isa(np_int16, 'int16'));
check('np_int32 value',     np_int32 == 99999);
check('np_int32 class',     isa(np_int32, 'int32'));
check('np_int64 value',     np_int64 == 1e9);
check('np_int64 class',     isa(np_int64, 'int64'));
check('np_uint8  value',    np_uint8  == 255);
check('np_uint8  class',    isa(np_uint8,  'uint8'));
check('np_uint16 value',    np_uint16 == 60000);
check('np_uint16 class',    isa(np_uint16, 'uint16'));
check('np_uint32 value',    np_uint32 == 1e6);
check('np_uint32 class',    isa(np_uint32, 'uint32'));
check('np_uint64 value',    np_uint64 == 1e12);
check('np_uint64 class',    isa(np_uint64, 'uint64'));
check('np_bool   value',    np_bool == true);
check('np_bool   class',    isa(np_bool, 'logical'));

fprintf('\n=== NumPy arrays (dtype preservation) ===\n');
check('arr_float64 values', isequal(arr_float64, [1.0 2.0 3.0]));
check('arr_float64 class',  isa(arr_float64, 'double'));
check('arr_float32 values', isequal(arr_float32, single([1.0 2.0 3.0])));
check('arr_float32 class',  isa(arr_float32, 'single'));
check('arr_int8  values',   isequal(arr_int8,  int8([-1 0 1])));
check('arr_int8  class',    isa(arr_int8,  'int8'));
check('arr_int16 values',   isequal(arr_int16, int16([100 200 300])));
check('arr_int16 class',    isa(arr_int16, 'int16'));
check('arr_int32 values',   isequal(arr_int32, int32([1000 2000 3000])));
check('arr_int32 class',    isa(arr_int32, 'int32'));
check('arr_int64 values',   isequal(arr_int64, int64([1e9 2e9])));
check('arr_int64 class',    isa(arr_int64, 'int64'));
check('arr_uint8  values',  isequal(arr_uint8,  uint8([0 128 255])));
check('arr_uint8  class',   isa(arr_uint8,  'uint8'));
check('arr_uint16 values',  isequal(arr_uint16, uint16([0 1000 65535])));
check('arr_uint16 class',   isa(arr_uint16, 'uint16'));
check('arr_uint32 values',  isequal(arr_uint32, uint32([0 1e6])));
check('arr_uint32 class',   isa(arr_uint32, 'uint32'));
check('arr_uint64 values',  isequal(arr_uint64, uint64([0 1e12])));
check('arr_uint64 class',   isa(arr_uint64, 'uint64'));
check('arr_bool values',    isequal(arr_bool, logical([1 0 1])));
check('arr_bool class',     isa(arr_bool, 'logical'));
check('arr_nan(1) == 1',    arr_nan(1) == 1.0);
check('arr_nan(2) is NaN',  isnan(arr_nan(2)));
check('arr_nan(3) == 3',    arr_nan(3) == 3.0);
check('arr_complex values', isequal(arr_complex, [1+2i, 3+4i]));
check('arr_complex class',  isa(arr_complex, 'double'));

fprintf('\n=== Array shapes ===\n');
check('arr_2d size',    isequal(size(arr_2d), [2 3]));
check('arr_2d values',  isequal(arr_2d, [1 2 3; 4 5 6]));
check('arr_3d size',    isequal(size(arr_3d), [2 3 4]));
check('arr_3d(1,1,1)',  arr_3d(1,1,1) == 0);
check('arr_3d(2,3,4)',  arr_3d(2,3,4) == 23);
check('arr_row size',   isequal(size(arr_row), [1 3]));
check('arr_row values', isequal(arr_row, [10 20 30]));
check('arr_col size',   isequal(size(arr_col), [3 1]));
check('arr_col values', isequal(arr_col, [10; 20; 30]));

fprintf('\n=== Python lists ===\n');
check('list_ints values',   isequal(list_ints,   [1 2 3]));
check('list_ints class',    isa(list_ints, 'double'));
check('list_floats(2)',     abs(list_floats(2) - 2.2) < 1e-10);
check('list_nested size',   isequal(size(list_nested), [2 2]));
check('list_nested values', isequal(list_nested, [1 2; 3 4]));
check('list_strs is cell',  iscell(list_strs));
check('list_strs size',     isequal(size(list_strs), [1 3]));
check('list_strs{1}',       strcmp(list_strs{1}, 'foo'));
check('list_strs{2}',       strcmp(list_strs{2}, 'bar'));
check('list_strs{3}',       strcmp(list_strs{3}, 'baz'));

fprintf('\n=== Python dict (struct) ===\n');
check('mystruct is struct',         isstruct(mystruct));
check('mystruct.double_field == 42',abs(mystruct.double_field - 42.0) < 1e-10);
check('mystruct.str_field',         strcmp(mystruct.str_field, 'world'));
check('mystruct.arr_field values',  isequal(mystruct.arr_field, [10 20 30]));
check('mystruct.int_field == 7',    mystruct.int_field == int32(7));
check('mystruct.int_field class',   isa(mystruct.int_field, 'int32'));

fprintf('\n=== Nested struct ===\n');
check('nested_struct is struct',        isstruct(nested_struct));
check('nested_struct.outer_val == 1',   nested_struct.outer_val == 1.0);
check('nested_struct.inner is struct',  isstruct(nested_struct.inner));
check('nested_struct.inner.inner_val',  nested_struct.inner.inner_val == 2.0);
check('nested_struct.inner.inner_str',  strcmp(nested_struct.inner.inner_str, 'deep'));

fprintf('\n=== All checks passed ===\n');
