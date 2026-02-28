clear
data = {}
data.int8_ = int8(2)
data.uint8_ = uint8(2)
data.uint16_ = uint16(12)
data.int16_ = int16(16)
data.int32_ = int32(1115)
data.uint32_ = uint32(5452)
data.int64_ = int64(65243)
data.uint64_= uint64(32563) 
data.bool_ = logical(0)
data.single_ = single(0.1)
data.double_ = double(0.1)
data.char_ = char('x')
data.arr_bool = logical([1,1,0])
data.arr_float = single([1.1,1.2,0.3;2,3,4])
data.arr_double = double([1.1,1.2,0.3])
data.arr_two_three = double([1,2;
                             3,4;
                             5,6])
data.arr_char = char('test')
data.arr_nan = [NaN,NaN]
data.nan_ = NaN
data.missing_ = missing
data.complex_ = complex(2, 3)
data.complex2_ = complex(123456789.123456789, 987654321.987654321)
data.complex3_ = complex(8.909089035006170e-04, 0.000000000000000e+00)
data.cell_char_ = {'Smith','Chung','Morales'; 'Sanchez','Peterson','Adams'}
data.cell_ = {double([1.1,2.2]), logical(0), logical([0,1]),1.1,0,'test', {'subcell', 0}}
data.string_ = 'tasdfasdf'
data.struct_ = struct('test', [1,2,3,4])
data.struct2_ = struct('type',{'big','little'},'color','red','x',{single([1.1,1.2,0.3;2,3,4]), double([1.1,1.2,0.3])})
data.structarr_ = struct('f1', {'some text'; [10,20,30]; magic(5)}, 'f2', {'v1'; 'v2'; 'v3';});
data.sparse_ = sparse([2, 4], [5, 8], [6, 7], 10, 8);

secondvar = [1,2,3,4]

keys = 'must_not_overwrite'

save('testfile1.mat','-v7.3')

clear all

%% second created file for https://github.com/skjerns/mat7.3/pull/57

A = sparse([0 0 0; 0 0 0]);
save('testfile13.mat','-v7.3')
clear all
%% third file: create a file to test all kind of empty dimensions that migh pop up

x_0 = rand(0)
x_1 = rand(1)
x_10 = 1:10
x_1_0 = rand(1, 0)
x_0_1 = rand(0, 1)
x_1_1 = rand(1, 1)
x_0_10 = rand(0, 10)
x_1_10 = rand(1, 10)
x_10_0 = rand(10, 0)
x_10_1 = rand(10, 1)
x_10_10 = rand(10, 10)
x_1_1_10_1_1 = rand(1, 1, 10, 1, 1)
x_10_1_1_10 = rand(10, 1, 1, 10)
save('testfile15.mat','-v7.3')
clear all


%% file to test 2D char arrays
char_arr_1d = ['abcd']
char_arr_2d = [ ...
    'PSTH tensor for image sequences (averaged across frames):'; ...
    'dimension 1: 2 scales (zoom1x, zoom2x)                   '; ...
    'dimension 2: 3 category (natural, synthetic, contrast)   '; ...
    'dimension 3: 10 movies                                   '; ...
    'dimension 4: sorted units                                '; ...
    'dimension 5: PSTH time bins                              '];

char_arr_3d = cat(3, ...
    ['abcd'; 'defg'], ...  % First "page"
    ['ghij'; 'jklm'], ...  % Second "page"
    ['mnöp'; 'pqrs'])     % Third "page"

save('testfile16.mat','char_arr_1d', 'char_arr_2d', 'char_arr_3d', '-v7.3')
clear all

%% file to test datetime loading
% Scalar datetime
dt_scalar = datetime(2023, 10, 26);

% Row vector of datetimes
dt_row_vector = [datetime(2023, 11, 1), datetime(2023, 11, 15), datetime(2023, 12, 25)];

% Column vector of datetimes
dt_column_vector = [datetime(2024, 1, 10); datetime(2024, 2, 20); datetime(2024, 3, 30)];

% 2x2 matrix of datetimes
dt_matrix = [datetime(2025, 1, 1), datetime(2025, 2, 1); ...
             datetime(2025, 3, 1), datetime(2025, 4, 1)];

% Datetime with specific time including milliseconds
dt_specific_time = datetime(2023, 3, 15, 14, 30, 45, 678);

% NaT (Not-a-Time)
dt_nat = NaT;

% Array with NaT values
dt_nat_array = [datetime(2023, 1, 1), NaT, datetime(2023, 1, 3)];

% Datetime with timezone (will be converted to UTC when stored)
dt_with_timezone = datetime(2023, 10, 26, 10, 0, 0, 'TimeZone', 'America/New_York');

% Past date (before Unix epoch)
dt_past_scalar = datetime(1500, 1, 1);

% Future date
dt_future_scalar = datetime(2500, 1, 1);

% Struct containing datetime fields
dt_struct.scalar = datetime(2026, 7, 4);
dt_struct.array = [datetime(2026, 8, 1); datetime(2026, 9, 15)];
dt_struct.mixed_datetime_in_struct_array.d = [datetime(2026, 10, 1); datetime(2026, 11, 1)];

% Cell array with datetime values
dt_cell_array = {datetime(2027, 5, 18), [datetime(2027, 6, 1), datetime(2027, 7, 1)]};

% Cell array column with datetime values
dt_cell_array_column = {[datetime(2027, 8, 1)]; [datetime(2027, 9, 1)]};

save('testfile_datetime.mat', 'dt_scalar', 'dt_row_vector', 'dt_column_vector', ...
     'dt_matrix', 'dt_specific_time', 'dt_nat', 'dt_nat_array', 'dt_with_timezone', ...
     'dt_past_scalar', 'dt_future_scalar', 'dt_struct', 'dt_cell_array', ...
     'dt_cell_array_column', '-v7.3')
clear all
