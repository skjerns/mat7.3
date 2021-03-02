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
data.missing_ = missing(4)
data.complex_ = complex(2, 3)
data.complex2_ = complex(123456789.123456789, 987654321.987654321)
data.complex3_ = complex(8.909089035006170e-04, 0.000000000000000e+00)
data.cell_char_ = {'Smith','Chung','Morales'; 'Sanchez','Peterson','Adams'}
data.cell_ = {double([1.1,2.2]), logical(0), logical([0,1]),1.1,0,'test', {'subcell', 0}}
data.string_ = 'tasdfasdf'
data.struct_ = struct('test', [1,2,3,4])
data.struct2_ = struct('type',{'big','little'},'color','red','x',{single([1.1,1.2,0.3;2,3,4]), double([1.1,1.2,0.3])})
data.structarr_ = struct('f1', {'some text'; [10,20,30]; magic(5)}, 'f2', {'v1'; 'v2'; 'v3';});

secondvar = [1,2,3,4]

keys = 'must_not_overwrite'

save('testfile1.mat','-v7.3')