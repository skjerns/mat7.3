clear
data = struct
data.int8_ = int8(2)
data.uint8_ = uint8(2)
data.uint16_ = uint16(12)
data.int16_ = int16(16)
data.int32_ = int32(1115)
data.uint32_ = uint32(5452)
data.int64_ = int64(65243)
data.uint64_= uint64(32563) 
data.bool_ = boolean(0)
data.single_ = single(0.1)
data.double_ = double(0.1)
data.char_ = char('x')
data.arr_bool = boolean([1,1,0])
data.arr_float = single([1.1,1.2,0.3;2,3,4])
data.arr_double = double([1.1,1.2,0.3])
data.arr_two_three = double([1,2;
                             3,4;
                             5,6])
data.arr_char = char('test')
data.arr_nan = [NaN,NaN]
data.nan_ = NaN
data.complex_ = complex(2, 3)
data.cell_char_ = {'Smith','Chung','Morales'; 'Sanchez','Peterson','Adams'}
data.cell_ = {double([1.1,2.2]), boolean(0), boolean([0,1]),1.1,0,'test', {'subcell', 0}}
data.string_ = 'tasdfasdf'
data.struct_ = struct('test', [1,2,3,4])
data.struct2_ = struct('type',{'big','little'},'color','red','x',{single([1.1,1.2,0.3;2,3,4]), double([1.1,1.2,0.3])})
secondvar = [1,2,3,4]


save('testfile1.mat','-v7.3')