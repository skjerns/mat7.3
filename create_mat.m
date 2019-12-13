data = struct
data.int16 = int16(16)
data.bool = boolean(0)
data.single = single(0.1)
data.double = double(0.1)
data.char = char('x')
data.arr_bool = boolean([1,1,0])
data.arr_float = single([1.1,1.2,0.3;2,3,4])
data.arr_double = double([1.1,1.2,0.3])
data.arr_char = char('test')
data.arr_nan = [NaN,NaN]
data.ab = {'third', 'fourth'}
data.aa = {'first', 'second'}
data.nan = NaN
data.complex = complex(2)
data.cell_char = {'Smith','Chung','Morales'; 'Sanchez','Peterson','Adams'}
data.cell = {double([1.1,2.2]), boolean(0), boolean([0,1]),1.1,0,'test', {'subcell', 0}}

save('data.mat', 'data','-v7.3')