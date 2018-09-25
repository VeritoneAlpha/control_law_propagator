%% Jordan Makansi
% test script for calling python from matlab
clear all;
clc;
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

%N = py.list({'Jones','Johnson','James'});

% result = py.python_matlab_test.do_stuff('did some stuff');
result2 = py.python_matlab_test.do_stuff(8);
display(result2);

% shape = cellfun(@int64,cell(result2.shape));
ls = py.array.array('d',result2.flatten('F').tolist());
p = double(ls);