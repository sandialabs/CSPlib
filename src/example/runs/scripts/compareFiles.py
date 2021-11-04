import numpy as np
import sys

args=sys.argv
if len(args) < 2:
   print('quit.. because less than 2 inputs')
   quit()

file1=args[1]
file2=args[2]

data1=np.loadtxt(file1).flatten()
data2=np.loadtxt(file2).flatten()

if len(data1) != len(data2) :
   print('quit.. because number of iterms in data is different between data1 and data2')
   quit()

max_abs_diff =0
max_rel_diff =0
for i in range(len(data1)):
   if( data1[i] != data2[i]):
     rel_error = (data1[i]- data2[i])/(data2[i]+1e-23)
     abs_error = (data1[i]- data2[i])
     if abs(rel_error) > max_rel_diff:
       max_rel_diff = rel_error  
     if abs(abs_error) > max_abs_diff:
       max_abs_diff=abs_error  

print('max relative error: ', max_rel_diff)
print('max abs error: ', max_abs_diff)


