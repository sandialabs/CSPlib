# make directory to save ouputs
mkdir outputs
Dir=outputs/Constraint
# input 1: number of constrains  2: first part of file name 
Nconstraints=11
#for ((i=0; i<=${Nconstraints}; i++)); do
for i in 0 4 11; do
  echo "...number of constraints ${i} "
  ./run_cspBatch.sh $i ${Dir}$i 
done


