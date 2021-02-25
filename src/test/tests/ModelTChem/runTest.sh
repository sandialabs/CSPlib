
exec=$CSPlib_INSTALL_PATH/test/model_class/driver_model_ODE_TChem2
inputs=../data/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
inputfile=$inputs"input.dat"
prefix=model_output/

$exec --chemfile=$chemfile --thermfile=$thermfile --inputfile=$inputfile --prefix=$prefix

dir1="model_output"
dir2="model_output_ref"
echo ""
echo ""
echo "--------------------"
echo "TESTING"
echo "Comparing files with references..."

cd $dir2

refsize=0

for file in *.dat 
do
  diff -a --suppress-common-lines -y  ../$dir1/$file $file > testResults.txt 
  size=$(wc -c <"testResults.txt")
  if [ $size -gt $refsize ] 
    then 
    echo "TEST FAIL" $file
  else
    echo "TEST PASS" $file
  fi 

done

rm -rf ../testResults.txt

