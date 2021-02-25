
exec=$CSPlib_INSTALL_PATH/example/index_class/run_index_ODE_TChem.exe
inputs=../data/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
inputfile=$inputs"input.dat"
useTChemSolution=true
prefix=csp_output/
rtol=1e-8
atol=1e-14

$exec --useTChemSolution=$useTChemSolution --chemfile=$chemfile --thermfile=$thermfile --inputfile=$inputfile --rtol=$rtol --atol=$atol --prefix=$prefix

dir1="csp_output"
dir2="csp_output_ref"
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

