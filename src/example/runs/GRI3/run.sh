exec=$CSPlib_INSTALL_PATH/example/indexODETChem/run_index_ODE_TChem.exe
inputs=data/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
inputfile=$inputs"input.dat"
useTChemSolution=true
csp_outputs=csp_output
mkdir $csp_outputs
prefix=$csp_outputs/
rtol=1e-8
atol=1e-14
$exec --useTChemSolution=$useTChemSolution --chemfile=$chemfile --thermfile=$thermfile --inputfile=$inputfile --rtol=$rtol --atol=$atol --prefix=$prefix
