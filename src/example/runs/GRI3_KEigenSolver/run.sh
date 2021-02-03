exec=$CSPlib_INSTALL_PATH/example/index_class/run_ODE_TChem_EigenK.exe
inputs=../GRI3/data/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
inputfile=$inputs"input.dat"
useTChemSolution=true
prefix=csp_output/
rtol=1e-8
atol=1e-14
$exec --useTChemSolution=$useTChemSolution --chemfile=$chemfile --thermfile=$thermfile --inputfile=$inputfile --rtol=$rtol --atol=$atol --prefix=$prefix
