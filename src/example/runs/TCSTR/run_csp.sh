exec=$CSPlib_INSTALL_PATH/example/indexTCSTRTChem/run_index_TCSTR_TChem.exe

inputs=inputs/
chemfile=$inputs"chemgri30.inp"
thermfile=$inputs"thermgri30.dat"
chemSurffile=$inputs"chemSurf.inp"
thermSurffile=$inputs"thermSurf.dat"
inputfile=$inputs"CSTRSolutionODE.dat"
samplefile=$inputs"sample_phi1.dat"
rtol=1e-3
atol=1e-13
Acat=1.347e-2
Vol=1.347e-1
mdotIn=1e-2
useAnalyticalJacobian=0
number_of_algebraic_constraints=$1
prefix=$2
verbose=true

$exec --verbose=$verbose --prefix=$prefix --numberOfAlgebraicConstraints=$number_of_algebraic_constraints --useAnalyticalJacobian=$useAnalyticalJacobian --samplefile=$samplefile --rtol=$rtol --atol=$atol --mdotIn=$mdotIn --Acat=$Acat --Vol=$Vol --inputfile=$inputfile --chemfile=$chemfile --thermfile=$thermfile --chemSurffile=$chemSurffile --thermSurffile=$thermSurffile
