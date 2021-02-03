exec=$CSPlib_INSTALL_PATH/example/kernel_class/driver_gODE_Davis_Skodje.exe
rtol=1e-4
atol=1e-14
y0=2.
z0=1.
tend=100.
nPoints=50000
$exec --tend=$tend --y0=$y0 --nPoints=$nPoints --z0=$z0 --rtol=$rtol --atol=$atol
