# Building CSPlib

CSPlib requires Tines and Kokkos for the computation of the eigendecomposition on GPU or CPU hardware, and for linear algebra operations. Additionally, CSPlib has an interface to [TChem](https://github.com/sandialabs/TChem).

For convenience, we explain how to build the CSPlib code using the following environment variables that one can modify according to their working environments.

```bash
/// repositories
export CSP_REPOSITORY_PATH=/where/you/clone/csp/git/repo

/// build directories
export CSP_BUILD_PATH=/where/you/build/csp

/// install directories
export TCHEM_INSTALL_PATH=/where/you/install/tchem
export KOKKOS_INSTALL_PATH=/where/you/install/kokkos
export TINES_INSTALL_PATH=/where/you/install/tines

/// Tines requires OpenBlass
export LIBRARY_PATH=${LIBRARY_PATH}:=/where/you/install/OpenBlas/lib
```

## Download CSPlib
Clone the CSPlib repository. Instructions on how to download and install TChem, kokkos and Tines are found in the [TChem repository](https://github.com/sandialabs/TChem).

```bash
  git clone https://github.com/sandialabs/CSPlib ${CSP_REPOSITORY_PATH};
```

## Configuring CSPlib

The following example cmake script compiles CSPlib on the host, linking with Tines.

```bash
cmake \
    -D CMAKE_INSTALL_PREFIX=${CSP_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER="${my_cxx}" \
    -D CMAKE_C_COMPILER="${my_cc}" \
    -D KOKKOS_INSTALL_PATH=${KOKKOS_INSTALL_PATH} \
    -D TINES_INSTALL_PATH=${TINES_INSTALL_PATH} \
    ${CSP_REPOSITORY_PATH}/src
```

The following cmake example compiles CSPlib with TChem. CSPlib uses TChem to compute source terms, the Jacobian of the source term and the $S$ matrix and the rate of progress. TChem requires [Kokkos github pages](https://github.com/kokkos/kokkos) and Tines. Therefore, these libraries must also be installed.   

```bash
cmake \

    -D CMAKE_INSTALL_PREFIX=${CSP_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER="${my_cxx}" \
    -D CMAKE_C_COMPILER="${my_cc}" \
    -D OPENBLAS_INSTALL_PATH=${OPENBLAS_INSTALL_PATH}  \
    -D CSP_ENABLE_TCHEMPP=ON \
    -D TCHEM_INSTALL_PATH=${TCHEM_INSTALL_PATH}\
    -D KOKKOS_INSTALL_PATH=${KOKKOS_INSTALL_PATH} \
    -D TINES_INSTALL_PATH=${TINES_INSTALL_PATH} \
    ${CSP_REPOSITORY_PATH}/src
make install   
```

TChem is designed and implemented using Kokkos (a performance portable parallel programming model), thus, CSPlib can also carry out computation on a GPU. For GPUs, we can use the above cmake script  and replace the compiler choice by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``.
