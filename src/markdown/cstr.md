# CSP Equation


$$ \frac{d\textbf{y}}{dt} = \textbf{g}(\textbf{y,z}), \quad \textbf{y}(t=0) = \textbf{y}_0$$

$$\textbf{f}(y,z)=0, \quad \textbf{z}(t=0)=\textbf{z}_0$$

$$\textbf{g}(\textbf{y,z}) = \textbf{G}(y)$$

$$ J_{ij} = \frac{\partial G_i}{\partial y_j} $$

$$ J_{ij} = \frac{\partial g_i(y,z)}{\partial y_j} \Big|_z + \sum _{k=1}^m \frac{\partial f_i(y,z)}{\partial z_k} \Big|_y \frac{ \partial  z_k}{\partial  y_j} $$

$$A_{ls} = \frac{ \partial  f_l}{\partial  z_s} \Big|_y, \quad B_{lj} = \frac{ \partial  f_l}{\partial  y_j} \Big|_z, \quad X_{sj]}=\frac{ \partial  z_s}{\partial  y_j} $$

$$ \textbf{A}\textbf{X} =\textbf{B} $$

#  CSP - A Software Toolkit for the Analysis of Complex Kinetic Models

 ## Mechanism of Methanol Synthesis on Cu through CO2 and CO Hydrogenation (    L. C. Grabow and M. Mavrikakis).


This paper presents a reaction mechanism of 49 surface reactions. It does not provided Chemkin inputs files or follows a general surface kinetics formalism. It uses a DTF methodology to obtained thermal properties (entropy, enthalpy and cp) of gas species and surface species.  


There a several difference between in the modeling of this paper and our current implementation of surface reaction rate.

* The coefficients to interpolate entropy, enthalpy and cp uses a Shomate equation. In TChem we use a NASA polynomials.

* The rate of progress is computed by:

$$q_i={k_f}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu'_{ji}}-
{k_r}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu''_{ji}}$$   

The forward rate constant is computed by:

$$
{k_f}_i=A_iT^{\beta_i}\exp\left(-\frac{E_i}{RT}\right),
$$

The reverse rate constant by:

$$
{k_r}_i={k_f}_i/{K_c}_i,
$$

In Grabow and Mavrikakis, $\mathfrak{X}$ is partial pressure in atm in gas species, and surface coverages (dimensionless) in surfaces species. The unit of $q_i$ is $\frac{mol}{site\ sec}$

In TChem, $\mathfrak{X}$ is molar concentration in $\frac{kmol}{m^3}$ for gas species and  the product of site fraction($Z_k$ dimensionless ) and the site density ($\Gamma$  $\frac{kmol}{m^2}$)  $Z_k \Gamma$. The unit of $q_i$ is $\frac{kmol}{m^2 \ s}$

The difference in $\mathfrak{X}$ makes that ${k_f}_i$ have different units in TChem.

For example for the reaction $ CO_2  + X <=> CO_2X  $, $q_i\ (mol/site/sec)$ is equal to:

$$q_i={k_f}_i P_{Co_2}Z_{X} - {k_r}_i Z_{CO_2X }$$

Therefore, units of ${k_f}_i$ are $\frac{mol}{atm\ site \ sec}$ and the units of {k_r}_i are $\frac{mole}{ site \ sec}$.


To make units consistent in TChem we need to times ${k_r}_i$ or $A$ by:

$F = \frac{RT}{101325*10^3}\frac{1}{\Gamma} \frac{site}{Area_{cat}} $

* Equilibrium constant

The equilibrium constant is computed by Grabow and Mavrikakis with:

$$K_{p.i} = exp\Big(\frac{\Delta S^o_i}{R} -  \frac{\Delta H^o_i}{RT} \Big)$$

In TChem an aditional factor is included:

$$K_{c,i} = K_{p,i} \Big( \frac{p^o}{RT} \Big)^{\sum_k=1 ^{Kg}\nu_{ki}} \prod_{n=N_s^f}^{N_s^l} (\Gamma_n^o)^{\Delta \sigma_(n,i)} $$  


<!-- $$A = \frac{kB}{h}T \exp(\Delta S_{TS}/R) $$ -->

## CSTR





$\frac{d(\theta_k)}{dt} = \dot{s}_k^{'}$   [mole/sites/s]

$F_{out} = F_{in} + \sum_{k=1}^{kg} \dot{s}_k^{'}$  [mole/sites/s]

$\frac{dp_k}{dt} = (\dot{s}_k^{'} + F_{in}*X_k - \frac{F_{out}}{P} )\frac{RT}{V}\frac{Sites}{p^{o}} $  [atm/s]

$\dot{s}_k^{'} = \sum_{i=1}^{Nr}\nu_{ki}q_i$  [mole/sites/s]

$q_i = k_{ifwd}\prod_{k=1}^{Ns}a_k^{\nu_{ki}^{'}} - k_{irev}\prod_{k=1}^{Ns}a_k^{\nu_{ki}^{''}}$   [mole/sites/s]

$a=\theta$ site fraction [mole/sites]

$a= p$ partial pressure [atm]


## CSTR

In TChem, we are resolving this system of equation for transient CSTR. We assume constant density, temperature and pressure.

$$    \dot{m}(t) = A\sum_{k=1}^{Kg} \dot{s}_kW_k + \dot{m^*}(t)$$

$$\rho V \frac{ d ( Y_k )}{dt} =  \dot{m^*} (Y_{k}^{*} - Y_k) - Y_k A\sum_{k=1}^{Kg} \dot{s}_kW_k + A \dot{s}_kW_k +V w_{k}W_k$$

$$ \frac{d(Z_k)}{dt} = \frac{\dot{s}_k}{\Gamma} $$





* ***overall mass continuity (eq 9.100 )***

$$\frac{ d (\rho V)}{dt}  - \dot{m^*}(t)+\dot{m}(t) = A\sum_{k=1}^{Kg} \dot{s}_kW_k$$

* ***individual-species continuity equation (eq 9.102)***

$$\rho V \frac{ d ( Y_k )}{dt} =  \dot{m^*} (Y_{k}^{*} - Y_k) - Y_k A\sum_{k=1}^{Kg} \dot{s}_kW_k + A \dot{s}_kW_k +V \dot{w}_{k}W_k$$

* ***energy equation eq (9.109)***
 $$  \frac{dT}{dt} =  - \frac{1}{c_p} \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  +    \frac{1}{\rho V c_p} (\dot{m^*}(h^{*} - h)  - hA\sum_{k=1}^{Kg} \dot{s}_kW_k ) $$


 * ***surface equation***

 $$ \dot{s}_k=0 $$

 * ***surface equation***

$$ \frac{d(Z_k)}{dt} = \frac{\dot{s}_k}{\Gamma} $$



##CSTR v2
* ***overall mass continuity (eq 9.100 )***
$$\frac{ d (\rho V)}{dt}  - \dot{m^*}(t)+\dot{m}(t) = A\sum_{k=1}^{Kg} \dot{s}_kW_k$$

* ***individual-species continuity equation (eq 9.102)***

$$\frac{ d (\rho Y_k V)}{dt} - \dot{m^*}Y_{k}^{*}+\dot{m}Y_{k} = A \dot{s}_kW_k +V w_{k}W_k$$

expanding term 1

$$\rho V \frac{ d ( Y_k )}{dt} +  Y_k\frac{ d ( \rho V )}{dt}  - \dot{m^*}Y_{k}^{*}+\dot{m}Y_{k} = A \dot{s}_kW_k +V w_{k}W_k$$

with continuity equation  


$$\rho V \frac{ d ( Y_k )}{dt} +  Y_k( \dot{m^*}(t)-\dot{m}(t) + A\sum_{k=1}^{Kg} \dot{s}_kW_k)  - \dot{m^*}Y_{k}^{*}+\dot{m}Y_{k} = A \dot{s}_kW_k +V w_{k}W_k$$

$$\rho V \frac{ d ( Y_k )}{dt} =  -Y_k\dot{m^*} + Y_k\dot{m} - Y_k A\sum_{k=1}^{Kg} \dot{s}_kW_k  + \dot{m^*}Y_{k}^{*}-\dot{m}Y_{k} + A \dot{s}_kW_k +V w_{k}W_k$$

$$\rho V \frac{ d ( Y_k )}{dt} =  \dot{m^*} (Y_{k}^{*} - Y_k) - Y_k A\sum_{k=1}^{Kg} \dot{s}_kW_k + A \dot{s}_kW_k +V w_{k}W_k$$

## energy equation eq (9.109)
$$\frac{ d (\rho h V)}{dt} - \dot{m^*}h^{*}+\dot{m}h = \hat{h} \Delta {T} + V\frac{dp}{dt} $$

with

$$ h = \sum_{k=1}^{Kg} h_k Y_K \quad  c_p = \sum_{k=1}^{Kg}c_{p_k} Y_K \quad  dh_k = c_{p_k} dT $$


$$\frac{ d (\rho h V)}{dt} = \rho V \frac{ d ( \sum_{k=1}^{Kg} h_k Y_K )}{dt} +  h\frac{ d (\rho  V)}{dt} $$


$$\frac{ d (\rho h V)}{dt} = \rho V \sum_{k=1}^{Kg} (Y_k c_{p_k} \frac{dT}{dt} + h_k \frac{dY_k}{dt} )  +  h\frac{ d (\rho  V)}{dt} $$

$$\frac{ d (\rho h V)}{dt} = \rho V c_p \frac{dT}{dt} +\rho V \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  +  h\frac{ d (\rho  V)}{dt} $$



$$\rho V c_p \frac{dT}{dt} =  - \rho V \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  -  h\frac{ d (\rho  V)}{dt} + \dot{m^*}h^{*}-\dot{m}h + \hat{h} \Delta {T} + V\frac{dp}{dt} $$

using continuity equation


$$\rho V c_p \frac{dT}{dt} =  - \rho V \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  -  h  \dot{m^*} + h\dot{m} -h  A\sum_{k=1}^{Kg} \dot{s}_kW_k + \dot{m^*}h^{*}-\dot{m}h + \hat{h} \Delta {T} + V\frac{dp}{dt} $$

 $$\rho V c_p \frac{dT}{dt} =  - \rho V \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  +    \dot{m^*}(h^{*} - h)  - hA\sum_{k=1}^{Kg} \dot{s}_kW_k + \hat{h} \Delta {T} + V\frac{dp}{dt} $$

 $$  \frac{dT}{dt} =  - \frac{1}{c_p} \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  +    \frac{1}{\rho V c_p} (\dot{m^*}(h^{*} - h)  - hA\sum_{k=1}^{Kg} \dot{s}_kW_k ) $$

* ***surface equation***

$$ d(Z_k) = \frac{\dot{s}_k}{\Gamma} $$


$
\dot{s}_k=\sum_{i=1}^{N_{reac}}\nu_{ki}q_i
$ [kmol/m$^2$/s]

$q_i={k_f}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu'_{ji}}-
{k_r}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu''_{ji}}$

$\mathfrak{X}_j=\frac{Y_j \rho}{W_j}$     molar concentration [kmol/m$^3$/s]

$\mathfrak{X}=\frac{Z_k\Gamma_n}{\sigma_{j,n}}$ surface concentration [kmol/m$^2$/s]


<!-- $$ V \frac{d \rho }{ dt} =- \rho\frac{dV}{dt} + \dot{m^*}-\dot{m} + A\sum_{k=1}^{Kg} \dot{s}_kW_k  $$ -->

<!-- ideal gas equation $$\rho = \frac{p\bar{W}}{RT}$$

$$\frac{dp}{dt} = R\bar{W}^{-1} \rho \frac{dT}{dt} + R\bar{W}^{-1}T \frac{d \rho }{ dt} + R \rho T \sum _{k=1}^{Kg} \frac{d Y_k}{ dt}\frac{1}{W_k}   $$

$$\bar{W}^{-1}  = \sum_{k=1}^{Kg} \frac{Y_k}{W_k}$$

$$\rho V \frac{ d ( Y_k )}{dt}  = -  Y_k\frac{ d ( \rho V )}{dt}  +  \dot{m^*}Y_{k}^{*}-\dot{m}Y_{k} + A \dot{s}_kW_k +V w_{k}W_k$$

$$\rho V c_p \frac{dT}{dt} =  - \rho V \sum_{k=1}^{Kg} (h_k \frac{dY_k}{dt} )  -  h\frac{ d (\rho  V)}{dt} + \dot{m^*}h^{*}-\dot{m}h + \hat{h} \Delta {T} + V\frac{dp}{dt} $$ -->

Temperature =
# CSP Index
$$
\frac{dy}{dt} = S_r (y)  R_r(y) + S_{rs}(y)  R_{rs}(y,z) \\
0 = C \cdot R_{rs}(y,z)
$$
$$\frac{dy}{dt} = g +c $$

$$ 0= d$$

$$ f_i = b^i (g+c) = \sum _{r=1}^{Rg} \beta_r^i R_r + \sum _{r=1}^{Rs} \zeta _r^i R_{sr}$$


$$ \beta_r^i = b^i S_r \quad \zeta_r^i = b^i S_{sr}$$

$$
g_{slow} = \sum _{i=M+1}^{N} a_i f^i =  \sum _{i=M+1}^{N} a_i\sum _{r=1}^{Rg} \beta_r^i R_r + \sum _{i=M+1}^{N} a_i \sum _{r=1}^{Rs} \zeta _r^i R_{sr}
$$


$$
 g_{slow} =  \sum _{r=1}^{Rg}\sum _{i=M+1}^{N} a_i \beta_r^i R_r +  \sum _{r=1}^{Rs} \sum _{i=M+1}^{N} a_i  \zeta _r^i R_{sr}
$$

$$
 g_{slow} =  \sum _{r=1}^{Rg}  \alpha_r^i R_r +  \sum _{r=1}^{Rs}  Z _r^i R_{sr}
$$

$$
 \alpha_r^i =  \sum _{i=M+1}^{N} a_i \beta_r^i
$$

$$
Z _r^i = \sum _{i=M+1}^{N} a_i  \zeta _r^i
$$

$$
(I^i_r)_{slow} = \frac{\alpha_r^i R_r}{\sum _{r=1}^{Rg}  | \alpha_r^i R_r| + \sum _{r=1}^{Rs}  |Z _r^i R_{sr}|}
$$


$$
(I^i_r)_{slow} = \frac{Z _r^i R_{sr}}{\sum _{r=1}^{Rg}  | \alpha_r^i R_r| + \sum _{r=1}^{Rs}  |Z _r^i R_{sr}|}
$$
