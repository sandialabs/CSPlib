# CSP Basic Concepts
## Formulation
Consider the autonomous ODE system in $\mathbb{R}^N$:  

$$ \frac{d\bm{y}}{dt} = \bm{g}(\bm{y})$$

With the initial value  $\bm{y}(t=0) = \bm{y}_0$.

Where $\bm{y}$ is a vector of state variables. For example, for a chemical kinetic model in a homogeneous gas phase constant pressure system, this can be comprised of the gas temperature and the mass fractions for the gas species. The right hand side (RHS) $\bm{g}(\bm{y})$ vector is a function of the state vector $\bm{y}$.

CSP analysis is primarily useful in the context of stiff dynamical systems exhibiting a wide range of fast/slow time scales. The goal of the analysis is to decouple fast and slow processes, thereby enabling specific dynamical diagnostic capabilities, by rewriting the system RHS using a suitable set of basis vectors [Lam 1993](https://www.tandfonline.com/doi/abs/10.1080/00102209308924120). CSP analysis seeks a set of basis vectors $\bm{a}_i$, $i=1,\ldots,N$, that linearly expand $\bm{g}$[Lam 1993](https://www.tandfonline.com/doi/abs/10.1080/00102209308924120):

$$\bm{g} = \sum_{i=1}^N\bm{a}_if^i \qquad (1)$$

where $f^i$ is the (signed) "amplitude" of $\bm{g}$ as projected on the basis vector $\bm{b}_i$,
$$ f^i = \bm{b}^i \cdot \bm{g} \qquad (2)$$

where the $\bm{b}^i$ vectors are, by construction, orthonormal to the $\bm{a}_i$ vectors.

$$ \bm{b}^i \cdot \bm{a}_i =\delta_j^i \qquad (3)$$

Given the $\bm{a}_i$ CSP basis vectors, the associated co-vectors $\bm{b}^i $ are computed using the orthonormality constraint (Eq. 3), and mode amplitudes $f^i$ (Eq. 2). CSP provides a refinement procedure to construct the basis vectors $\bm{a}_i$ [Lam 1993](https://www.tandfonline.com/doi/abs/10.1080/00102209308924120), [Valorani 2001](https://www.sciencedirect.com/science/article/pii/S0021999101967099). Alternatively, the right eigenvectors of the jacobian $J_{ij} = \frac{\partial g_i}{\partial y_j}$ provide a first order approximation of the ideal CSP $\bm{a}_i$ basis vectors. For a linear ODE system, the eigensolution perfectly decouples the fast and the slow time scales of $\bm{g}$. For a nonlinear system it provides only approximate decoupling. This library uses the Jacobian eigenvectors as the CSP basis vectors. Given that we are dealing with real, generally non-symmetric, Jacobian matrices, we can expect that any complex eigenvalues will be complex conjugate pairs, and similarly for the associated eigenvectors. When a pair of modes are complex conjugates, we do not use the complex eigenvectors as CSP basis vectors, rather we use two real eigenvectors that span the same plane. Thus we always have real CSP basis vectors.


We order the eigenmodes in terms of decreasing eigenvalue magnitude $|\lambda_i|$, Thus in order of decreasing time scales $\tau_i=1/|\lambda_i|$,
$$
\tau_1 < \tau_2 < \cdots < \tau_N
$$
so that mode 1 is the fastest mode, mode 2 is the next slower mode, etc.

Typically, chemical kinetic ODE models exhibit a number of fast decaying eigenmodes, associated with eigenvalues having large magnitudes (small timescales) with negative real components. These modes exhibit fast decay towards a slow invariant manifold developed from the equilibration of fast exhausted processes. Typical dynamics in systems that evolve towards an equilibrium involve a gradual increase in the number of fast exhausted modes, as successive time scales are exhausted, and the system approach the equilibrium point.  

At any point in time, presuming $M$ fast exhausted modes, we split $\bm{g}$ into slow and fast components:

$$\bm{g} = \underbrace{\sum_{i=1}^M\bm{a}_if^i}_{g_{fast}\approx 0} + \underbrace{\sum_{i=M+1}^N\bm{a}_if^i}_{g_{slow}} \qquad (4)$$

Thus, $M$ defines the dimension of the fast subspace. It is computed as the maximum $M$ for which

$$  \delta y^{i}_{fast} = \Big | \sum_{r=1}^{M} \bm{a}^{i}_r f^r\frac{e^{ \lambda^r\tau^{M+1}}-1 }{\lambda^r} \Big| < \delta y^{i}_{error} \qquad(5) $$

Where $\kappa=\mathrm{min}(M+1, N)$. Note that $\delta y^{i}_{\mathrm{error}}$ is critical to calculate $M$. We estimate $\delta y^{i}_{\mathrm{error}}$ with employing absolute and relative tolerances,

$$\delta \bm{y}_{error}= \mathrm{tol}_{\mathrm{relative}}|\bm{y}| + \mathrm{tol}_{\mathrm{absolute}} \qquad (6)$$

In equation [5], $\tau=\frac{1}{|\lambda|}$ is the time scale, and $\lambda$ is an eigenvalue.

With the CSP basis vectors we can also compute the CSP pointers. The CSP pointers identify the degree of orthogonality between the dimension of each species in the configuration space and the equation of state constraint developed out of the exhaustion of each of the fast modes [Lam 1993](https://www.tandfonline.com/doi/abs/10.1080/00102209308924120). The pointer for mode $i$ and species $j$ is defined as:
$$
\mathrm{CSPpointer}_{ij} = a_{ij} b_{ij} \qquad(7).
$$

The equations presented above outline the basics of CSP. Detailed mathematical derivations and description of the method are presented in [Lam 1985](https://link.springer.com/chapter/10.1007/978-1-4684-4298-4_1), [Lam 1989](https://www.sciencedirect.com/science/article/pii/S008207848980102X), [Lam 1991](https://link.springer.com/chapter/10.1007/BFb0035372), [Lam 1993](https://www.tandfonline.com/doi/abs/10.1080/00102209308924120). Application of CSP in combustion and other fields are presented in [Trevino  1988](https://www.sciencedirect.com/science/article/pii/S0082078406800196) [Goussis  1992](https://www.sciencedirect.com/science/article/pii/S0082078406800184) [Goussis 1995](https://www.sciencedirect.com/science/article/pii/S0021999196902090) [Ardema 1989](https://www.sciencedirect.com/science/article/pii/S1474667017519503) [Rao 1994](https://arc.aiaa.org/doi/abs/10.2514/6.1995-3262) [Goussis](https://arc.aiaa.org/doi/10.2514/6.1990-644) [Valorani 2001](https://www.sciencedirect.com/science/article/pii/S0021999101967099) [Valorani 2003](https://www.sciencedirect.com/science/article/pii/S0010218003000671) [Valorani 2005](https://www.sciencedirect.com/science/article/pii/S0021999105001981) [Valorani 2006](https://www.sciencedirect.com/science/article/pii/S0010218006001180) [Valorani 2015](https://www.sciencedirect.com/science/article/pii/S0010218015001534) [Malpica 2017](https://www.sciencedirect.com/science/article/pii/S0010218017300652) [Prager 2011](https://www.sciencedirect.com/science/article/pii/S0010218011001039) [Gupta 2013](https://www.sciencedirect.com/science/article/pii/S1540748912003690).  


## CSP Indices
The following definitions for CSP indices are relevant for an elementary reaction based chemical kinetic mechanism, involving $N_s$ species and $N_r$ reactions. The model is presumed to involve $N=N_s+1$ state variables, being the temperature $T$, and the mass fractions of the species.

We start by writing the RHS $\bm{g}$ as the product of the $N\times\Re$ matrix $S$, which is the generalized stoichiometric matrix, and the vector $[\mathcal{R}_1,\ldots,\mathcal{R}_{\Re}]$, where $\mathcal{R}_k$ is the rate of progress for elementary reaction $k$. By construction, we treat each reaction as reversible, thus we have $\Re=2N_r$ reactions. In this context, an irreversible reaction is assigned a zero-rate in the opposite direction. Thus, we write $\bm{g}$ as

$$\bm{g} = \sum _{k=1}^{\Re}S_k\mathcal{R}_k \qquad (8)$$

where $S_k$ is the $k$-th column of $S$.

The $S$ matrix is defined by $S=[Q\mathcal{S},Q\mathcal{S}]$, where  $\mathcal{S}$ is the $(N_s\times N_r)$ matrix of stoichiometric coefficients. For a constant pressure, homogeneous batch reactor, the $(N\times N_s)$ matrix $Q$ is defined by:

$$Q=
 \begin{bmatrix}
-\frac{1}{\rho c_p}W_{1}h_{1}             & -\frac{1}{\rho c_p}W_{2}h_{2}                    &   \cdots    & -\frac{1}{\rho c_p}W_{N_s}h_{N_s}  \\
\frac{1}{\rho}W_1     &    0          &    \cdots   &0 \\
0     &    \frac{1}{\rho}W_2          &    \cdots   &0 \\
\vdots            &    \vdots                  &    \vdots   &\vdots \\
0  &    0          &    \cdots   &\frac{1}{\rho}W_{N_{s}}
 \end{bmatrix}$$
where $\rho$ is density and $c_p$ is specific heat at constant pressure of the gas mixture, $h_k$ is the enthalpy of species $k$, and $W_k$ is the molar mass of species $k$.

The rate of progress is defined by $\mathcal{R}_k= [q_{\mathrm{fwd},1}, ..., q_{\mathrm{fwd},N_r}, -q_{\mathrm{rev},1}, ..., -q_{\mathrm{rev},N_r} ]$. Where $q_{\mathrm{fwd},k}$ and $q_{\mathrm{rev},k}$ are the forward and reverse rates of progress of reaction $k$.  


With the definition of the amplitude of the $i-th$ mode $f_i$.  

$$ f_i = \bm{b}^i\cdot\bm{g} = \sum _{k=1}^{\Re} \beta_k^i \mathcal{R}_k $$

$$ \beta_k^i = \bm{b}^i \cdot S_k$$

### CSP Slow Importance Index

The csp representation of the source term in the slow subspace is given by
$$
g_{\mathrm{slow}} = \sum _{i=M+1}^{N} \bm{a}_i f^i =  \sum _{i=M+1}^{N} \bm{a}_i\sum _{k=1}^{\Re} \beta_k^i \mathcal{R}_k = \sum _{k=1}^{\Re}  \alpha_k \mathcal{R}_k
$$
where
$$
 \bm{\alpha}_k =  \sum _{i=M+1}^{N} \bm{a}_i \beta_k^i
$$
and $\bm{\alpha}_k=(\alpha_k^1,\ldots,\alpha_k^N)$. The slow importance index of reaction $k$ with respect to state variable $j$ is defined as:
$$
(I^j_k)_{\mathrm{slow}} = \frac{\alpha_k^j \mathcal{R}_k}{\sum _{r=1}^{\Re}  | \alpha_r^j \mathcal{R}_r|} \quad (9)
$$

### CSP Fast Importance Index

The csp representation of the source term in the fast subspace is given by
$$
g_{\mathrm{fast}} = \sum _{i=1}^{M} a_i f^i =  \sum _{i=1}^{M} \bm{a}_i\sum _{k=1}^{\Re} \beta_k^i \mathcal{R}_k  
 =  \sum _{k=1}^{\Re}  \bm{\gamma}_k \mathcal{R}_k
$$
where
$$
\bm{\gamma}_k =  \sum _{i=1}^{M} \bm{a}_i \beta_k^i
$$
with $\bm{\gamma}_k =(\gamma_k^1,\ldots,\gamma_k^N)$. The fast importance index of reaction $k$ with respect to state variable $j$ is defined as:
$$
(I^j_k)_{\mathrm{fast}} = \frac{\gamma_k^j \mathcal{R}_k}{\sum _{r=1}^{\Re}  | \gamma_r^j \mathcal{R}_r|} \quad (10)
$$

### CSP Participation Index

The Participation Index of the $k-th$ reaction in the $i-th$ mode is defined as

$$P^i_k = \frac{\beta_k^i \mathcal{R}_k}{\sum _{r=1}^{\Re} |\beta_r^iR_r|  }\quad (11)$$
