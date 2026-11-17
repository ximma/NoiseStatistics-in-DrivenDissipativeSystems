# Methods

PolaritonHorizons.jl computes the mean field, the linear response and the quantum noise of one-dimensional driven-dissipative bosonic fields, with exciton–polariton fluids in all-optically controlled geometries as the main application.
In this document we state the equations the package solves and name the functions that implement each step.
Two kinds of computation are used throughout.
Time integration of the stochastic Gross–Pitaevskii equation (GPE) contains every nonlinear effect and is expensive.
Linear theory on the same grid uses the same discretised operators, is exact for small fluctuations, and is solved by dense linear algebra without time stepping.
Each is used to check the other.

Lengths are in µm, times in ps and energies in meV.
Frequencies, including detunings, potentials and loss rates, are energies divided by $\hbar$ and are given in ps⁻¹, with $\hbar = 0.6582$ meV ps.
The field $\psi$ is in $\mu\text{m}^{-1/2}$, so that $|\psi|^2$ is a linear density, and pump amplitudes are in $\text{ps}^{-1}\,\mu\text{m}^{-1/2}$.
All frequencies are measured in the frame rotating at the pump frequency $\omega_p$.
A component $e^{i(kx-\omega t)}$ appears at $(+k, +\omega)$ in every spectrum computed by the package: spatial transforms are `fftshift(fft(·))` on the axis `kaxis(grid)`, and demodulation at $\omega$ is $\sum_j f(t_j)\, e^{+i\omega t_j}$.

## 1. Model

### 1.1 The driven-dissipative GPE

The lower-polariton field $\psi(x,t)$ on a periodic grid of length $L$ and spacing $dx$ (`Grid(L, N)`) obeys, in the frame rotating at $\omega_p$ [5],

$$
i\partial_t\psi = \Big[e(-i\partial_x) - \delta_0 + V(x) + V_{\rm res}(x) - i\kappa(x) + g\big(|\psi|^2 - s/dx\big)\Big]\psi + F(x,t) + \sqrt{\kappa(x)/dx}\;\xi(x,t),
$$

with $\delta_0 = \omega_p - \omega_0$ the detuning from the bottom $\omega_0$ of the branch (`detuning0`).
The kinetic term $e(k)$ is evaluated spectrally.
It is the parabola $\hbar k^2/2m$ by default.
When the Rabi splitting `ΩR` of `Polariton` is finite, $e(k)$ is the lower-polariton dispersion

$$
e(k) = \tfrac12\Big[\Delta_C(k) - \sqrt{\Delta_C(k)^2 + \Omega_R^2}\Big] - (\text{same at } k=0), \qquad \Delta_C(k) = \Delta_{xc} + \frac{\hbar k^2}{2m_{\rm ph}},
$$

where $\Delta_{xc}$ is the cavity–exciton detuning and $m_{\rm ph} = |C_0|^2 m$, so that $m$ remains the effective mass at $k=0$.
The function `kinetic(pol, k)` returns $e(k)$, and `kinetic_d1` and `kinetic_d2` its first and second derivatives, which replace $\hbar k/m$ and $\hbar/m$ wherever a group velocity or a curvature enters.
The exciton mass is taken as infinite, and $g$ and $\gamma$ as independent of $k$; `photonic_fraction` gives the Hopfield coefficient $|C_k|^2$, and `inflection_k`, `max_velocity` and `wavevector_for_velocity` describe the bounded group velocity of the lower polariton.

The real potential $V(x)$ is a sum of Gaussian defects (`GaussianDefect`).
The amplitude damping is $\kappa(x) = \gamma/2 + \kappa_{\rm abs}(x)$, where $\gamma$ is the intensity loss rate of the polaritons and $\kappa_{\rm abs}$ is the sum of the edge absorbers (`EdgeAbsorber`), which remove outgoing waves before they wrap around the periodic box.
The potential $V_{\rm res}(x)$ describes a frozen excitonic reservoir, $V_{\rm res} = \beta g n$, which is slow on the time scales of the collective excitations and therefore enters both the mean field and the fluctuations as a static potential [3].
It is zero unless set with `remake(s; Vres=…)` or by the reservoir solvers of Sec. 1.3.
The coherent pump $F(x)$ is usually a step made of two plane waves, $F_{\rm up}e^{ik_{\rm up}x}$ upstream and $F_{\rm down}e^{ik_{\rm down}x}$ downstream, joined at $x_{\rm step}$, with optional sech-shaped boosts near the boundaries (`StepPump`, `SechBoost`, `pump_amplitude`).
The local pump wavevector sets the flow velocity and the pump amplitude sets the operating point on the bistability loop, $|F|^2 = n[(gn-\delta)^2 + \gamma^2/4]$ (`pump_intensity`, `sonic_pump_amplitude`, `turning_points`).
A weak probe $F_{\rm pr}e^{-(x-x_0)^2/\sigma^2}e^{i(kx-\omega t)}$ can be added (`GaussianProbe`, `probe_profile`), as can an arbitrary source $F_{\rm extra}(x,t)$.

The last term of the GPE is the truncated-Wigner noise, with $\langle\xi(x,t)\xi^*(x',t')\rangle = \delta_{xx'}\delta(t-t')$, and is switched on only for truncated-Wigner runs (Sec. 6).
The term $-gs/dx$ with $s=1$ (`wigner_shift=true`, the default) is the Wigner-ordering correction of the cubic nonlinearity: for symmetrically ordered fields, $\psi^\dagger\psi\psi$ corresponds to $(|\psi|^2 - 1/dx)\psi$ on the grid [7].
The same term is kept in deterministic runs and in the linear theory, so that all parts of the package describe one equation.
In a noise-free calculation it acts as an additional detuning $g/dx$.

A geometry is a `Setup`, assembled by `make_setup(pol, grid; ωp, pump, defects, absorbers)` and copied with modified fields by `remake`.
The function `evolve(s, u0, tspan; dt, …)` integrates the GPE from an $N$-vector or from an $N\times B$ batch of independent trajectories, and `Stepper`, `advance!` and `state` give a time-stepping iterator for loops that process the field on the fly.
Time integration uses the Strang split-step scheme of GeneralizedGrossPitaevskii.jl (https://github.com/marcsgil/GeneralizedGrossPitaevskii.jl), which runs on CPUs and GPUs (Sec. 6.4).
A start-up envelope $h(t) = 1 + Ae^{-t/\tau}$ multiplying the static pump (`envelope=(A, τ)`) helps the fluid reach the upper branch of the bistability loop.

### 1.2 Presets

The function `burkhard2026()` builds the transcritical geometry of Burkhard *et al.* [1]: $L = 800$ µm, $N = 2048$, $\hbar^2/2m = 1.29$ meV µm², $\hbar\gamma = 0.047$ meV, $\hbar g = 3\times10^{-4}$ meV µm, $\hbar\omega_0 = 1473.36$ meV and $\hbar\omega_p = 1473.85$ meV.
The pump has $k_{\rm up} = 0.27$ µm⁻¹ and $k_{\rm down} = 0.539$ µm⁻¹, a step at $x_d - 7$ µm, and an upstream boost; a Gaussian defect of depth 0.85 meV and width 0.75 µm sits at $x_d = 400$ µm, and an absorber of 4.5 meV sits at the box edge.
By default both pump wavevectors are shifted by $2\pi/L$, the convention of the simulations of arXiv:2511.12339, and `gauge=:nominal` uses the nominal values.
The options `defect=:attractive`, `:repulsive`, `:none` or a depth in meV, and `support=false` (no downstream pump), give the variants of the same flow, and `burkhard2026_probe` returns the probe of that work.

The function `gildeoliveira2026()` builds the geometry of Gil de Olivera *et al.* [2]: $L = 1024$ µm, $N = 512$, $m = 1/6$ meV ps² µm⁻², $\hbar\delta_0 = 0.49$ meV, a repulsive defect of 0.85 meV at $L/2$, absorbers at both edges, and pump wavevectors $0.148$ and $0.614$ µm⁻¹.
Its runs start with the envelope `gildeoliveira2026_envelope`, and `gildeoliveira2026_protocol()` returns the truncated-Wigner protocol of that work.

The function `lkb_gaas_polariton()` returns the parameters of the GaAs microcavity of Falque *et al.* [3]: $m^* = 7.0\times10^{-35}$ kg and $\hbar\gamma = 80$ µeV.
The interaction constant is not fixed by that work and only sets the density scale, since the dynamics of the excitations depend on $gn$.

Any other step-pumped transcritical flow is built by `transcritical_setup(pol, grid; ωp, x_d, k_up, k_down, F_up, F_down, defect, absorbers, …)`, with `above_sonic` setting the pump amplitudes as offsets from the sonic point $gn = \delta$ of each region.

### 1.3 Inverse design

With control over both the amplitude and the phase of the pump, any smooth target field $\psi_t(x)$ is an exact stationary solution of the GPE for the pump

$$
F(x) = -\Big[e(-i\partial_x) - \delta_0 + V + V_{\rm res} + g\big(|\psi_t|^2 - s/dx\big) - i\kappa\Big]\psi_t ,
$$

evaluated with the same spectral kinetic operator as the time stepper (`exact_pump(s, ψ)`).
A target is specified by a flow velocity $v(x)$ and a Bogoliubov sound speed $c(x)$ (`design_flow`).
The local wavevector $k(x)$ is the one with group velocity $e'(k) = v$, the phase is $\theta = \int k\,dx$, and the density follows from

$$
c^2 = e''(k)\big[(2+\beta)gn - \delta(v) + V\big], \qquad \delta(v) = \delta_0 - e(k) + g s/dx ,
$$

which reduces to $c^2 = (\hbar/m)(2gn - \delta_{\rm eff})$, with $\delta_{\rm eff} = \delta(v) - V - \beta gn$, for the parabola.
The total phase must wind by a multiple of $2\pi$ over the periodic box; `design_flow` enforces it by a constant wavevector shift on chosen indices, for example inside an absorber.
The helpers `waterfall` and `bump` give smooth velocity and sound-speed profiles, `lowpass_pump` band-limits the pump to the spatial frequencies an imaging system transmits, and `designed_setup` combines these steps and returns the setup with its target field.
When the reservoir follows the density, $V_{\rm res} = \beta g|\psi|^2$, the self-consistent stationary state is found by Newton's method on the stationary equation (`reservoir_newton`, with the Jacobian `reservoir_jacobian`), or by alternating evolution and reservoir updates (`reservoir_steady_state`).

## 2. Mean field and the local description

The stationary state is obtained by noise-free evolution from the empty cavity, or from a designed field, for a long time (`steady_state(s; T, dt)`).
Convergence is checked by `stationarity`, the relative change of $\psi$ over a further evolution; the fixed point of the split-step scheme moves by $O(dt^2)$, so the check uses the same $dt$.
The function `profiles(s, ψ0)` returns the density $n = |\psi_0|^2$, the local wavevector $\partial_x\theta$ (`wavevector`, from a centred phase difference), the velocity $v = e'(\partial_x\theta)$ (`velocity`), the effective detuning $\delta = \delta_0 - e(\partial_x\theta) + gs/dx - V_{\rm res}$, the speed $\sqrt{\hbar g n/m}$ (`sound_speed_gn`), the Bogoliubov speed $c_B = \sqrt{e''(\partial_x\theta)(2gn - \delta)}$ (`bogoliubov_speed`), and the Mach number $v/c_B$.
The function `horizon(s, ψ0; near)` locates the point where the flow turns from subsonic to supersonic.
We use this position only to place windows and apertures; the scattering of the fluctuations is a property of the whole stationary flow.

Far from the horizon the flow is homogeneous, and each region is described in the local-density approximation (LDA).
The function `local_fluid(s, ψ0, x_beg, x_end)` averages the mean field over a region and returns a `LocalFluid` with density $n$, velocity $v$, effective detuning $\delta$ and carrier wavevector $k_0$.
Its laboratory-frame Bogoliubov dispersion (`lda_omega(f, k, branch)`) is

$$
\omega_\pm(k) = vq \pm \mathrm{Re}\sqrt{\big(\varepsilon_q + gn - \delta\big)\big(\varepsilon_q + 3gn - \delta\big)}, \qquad q = k - k_0,\quad \varepsilon_q = \hbar q^2/2m ,
$$

with the positive-norm branch for `branch = +1` and the negative-norm branch for `-1` [1,2].
For the lower-polariton dispersion the Doppler term $vq$ becomes $[e(k) - e(2k_0-k)]/2$ and $\varepsilon_q$ becomes $[e(k) + e(2k_0-k)]/2 - e(k_0)$.
The roots of $\omega_\pm(k) = \omega$ (`lda_roots`) and their group velocities (`group_velocity`) give the asymptotic incoming and outgoing modes at each frequency.
The Hawking window is the frequency interval $\omega_{\min} < \omega < \omega_{\max}$, where $\omega_{\max}$ is the maximum of the downstream negative-norm branch (`omega_max`) and $\omega_{\min}$ is the minimum of the upstream positive-norm branch, its effective-mass gap (`omega_min`).
Inside this window the supersonic region supports propagating negative-norm modes at positive frequency, and the scattering matrix of the flow mixes positive- and negative-norm modes.

## 3. Linearised fluctuations

### 3.1 Bogoliubov operator and spectrum

Linearising the GPE around $\psi_0$ with $\delta\psi = u\,e^{-i\omega t} + v^*e^{+i\omega^* t}$ gives the eigenvalue problem $\omega(u,v) = M(u,v)$ with

$$
M = \begin{pmatrix} L & g\psi_0^2 \\ -g\psi_0^{*2} & -L^* \end{pmatrix}, \qquad L = e(-i\partial_x) - \delta_0 + V + V_{\rm res} - i\kappa + 2g|\psi_0|^2 - gs/dx
$$

(`bogoliubov_matrix(s, ψ0)`, with the circulant kinetic operator `kinetic_matrix`).
The phase of $\psi_0$ is not factored out, so $\omega$ is the pump-frame frequency of the laboratory field.
Since $M$ uses the same kinetic operator, potentials and damping as the time stepper, its eigenvalues are exactly the small-amplitude frequencies of the simulated model.
The function `bogoliubov_spectrum(s, ψ0)` diagonalises the $2N\times2N$ matrix and returns a `BogoliubovSpectrum` with frequencies $\omega$, mode functions normalised to $\sum(|u|^2 + |v|^2)dx = 1$, and the symplectic norm $\eta = \sum(|u|^2 - |v|^2)dx \in [-1,1]$.
The spectrum is symmetric under $(\omega, u, v)\mapsto(-\omega^*, v^*, u^*)$.
Uniform losses give $\mathrm{Im}\,\omega = -\gamma/2$ for every mode; deviations come from the absorbers and from pairs of complex frequencies of the conservative problem, which are either dynamical instabilities or zero-norm modes such as quasinormal modes (QNMs).
The function `find_qnm(sp, x_H; window, loc, zero_norm)` returns the candidate QNMs, the modes whose weight near the horizon exceeds `loc` (`mode_weight`) and whose norm satisfies $|\eta| <$ `zero_norm`.

### 3.2 Exact driven response and norm balance

For a probe $F_{\rm pr}(x)e^{-i\omega t}$, the first-order response $\delta\psi = u\,e^{-i\omega t} + v^*e^{+i\omega t}$ solves

$$
(\omega - M)\begin{pmatrix}u\\ v\end{pmatrix} = \begin{pmatrix}F_{\rm pr}\\ 0\end{pmatrix}.
$$

The constructor `LinearResponse(s, ψ0)` reduces $M$ to Hessenberg form once, at $O(N^3)$ cost, after which each frequency is an $O(N^2)$ shifted solve (`respond(lr, F, ω)` or `respond(lr, probe)`).
Here $u$ is the response at $+\omega$, on the laboratory wavevectors of the probe, and $v^*$ the response at $-\omega$.
This is the probe simulation of Sec. 7 without its time window, demodulation or splitting error.
The Green operator $G(\omega) = (\omega - M)^{-1}$ that appears here is also the one that propagates the quantum noise in Sec. 4.

For the parabolic dispersion the linearised equations imply the continuity equation

$$
\partial_x J = -2\kappa(x)\rho(x) + \text{source}, \qquad \rho = |u|^2 - |v|^2, \qquad J = \frac{\hbar}{m}\,\mathrm{Im}\big(u^*\partial_x u + v^*\partial_x v\big),
$$

where the probe is the only source.
The functions `norm_density`, `norm_current` and `dissipation` (which returns $2\int\kappa\rho\,dx$) evaluate the balance $J(x_b) - J(x_a) = -2\int_{x_a}^{x_b}\kappa\rho\,dx$ between two points with no source in between.
For a non-parabolic dispersion the current is not local, and the mode-resolved fluxes of Sec. 7.2 are used instead.

## 4. Quantum noise

### 4.1 Input ports

Every loss channel $c$ damps the field amplitude at rate $\kappa_c(x)$ and, by the fluctuation–dissipation theorem, injects its own input field $c_{\rm in}$ [4]:

$$
\partial_t\psi = -i\big[H[\psi] - i\kappa\big]\psi - iF + \sum_c\sqrt{2\kappa_c}\;c_{\rm in}(x,t), \qquad \kappa = \kappa_r + \kappa_{nr} + \kappa_{\rm abs}.
$$

The inputs are white, $[c_{\rm in}(x,t), c^\dagger_{\rm in}(x',t')] = \delta(x-x')\delta(t-t')$ and $\langle c^\dagger_{\rm in}c_{\rm in}\rangle = n_c\,\delta\delta$, with $\delta(x-x')\to\delta_{ij}/dx$ on the grid.
The radiative port, through the cavity mirrors, has rate $\kappa_r = f_r\gamma/2$ and vacuum input; the pump and the probe are the coherent part of this input.
The non-radiative excitonic port has rate $\kappa_{nr} = (1-f_r)\gamma/2$ and an input of occupation $n_x$, which describes the incoherent excitonic reservoir.
For a polariton, $\kappa_r = |C|^2\kappa_{\rm cav}$ and $\kappa_{nr} = |X|^2\kappa_{\rm exc}$, so $f_r = |C|^2\kappa_{\rm cav}/\kappa$.
The absorbers, $\kappa_{\rm abs} = \kappa - \gamma/2$, take vacuum input and stand in for the rest of an infinite cavity.
These ports are set by `NoiseModel(s; radiative_fraction=f_r, n_x)`, where $n_x$ may vary in space.

Linearising around $\psi_0$, the fluctuations $\Phi = (\delta\psi, \delta\psi^\dagger)$ obey $\partial_t\Phi = -iM\Phi + \Xi$.
In symmetric (Wigner) ordering, $\langle\Xi(t)\Xi(t')^\dagger\rangle = D\,\delta(t-t')$ with

$$
D = \mathrm{diag}(d, d)/dx, \qquad d = \kappa_r + \kappa_{nr}(1 + 2n_x) + \kappa_{\rm abs}
$$

(`diffusion`, `noise_diagonal`).
For $n_x = 0$, $d = \kappa$, and the noise is exactly the truncated-Wigner noise of the GPE of Sec. 1.

### 4.2 Lattice phonons

The excitonic fraction of the fluid also scatters off longitudinal acoustic phonons of the lattice through the deformation potential, $H = \hbar\int dx\,\psi^\dagger\psi\,\xi_{\rm ph}(x,t)$ with $\xi_{\rm ph}$ a Hermitian bath field, as shown by Frérot *et al.* [6].
Linearised around $\psi_0$ and integrated over the bath, in the Markov approximation apart from the frequency dependence of the rate and with the Kramers–Kronig shift absorbed in $g$, the bath adds a local self-energy and a local noise source to the Bogoliubov problem.
Per grid site and in Nambu form,

$$
\Sigma(\omega) = -\frac{i}{2}\,\mathrm{sgn}(\omega)\,R(x,|\omega|)\begin{pmatrix}1 & e^{2i\varphi_0}\\ -e^{-2i\varphi_0} & -1\end{pmatrix}, \qquad
D_{\rm ph}(\omega) = \frac{R(x,|\omega|)\,\big(n_T(|\omega|) + \tfrac12\big)}{dx}\begin{pmatrix}1 & -e^{2i\varphi_0}\\ -e^{-2i\varphi_0} & 1\end{pmatrix},
$$

where $\varphi_0$ is the phase of $\psi_0$ and $n_T$ the Bose occupation of phonons of energy $\hbar|\omega|$ at the cryostat temperature $T$ (`phonon_occupation`).
The pump-frame frequency $\omega$ is the energy exchanged with the lattice, since the lattice is at rest in the laboratory.
The damping is independent of $T$; the noise carries $T$.
A negative-norm excitation at $\omega > 0$, as in the supersonic region, is anti-damped by the bath, because creating it emits a phonon; this is the Landau criterion with respect to the lattice.
Spontaneous phonon emission feeds the negative-frequency branch even at $T = 0$.
The local rate is $R(x,\omega) = \text{strength}\times(2.5~\mu\text{eV}/0.19~\text{meV})\times g|\psi_0(x)|^2\times\text{shape}(\omega)$, the phonon scattering rate of Ref. [6] scaled with the local blueshift, with $\text{shape}(\omega) = (\omega/\omega_m)\,e^{(1-\omega^2/\omega_m^2)/2}$ and $\hbar\omega_m = 1.03$ meV (`PhononBath(; T, strength, ħωm)`, `phonon_shape`, `phonon_rates`, `phonon_self_energy`).
The parameter `strength` absorbs the material ratio of phonon rate to blueshift; `strength = 1` takes the ratio of Ref. [6].
With the bath, $G(\omega) = (\omega - M - \Sigma(\omega))^{-1}$ and $D\to D + D_{\rm ph}(\omega)$, and the output commutators remain canonical.

### 4.3 Baths and temperatures

A complete noise calculation includes all of these baths: the radiative and non-radiative ports, the absorbers, and the lattice phonons; the example `examples/uniform_fluid.jl` shows each contribution.
The occupation $n_x$ of the excitonic port is a parameter of its own; the cryostat temperature enters through the phonon occupation $n_T$.
The standard set of cryostat temperatures is 4 K, 2 K and 0.3 K.
The phonon-free functions below (`spectral_covariance`, `stationary_covariance`, `mode_state`, `self_homodyne`, `spontaneous_noise`) compute the contribution of the loss ports, which is the limit `strength = 0`, and are the building blocks of the full calculation.
The functions `phonon_covariance`, `PhononNoise` and `phonon_mode_state` add the phonon bath to them; `phonon_mode_state` accepts a list of temperatures, `Ts = [4.0, 2.0, 0.3]`, and shares one factorisation among them because the damping does not depend on $T$.
At present the self-homodyne functions of Sec. 5.2 take the loss ports only; the phonon contribution to those spectra follows from the same replacements of $G$ and $D$.

### 4.4 Spectral and equal-time covariances

With $\Phi(\omega) = \int dt\,e^{i\omega t}\Phi(t)$, the fluctuations are $\Phi(\omega) = iG(\omega)\Xi(\omega)$, and

$$
\langle\Phi(\omega)\Phi(\omega')^\dagger\rangle_{\rm sym} = 2\pi\delta(\omega-\omega')\,S(\omega), \qquad S(\omega) = G(\omega)\,D\,G(\omega)^\dagger
$$

(`spectral_covariance(s, ψ0, ω; noise)`).
The block $S_{11}$ is the fluctuation spectrum at the laboratory frequency $\omega_p + \omega$, $S_{22}$ the spectrum at $\omega_p - \omega$, and $S_{12}$ holds the correlations between $+\omega$ and $-\omega$, which include the Hawking pairs.
The stimulated response (the columns of $G$) and the noise ($GDG^\dagger$) are therefore built from the same operator.
The Hawking effect is the mixing of positive- and negative-norm modes by $G$, and each bath is converted into outgoing radiation by the same scattering matrix.
The stationary state is set by the loss baths through this input–output relation.

The equal-time covariance $C = \int S(\omega)\,d\omega/2\pi = \langle\Phi\Phi^\dagger\rangle_{\rm sym}$ solves the Lyapunov equation

$$
AC + CA^\dagger + D = 0, \qquad A = -iM,
$$

which `stationary_covariance(s, ψ0; noise)` solves with a Schur-based (Bartels–Stewart) method that avoids the ill-conditioned eigenvector basis of $M$.
It requires all eigenvalues of $M$ to have negative imaginary part.
For $\psi_0 = 0$, $C = 1/(2dx)$ for any $\kappa(x)$, absorbers included: the noise compensates the damping exactly.
With the phonon bath the self-energy depends on $\omega$, the Lyapunov equation does not apply, and `phonon_covariance(s, ψ0, bath)` integrates $G(D + D_{\rm ph})G^\dagger$ over a frequency grid that covers the Bogoliubov spectrum with a margin of many linewidths.

The normally ordered moments are $N(x,x') = \langle\delta\psi^\dagger(x)\delta\psi(x')\rangle = C_{11}^{T} - 1/(2dx)$ and $\mathcal{M}(x,x') = \langle\delta\psi(x)\delta\psi(x')\rangle = C_{12}$, collected in `Fluctuations(s, ψ0, C)`.
For linear transforms of the field $a = F_a\psi$ and $b = F_b\psi$, the identity in position space or the windowed Fourier transform `window_matrix` in momentum space, Wick's theorem gives

$$
G^{(2)}_{ab} - n_a n_b = 2\,\mathrm{Re}\big[\alpha_a\alpha_b^*N_{ab}\big] + 2\,\mathrm{Re}\big[\alpha_a^*\alpha_b^*\mathcal{M}_{ab}\big] + |N_{ab}|^2 + |\mathcal{M}_{ab}|^2,
$$

with $N_{ab} = F_a^*NF_b^T$, $\mathcal{M}_{ab} = F_a\mathcal{M}F_b^T$, $\alpha = F\psi_0$ and $n = |\alpha|^2 + \langle\delta^\dagger\delta\rangle$.
The function `bogoliubov_g2m1(fl; Fa, Fb)` returns $g_2 - 1 = (G^{(2)} - n_an_b)/(n_an_b)$, directly comparable with the truncated-Wigner estimator of Sec. 6.
Non-Gaussian corrections are absent by construction, and the comparison with truncated-Wigner runs measures them.

### 4.5 Input–output relation and normal ordering

Light detected through a mirror of rate $\kappa_d = f_{\rm out}\kappa_r \le \kappa_r$ is

$$
A_{\rm out}(x,t) = \sqrt{2\kappa_d}\,\psi(x,t) - A_{\rm in}(x,t)
$$

[4], with $f_{\rm out} = 1/2$ for a symmetric cavity seen in transmission (`OutputNoise(s, ψ0; noise, output_fraction)`).
With $K = \sqrt{2\kappa_d}$ in Nambu form, the symmetrised output spectrum is $S_{\rm out}(\omega) = KGDG^\dagger K + [1 - iKGK + iKG^\dagger K]/(2dx)$.
Output fields are free fields, $[c, c^\dagger] = 1$ for each normalised detected mode, so the normally ordered output spectrum is

$$
\mathcal{N}(\omega) = S_{\rm out}(\omega) - \frac{1}{2dx} = KGDG^\dagger K + \frac{i}{2dx}\,K\big(G^\dagger - G\big)K .
$$

This is the quantity from which all detected moments of Sec. 5 are computed.
Only adjoint solves $X = G(\omega)^\dagger KP$ for the detected profiles $P$ are needed, which gives $P^\dagger\mathcal{N}P = X^\dagger DX + i\big[(KP)^\dagger X - X^\dagger KP\big]/(2dx)$.

## 5. Detection

### 5.1 Detected modes

A detected mode $i$ is a normalised spatial profile $f_i$ ($\sum|f_i|^2dx = 1$) at the pump-frame frequency $\sigma_i\omega$, $\sigma_i = \pm1$, with amplitude $A_i = \sum_x dx\,f_i^*A_{\rm out}(x,\sigma_i\omega)$.
In the stationary state, fields at different frequencies are uncorrelated, so the only moments are

$$
n_{ij} = \langle A_i^\dagger A_j\rangle = p_j^\dagger\mathcal{N}(\sigma\omega)\,p_i \quad(\sigma_i = \sigma_j = \sigma), \qquad
m_{ij} = \langle A_iA_j\rangle = p_i^\dagger\mathcal{N}(\sigma_i\omega)\,\tilde p_j \quad(\sigma_j = -\sigma_i),
$$

with $p = dx\,(f, 0)$ and $\tilde p = dx\,(0, f^*)$.
These are the covariances of one spectral–temporal mode, which a balanced homodyne detector measures at zero analysis frequency with a local oscillator at $\omega_p + \sigma_i\omega$ in profile $f_i$.
Profiles at the same $\sigma$ must be orthonormal; `orthonormalize_modes` makes them so with the symmetric (Löwdin) transformation $F\to F(F^\dagger F\,dx)^{-1/2}$, which changes each profile as little as possible.
The function `aperture_mode(s, (xa, xb), k)` gives a plane wave on an aperture, with a complex $k$ for the spatially damped outgoing channels of Sec. 7.2.

The function `mode_state(on, modes, ω; output_scale, schur)` returns $(n, m)$ for `modes = [(f_i, σ_i), …]`.
It applies the resolvent either by Hessenberg solves on $M^\dagger$ or, with a precomputed `SchurResolvent(M)`, by triangular solves on the Schur form $M = ZTZ^\dagger$, which is cheaper when many modes or frequencies are needed.
The function `pixel_state` gives the same moments in the basis of single grid points at $\pm\omega$; every detectable mode in a region is a linear combination of these (`transform_modes`), so its entanglement bounds that of any choice of local-oscillator profiles there.
Detection losses, or a smaller detected mirror rate, scale $n$ and $m$ by the efficiency $\eta$ (`output_scale`).
With the phonon bath the same moments are returned by `phonon_mode_state(PhononNoise(on, bath), modes, ω; Ts)`.

### 5.2 Self-homodyne intensity noise

When a coherent probe is present, the coherent output $B = \sqrt{2\kappa_d}\beta$ has three components in the pump frame: $\psi_0$ at $\nu = 0$, $u$ at $\nu = +\omega_s$ (probe band) and $v^*$ at $\nu = -\omega_s$ (idler band), with $(u,v)$ from `respond` (`coherent_components`; the pump line is included only with `pump=true`, which models a detection path without a notch at $\omega_p$).
A `Detector(name, s, (xa, xb), (ka, kb))` is a hard aperture followed by a Fourier-plane slit that keeps the aperture's own Fourier modes $e_q = e^{ik_qx}/\sqrt{L_a}$, $k_q = 2\pi q/L_a$, in a band; it is an orthogonal projector $\Pi_j$, so no vacuum enters at the filters (`coefficients`, `project`).
To first order in the fluctuations, the photocurrent of detector $j$ is $\delta I_j(\Omega) = \sum_\nu w_{j\nu}^\dagger\Phi_{\rm out}(\nu+\Omega)$, with $w_{j\nu} = dx\,(\Pi_jB_c, 0)$ at $\nu = \omega_c$ and $dx\,(0, (\Pi_jB_c)^*)$ at $\nu = -\omega_c$.
Since different frequencies are uncorrelated, the cross-spectral matrix is

$$
S_{jk}(\Omega) = \sum_\nu w_{j\nu}^\dagger\,S_{\rm out}(\nu+\Omega)\,w_{k\nu},
$$

and shot noise is $S_{jj} = I_j$ (`self_homodyne(on, dets, components; Ω)`, which returns the mean fluxes $I$ and $S$).
The function `with_efficiency` applies a detection efficiency as a beamsplitter, $I\to\eta I$ and $S\to\eta_j\eta_kS + \delta_{jk}\eta_j(1-\eta_j)I_j$.
From $I$ and $S$ the package computes the Fano factor $S_{jj}/I_j$ (`fano`), the normalised correlation $S_{jk}/\sqrt{S_{jj}S_{kk}}$ (`intensity_correlation`), the conditional variance $(S_{jj} - S_{jK}S_{KK}^{-1}S_{Kj})/I_j$ after optimal subtraction of detectors $K$ (`conditional_variance`), and the balanced intensity-difference noise normalised to shot noise (`gemellity`); a value below 1 of the last two is a quantum signature.

The spontaneous light, Hawking pairs included, adds terms quadratic in the fluctuations.
For the modes $c_q$ of a `DetectionBasis`, `spontaneous_noise(on, basis, ω)` integrates $n_{qq'}(\omega)$ and $|n_{qq'}|^2 + |m_{qq'}|^2$ over a uniform frequency grid that must cover every resonance with margin, using the Schur form of $M$.
By Wick's theorem the photocurrents then have mean $I_j = I_{{\rm coh},j} + \sum_{q\in j}\int n_{qq}\,d\omega/2\pi$ and noise $S_{jk} = S_{{\rm lin},jk} + \delta_{jk}I_{{\rm sp},j} + \sum_{q\in j,q'\in k}\int(|n_{qq'}|^2 + |m_{qq'}|^2)\,d\omega/2\pi$ at $\Omega\to0$ (`spontaneous_terms`, `spontaneous_flux`).
The linear self-homodyne regime requires the coherent flux in each detector to dominate the spontaneous flux.

### 5.3 Entanglement criteria and mode optimisation

From $(n, m)$, `quadrature_covariance` builds the symmetrically ordered covariance $\sigma$ of the quadratures $x = (A + A^\dagger)/\sqrt2$, $p = (A - A^\dagger)/(i\sqrt2)$, for which the vacuum is $I/2$, and `symplectic_eigenvalues` returns its symplectic eigenvalues, which are at least $1/2$ for a physical state.
The logarithmic negativity of a bipartition, $E_N = \sum\max(0, -\log_2 2\tilde\nu)$ over the symplectic eigenvalues $\tilde\nu$ of the partially transposed covariance (`log_negativity(σ, A)`), is a sufficient condition for entanglement, and a necessary one when one side is a single mode.
For two modes at opposite frequencies, the Cauchy–Schwarz ratio $|m_{ij}|^2/(n_{ii}n_{jj}) > 1$ (`pair_ratio`) implies entanglement; `max_pair_ratio` maximises it over one combination of modes on each side, which reduces to the leading singular value of the whitened correlation matrix.
Since $E_N$ is invariant under local Bogoliubov transformations, combining each detected line with its mirror line at the opposite frequency gives the entanglement of the outgoing quasiparticles.

The function `principal_modes(n, groups)` diagonalises the occupations within each group of modes (one region at one $\sigma$) and drops near-vacuum modes, which carry no correlation since $|m_{ij}|^2\le n_i(n_j+1)$.
The function `optimise_modes(n, m, groups, A; k)` chooses $k$ orthonormal combinations in each group to maximise $E_N$ between the groups $A$ and the others.
Its objective is $-\log_2 2\tilde\nu_1 - \sum_{i>1}\min(0, \log_2 2\tilde\nu_i)$ over the sorted symplectic eigenvalues, which equals $E_N$ when $\tilde\nu_1 < 1/2$ and has no flat region otherwise.
It is maximised by a quasi-Newton (BFGS) search from starting points built from the singular vectors of the cross-correlations $m$ and from random restarts.

### 5.4 Probe-modulated noise

With a strong probe the coherent field is periodic, $\psi_c(t) = \psi_0 + \sum_ha_he^{-ih\omega t}$, and the fluctuations obey $\partial_t\Phi = -iM(t)\Phi + \Xi$ with $M(t) = M_0 + \sum_{h\ne0}M_he^{-ih\omega t}$, since the field-dependent part of the Bogoliubov matrix is local.
In frequency, with sidebands $\Phi_n = \Phi(\varepsilon + n\omega)$, the problem becomes $\sum_mL_{nm}\Phi_m = i\Xi_n$ with $L_{nm} = (\varepsilon + n\omega - M_0 - \Sigma(\varepsilon + n\omega))\delta_{nm} - M_{n-m}(1-\delta_{nm})$, and different sidebands of the input noise are independent.
Every formula of this section then carries over with $G\to L^{-1}$ on the sideband space; the new feature is that fields at $\nu$ and $\nu'$ are correlated whenever $\nu - \nu'$ is a multiple of $\omega$.
This Floquet extension is assembled from `bogoliubov_matrix`, `phonon_self_energy` and the noise matrices above and is not a separate exported function.

## 6. Truncated Wigner

### 6.1 Noise and initial state

In the truncated-Wigner approximation (TWA) [5,7], each trajectory obeys the GPE of Sec. 1 with the noise $\sqrt{\kappa(x)/dx}\,\xi$, which is the symmetric-order noise $D$ of Sec. 4.1 for vacuum inputs ($n_x = 0$).
The phonon bath is not included in the TWA.
A run starts either from the stationary state (`initial=:state`) or from the stationary state plus Wigner vacuum noise $\xi/\sqrt{2dx}$, with $\langle|\xi|^2\rangle = 1$ (`initial=:vacuum`).
The protocol (`TWAProtocol(; T, dt, envelope, initial, nsamples, Δt_sample)`) evolves each trajectory for a time $T$ and samples it once at the end, or, for stationary geometries, keeps it running and samples it every `Δt_sample`; successive samples are then correlated over about $1/\gamma$, and statistical estimates must count trajectories rather than samples.
The function `twa_batch` runs $B$ independent trajectories as the columns of one batch.

### 6.2 Online moments and Wigner-ordering corrections

The accumulator `TWAAccumulator(s, window_pairs; position)` keeps, for the field in position space and for each pair of spatial windows in momentum space, the Wigner moments $\langle|a|^2\rangle_W$, $\langle|b|^2\rangle_W$, $\langle ab^*\rangle_W$ and the co-moment of $(|a|^2, |b|^2)$ (`Moments`, `update!`).
Batches are combined by an exact pairwise update that uses only matrix products, so that runs on different machines can be merged.
The function `twa_run!(acc, s, ψ0; ntraj, batch, protocol)` accumulates `ntraj` trajectories in batches and can be resumed.
A windowed transform is $a(k_m) = \sum_nw_n\psi(x_n)e^{-ik_mn\,dx}$ with a periodic Hann window (`SpatialWindow`, `windowed_fft`, `window_k`).

Wigner averages are symmetrically ordered.
For two windowed field operators with commutators $c_{aa}$, $c_{bb}$ and $c_{ab}(j,k) = [a_j, b_k^\dagger]$, the normally ordered quantities are [2]

$$
n_a(j) = \langle|a_j|^2\rangle_W - \frac{c_{aa}}{2}, \qquad
G^{(2)}(j,k) - n_an_b = \mathrm{Cov}_W\big(|a_j|^2, |b_k|^2\big) - \mathrm{Re}\big[c_{ab}^*\langle a_jb_k^*\rangle_W\big] + \frac{|c_{ab}|^2}{4},
$$

and $g_2 - 1 = (G^{(2)} - n_an_b)/(n_an_b)$ (`populations`, `g2m1`, `connected_G2`).
In position space $a = b = \psi(x)$ and $c = \delta_{xx'}/dx$ (`position_commutators`); for windowed transforms $c_{aa} = \sum|w_a|^2/dx$ and $c_{ab}$ is the overlap of the two windows in Fourier space (`momentum_commutators`).

### 6.3 Comparing with the linear theory: the $g/dx$ compensation

The stationary state $\psi_0$ and the Bogoliubov operator are computed with the term $-gs/dx$ included.
In a TWA run on the same setup, the vacuum fluctuations, $\langle|\delta\psi|^2\rangle_W \approx 1/(2dx)$ per site, add back $2g\times1/(2dx) = g/dx$ to the mean nonlinear term.
To first order, the uncompensated TWA therefore describes the flow of the setup with $V + g/dx$, which is detuned by $g/dx$ from the flow $\psi_0$ of the linear theory.
To compare the TWA with the linear theory on the same flow, the TWA is run on the setup with the potential lowered by $g/dx$,

```julia
s_twa = remake(s; V = s.V .- s.pol.g / s.grid.dx)
```

starting from the same $\psi_0$.
The size of the effect depends on the geometry, through $g/dx$ relative to $gn_0$ and through the sensitivity of the flow to its detuning.
It matters most where the excitations are soft, for example near the sonic point of a horizon, where low-frequency modes respond strongly to small changes of the flow.

### 6.4 Time step, splitting and GPU batches

Each Strang step applies half a step of the local terms (potential, damping, nonlinearity and pump), a full step of the kinetic term in Fourier space, and another half step of the local terms; the Wigner noise $\sqrt{dt/2}\sqrt{\kappa/dx}\,\xi$ is added at the end of each half step.
The time step that `evolve` uses divides the sampling interval exactly (`effective_dt`), and the pump is evaluated at the correct time within the step.
The stationary statistics of the discrete scheme depend on $dt$, and the continuous-time linear theory of Sec. 4 is their $dt\to0$ limit; comparisons between TWA and linear theory are made at small $dt$, or with $dt$ treated as a parameter of the comparison.
Trajectories run as the columns of one batch through GeneralizedGrossPitaevskii.jl, on the CPU or, with `array=CuArray`, on a GPU; the moments are accumulated on the device.

## 7. Stimulated scattering in the time domain

### 7.1 Probe runs

A probe run starts from the stationary state, switches on a weak monochromatic probe at $(k, \omega)$, relaxes for a time $t_{\rm relax}$, and records the deviations $\psi(t_j) - \psi_0$ every $\Delta t$ over $n_s$ samples (`ProbeProtocol(; t_relax, nsamples, Δt, dt)`, `probe_samples`).
The time series is multiplied by a Hann window and demodulated at $+\omega$ (the probed component $e^{-i\omega t}$, i.e. $u$) and at $-\omega$ (the conjugate component, i.e. $v^*$), with frequency resolution $2\pi/(n_s\Delta t)$ (`demodulate`, `frequency_resolution`).
In each region, the demodulated fields are multiplied by a spatial Hann window and Fourier transformed (`hann_window`, `spatial_spectrum`), which gives $|\cdot|^2$ spectra on `kaxis(grid)` for the upstream and downstream regions (`probe_readout`).
The function `probe_scan(s, ψ0, probes; batch, windows)` runs many probes at once as the columns of a batch, each with its own frequency, and accumulates the demodulation sums while stepping, so that no snapshots are stored (`probe_demodulated`).
Probe runs need double precision: single-precision rounding errors, re-injected at every step, accumulate to a level comparable with a weak probe's response.
The function `incident_probes` places probes exactly on the incoming upstream positive-norm mode at each frequency, and `spacetime_spectrum` gives $|S(k,\omega)|^2$ of arbitrary space–time samples.

### 7.2 Mode-resolved readout

In a homogeneous region with background $\sqrt n\,e^{ik_0x}$, a mode of real frequency $\omega$ is

$$
\delta\psi = a\Big[U e^{i(kx-\omega t)} + V^*e^{-i(k^*x-\omega t)}e^{2ik_0x}\Big],
$$

so demodulation at $+\omega$ gives $aUe^{ikx}$ at the laboratory wavevector $k$, and demodulation at $-\omega$ gives $a^*V^*e^{i(2k_0-k^*)x}$ at the mirror wavevector $2k_0 - k$.
The pair $(U, V)$ solves the $2\times2$ laboratory-frame Bogoliubov problem at $k$, normalised to $|U|^2 - |V|^2 = \pm1$ (`bogoliubov_uv`).
The function `channel_modes(f, ω, region; κ)` lists all propagating LDA modes at $\omega$, labelled by branch and direction of propagation, with complex wavevectors $k_c$ that solve $\omega + i\kappa = \Omega_b(k_c)$ exactly (`complex_root`), to first order $k_c \approx k + i\kappa/v_g$ (`ScatteringMode`).
The function `fit_modes(x, ψp, ψm, ψ0, modes; xref)` fits the demodulated response of a region to these modes by least squares and returns their complex amplitudes $a_j$ at a reference point.
The norm flux of a mode is $J = |a|^2(|U|^2 - |V|^2)v_g$ (`norm_flux`), and scattering coefficients are the outgoing fluxes normalised to the incident one.
The same fit applied to the exact response of Sec. 3.2 gives the coefficients without time-domain errors.

### 7.3 Ringdown

A wavepacket is sent towards the horizon by a moving source, $F_{\rm extra}(x,t) = iF\exp[-(w_k/2)^2\xi^2]e^{ik_0\xi}\Theta(x_{\rm stop} - x)$ with $\xi = x - x_{\rm stop} - c(t - t_c)$ (`WavePacketSource`), and the relative density change at a chosen point is recorded as a function of time (`ringdown(s, ψ0, src; t_end, Δt)`).
Its free decay after the packet has passed is then analysed.
The function `matrix_pencil(y, Δt)` fits $y_n \approx \sum_ja_je^{-i\omega_jn\Delta t}$ with complex frequencies $\omega_j$ and no initial guess, the model order being set by the number of significant singular values of the Hankel matrix of the signal.
Damped modes have $\mathrm{Im}\,\omega_j < 0$.

## 8. References

1. M. Burkhard, M. Kroj, K. Falque, A. Bramati, I. Carusotto, M. J. Jacquet, C. R. Phys. 25, 341 (2026), arXiv:2511.12339
2. M. Gil de Olivera, M. Joly, A. Z. Khoury, A. Bramati, M. J. Jacquet, Phys. Rev. A 114, L021701 (2026), arXiv:2512.17807
3. K. Falque et al., Phys. Rev. Lett. 135, 023401 (2025)
4. C. W. Gardiner and M. J. Collett, Phys. Rev. A 31, 3761 (1985)
5. I. Carusotto and C. Ciuti, Rev. Mod. Phys. 85, 299 (2013)
6. I. Frérot et al., Phys. Rev. X 13, 041058 (2023)
7. A. Sinatra, C. Lobo, and Y. Castin, J. Phys. B 35, 3599 (2002)
