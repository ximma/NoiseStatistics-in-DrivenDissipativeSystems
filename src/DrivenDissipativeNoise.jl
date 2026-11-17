"""
    DrivenDissipativeNoise

Driven-dissipative polariton fluids in 1+1D all-optically controlled geometries: mean
field, LDA dispersion, stimulated (probe) and spontaneous (TWA) scattering at acoustic
horizons, Bogoliubov spectra and quasinormal modes. Time integration uses
GeneralizedGrossPitaevskii.jl (Strang splitting, CPU/GPU).

Units: µm, ps, meV. Frequencies are in ps⁻¹, in the frame rotating at the pump frequency.
"""
module DrivenDissipativeNoise

using GeneralizedGrossPitaevskii, FFTW, LinearAlgebra, Statistics, Random

include("params.jl")
include("geometry.jl")
include("model.jl")
include("meanfield.jl")
include("lda.jl")
include("spectra.jl")
include("probe.jl")
include("bogoliubov.jl")
include("scattering.jl")
include("linear_response.jl")
include("ringdown.jl")
include("configs.jl")
include("correlations.jl")
include("twa.jl")
include("noise.jl")
include("noise_spectra.jl")
include("gaussian_modes.jl")
include("design.jl")
include("phonons.jl")

export Polariton, Grid, kaxis, dk,
    GaussianDefect, EdgeAbsorber, SechBoost, StepPump, GaussianProbe, pump_amplitude, probe_profile,
    Setup, make_setup, remake, evolve, detuning0,
    steady_state, stationarity, density, velocity, local_detuning, sound_speed_gn, bogoliubov_speed,
    wavevector, detuning_at_k, kinetic, kinetic_d1, kinetic_d2, photonic_fraction, is_parabolic, inflection_k,
    max_velocity, wavevector_for_velocity, sound_speed,
    profiles, horizon, pump_intensity, sonic_pump_amplitude, turning_points,
    LocalFluid, local_fluid, lda_omega, omega_max, omega_min, lda_roots, group_velocity,
    hann, hann_window, demodulate, spatial_spectrum, spacetime_spectrum,
    ProbeProtocol, frequency_resolution, probe_samples, probe_readout, probe_demodulated, probe_scan, incident_probes,
    Stepper, advance!, state, effective_dt, gpe_problem,
    WavePacketSource, packet_halfwidth, packet_front_time, ringdown, matrix_pencil,
    LinearResponse, respond, norm_density, norm_current, dissipation,
    bogoliubov_uv, complex_root, ScatteringMode, norm_flux, channel_modes, fit_modes,
    BogoliubovSpectrum, kinetic_matrix, bogoliubov_matrix, bogoliubov_spectrum, mode_weight, find_qnm,
    transcritical_setup, above_sonic, defect_height,
    burkhard2026, burkhard2026_probe, gildeoliveira2026, gildeoliveira2026_envelope,
    SpatialWindow, window_k, windowed_fft, momentum_commutators, position_commutators,
    Moments, update!, populations, g2m1, connected_G2,
    TWAProtocol, gildeoliveira2026_protocol, twa_batch, TWAAccumulator, twa_run!,
    NoiseModel, diffusion, noise_diagonal, spectral_covariance, stationary_covariance, Fluctuations,
    window_matrix, bogoliubov_g2m1,
    Detector, coefficients, project, OutputNoise, coherent_components, self_homodyne, spontaneous_flux,
    with_efficiency, fano, intensity_correlation, conditional_variance, gemellity,
    DetectionBasis, modes_of, SpontaneousNoise, spontaneous_noise, spontaneous_terms,
    aperture_mode, orthonormalize_modes, mode_state, SchurResolvent, pixel_state, transform_modes, principal_modes, optimise_modes, quadrature_covariance, symplectic_eigenvalues, log_negativity, pair_ratio, max_pair_ratio,
    lkb_gaas_polariton, waterfall, bump, design_flow, exact_pump, lowpass_pump, designed_setup, reservoir_steady_state, reservoir_newton, reservoir_jacobian,
    PhononBath, phonon_shape, phonon_occupation, phonon_rates, phonon_self_energy, PhononNoise, phonon_mode_state, phonon_covariance

end
