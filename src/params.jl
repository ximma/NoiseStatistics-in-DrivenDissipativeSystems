# Material parameters and numerical grid.
#
# Units throughout the package: lengths in µm, times in ps, energies in meV.
# Frequencies (ω, γ, detunings, potentials) are energies divided by ħ, in ps⁻¹.
# The field ψ is in µm^{-1/2}, so that |ψ|² is a linear density. Pump amplitudes F are in
# ps⁻¹ µm^{-1/2} (the GPE is i∂ₜψ = … + F).

"""
    Polariton(; ħ=0.6582, m=ħ²/(2·1.29), γ=0.047/ħ, g=3e-4/ħ, ω0=1473.36/ħ)

Lower-polariton parameters. The defaults are those of Burkhard *et al.*, C. R. Phys. 25, 341 (2026), arXiv:2511.12339:

| field | meaning | unit |
|-------|---------|------|
| `ħ`  | reduced Planck constant | meV ps |
| `m`  | effective mass (ħ²/2m = 1.29 meV µm²) | meV ps² µm⁻² |
| `γ`  | loss rate of the polariton *intensity* (amplitude decays at γ/2) | ps⁻¹ |
| `g`  | interaction constant (energy ħ g n) | µm ps⁻¹ |
| `ω0` | bottom of the lower-polariton branch | ps⁻¹ |
| `ΩR` | Rabi splitting ħΩ_R/ħ; `Inf` (default): parabolic dispersion | ps⁻¹ |
| `Δxc` | exciton–cavity detuning (E_C(0) − E_X)/ħ, used when `ΩR` is finite | ps⁻¹ |

With finite `ΩR` the kinetic energy is that of the lower polariton,

    E(k) − E(0) = ½[E_C(k) − √((E_C(k) − E_X)² + (ħΩ_R)²)] − (same at k = 0),
    E_C(k) = E_C(0) + ħ²k²/2m_ph,   m_ph = |C₀|² m,

so that `m` stays the effective mass at k = 0 (see [`kinetic`](@ref)). The exciton mass is
taken as infinite, and g and γ as k-independent.
"""
Base.@kwdef struct Polariton{T<:AbstractFloat}
    ħ::T = 0.6582
    m::T = 0.6582^2 / (2 * 1.29)
    γ::T = 0.047 / 0.6582
    g::T = 3e-4 / 0.6582
    ω0::T = 1473.36 / 0.6582
    ΩR::T = Inf
    Δxc::T = 0.0
end

"""`true` for the parabolic kinetic term ħk²/2m (the default, `ΩR = Inf`)."""
is_parabolic(pol::Polariton) = isinf(pol.ΩR)

"""Lower-polariton constants: Rabi frequency Ω, detuning d, photon a_ph = ħ/2m_ph, offset e(0)."""
function _lpconst(pol::Polariton)
    Ω, d = pol.ΩR, pol.Δxc
    C0 = (1 - d / sqrt(d^2 + Ω^2)) / 2
    (; Ω, d, aph=pol.ħ / (2pol.m * C0), e0=(d - sqrt(d^2 + Ω^2)) / 2, C0)
end

"""
    kinetic(pol, k)

Kinetic frequency (E(k) − E(0))/ħ (ps⁻¹) at wavevector `k` (real or complex): ħk²/2m, or the
lower polariton if `pol.ΩR` is finite (see [`Polariton`](@ref)).
"""
function kinetic(pol::Polariton, k)
    is_parabolic(pol) && return pol.ħ * k^2 / (2pol.m)
    c = _lpconst(pol); εC = c.d + c.aph * k^2
    (εC - sqrt(εC^2 + c.Ω^2)) / 2 - c.e0
end
"""Group velocity ∂ₖ kinetic (µm/ps)."""
function kinetic_d1(pol::Polariton, k)
    is_parabolic(pol) && return pol.ħ * k / pol.m
    c = _lpconst(pol); εC = c.d + c.aph * k^2
    c.aph * k * (1 - εC / sqrt(εC^2 + c.Ω^2))
end
"""Curvature ∂ₖ² kinetic (µm²/ps): ħ/m for the parabola, ħ/m_eff(k) in general."""
function kinetic_d2(pol::Polariton, k)
    is_parabolic(pol) && return pol.ħ / pol.m + zero(k)
    c = _lpconst(pol); εC = c.d + c.aph * k^2; r = sqrt(εC^2 + c.Ω^2)
    c.aph * (1 - εC / r) - 2c.aph^2 * k^2 * c.Ω^2 / r^3
end
"""Photonic Hopfield fraction |C_k|² of the lower polariton (1 for the parabolic model)."""
function photonic_fraction(pol::Polariton, k)
    is_parabolic(pol) && return one(float(k))
    c = _lpconst(pol); εC = c.d + c.aph * k^2
    (1 - εC / sqrt(εC^2 + c.Ω^2)) / 2
end

"""Inflection point k_i > 0 of the lower polariton (curvature zero), `Inf` for the parabola."""
function inflection_k(pol::Polariton)
    is_parabolic(pol) && return Inf
    lo, hi = 0.0, 1.0
    while kinetic_d2(pol, hi) > 0; lo = hi; hi *= 2; end
    for _ in 1:200; mid = (lo + hi) / 2; kinetic_d2(pol, mid) > 0 ? (lo = mid) : (hi = mid); end
    (lo + hi) / 2
end
"""Largest group velocity of the dispersion (at the inflection point; `Inf` for the parabola)."""
max_velocity(pol::Polariton) = is_parabolic(pol) ? Inf : kinetic_d1(pol, inflection_k(pol))

"""
    wavevector_for_velocity(pol, v)

Carrier wavevector k with group velocity `kinetic_d1(pol, k) = v` on the branch below the
inflection point (m v/ħ for the parabola). Throws if |v| exceeds [`max_velocity`](@ref).
"""
function wavevector_for_velocity(pol::Polariton, v)
    is_parabolic(pol) && return pol.m * v / pol.ħ
    v == 0 && return zero(float(v))
    ki = inflection_k(pol); vmax = kinetic_d1(pol, ki)
    abs(v) <= vmax || throw(ArgumentError("|v| = $(abs(v)) µm/ps exceeds the maximum lower-polariton group velocity $vmax µm/ps"))
    lo, hi = 0.0, ki
    for _ in 1:200; mid = (lo + hi) / 2; kinetic_d1(pol, mid) < abs(v) ? (lo = mid) : (hi = mid); end
    sign(v) * (lo + hi) / 2
end

"""
    Grid(L, N)

Periodic 1D grid of length `L` (µm) with `N` points: `x = 0, dx, …, L - dx`. `k` holds the
wavevectors in FFT order (`fftfreq`), consistent with GeneralizedGrossPitaevskii.jl.
`kaxis(grid)` gives them in increasing order, matching `fftshift(fft(·))`.
"""
struct Grid{T<:AbstractFloat}
    L::T
    N::Int
    dx::T
    x::Vector{T}
    k::Vector{T}
end
function Grid(L::Real, N::Integer)
    T = float(typeof(L))
    Lf = T(L)
    Grid{T}(Lf, Int(N), Lf / N, collect(GeneralizedGrossPitaevskii.direct_grid(Lf, N)),
        collect(GeneralizedGrossPitaevskii.reciprocal_grid(Lf, N)))
end
Base.show(io::IO, g::Grid) = print(io, "Grid(L=$(g.L) µm, N=$(g.N), dx=$(g.dx) µm)")

"""Wavevectors in increasing order (the axis of `fftshift(fft(v))`)."""
kaxis(g::Grid) = fftshift(g.k)

"""Wavevector spacing 2π/L."""
dk(g::Grid) = 2π / g.L
