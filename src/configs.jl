# General transcritical geometries and the paper presets.

"""
    defect_potential(defect, ħ)

Defect height V0 in meV: `:attractive` (−0.85), `:repulsive` (+0.85), `:none` (0), or a number in meV.
"""
defect_height(defect) = defect === :attractive ? -0.85 : defect === :repulsive ? 0.85 :
                        defect === :none ? 0.0 : defect isa Real ? float(defect) :
                        throw(ArgumentError("defect must be :attractive, :repulsive, :none or a number (meV)"))

"""
    transcritical_setup(pol, grid; ωp, x_d, x_step=x_d − 7, k_up, k_down, F_up, F_down,
        boosts=SechBoost[], regional_boosts=false, defect=:attractive, defect_width=0.75,
        absorbers, wigner_shift=true, description="")

General 1D all-optically controlled transcritical geometry:
- a step pump (`StepPump`): k_up and F_up upstream, k_down and F_down downstream, with optional
  sech boosts at the boundaries;
- a Gaussian defect at x_d, `defect` = `:attractive`, `:repulsive`, `:none` or V0 in meV;
- edge absorbers.

`F_up` and `F_down` are pump amplitudes. [`above_sonic`](@ref) builds them as offsets from
the sonic pump amplitude. `F_down = 0` means no downstream density support.
"""
function transcritical_setup(pol::Polariton, grid::Grid; ωp, x_d, x_step=x_d - 7, k_up, k_down, F_up, F_down,
    boosts=SechBoost{Float64}[], regional_boosts::Bool=false, defect=:attractive, defect_width=0.75,
    absorbers, wigner_shift::Bool=true, description::AbstractString="")
    pump = StepPump(; x_step, k_up, k_down, F_up, F_down, boosts=collect(boosts), regional=regional_boosts)
    make_setup(pol, grid; ωp, pump, wigner_shift, description,
        defects=(GaussianDefect(x_d, defect_width, defect_height(defect) / pol.ħ),), absorbers)
end

"""
    above_sonic(pol, ωp, k, offset)

Pump amplitude `F_sonic(δ(k)) + offset`, with δ(k) = ω_p − ω₀ − ħk²/2m. This is the
original recipe for setting the operating point just above the turning point of the
bistability loop.
"""
above_sonic(pol::Polariton, ωp, k, offset) = sonic_pump_amplitude(pol, ωp - pol.ω0 - kinetic(pol, k)) + offset

"""
    burkhard2026(; defect=:attractive, support=true, gauge=:shifted, F_up_offset=0.01, F_down_offset=0.13)

Geometry of Burkhard *et al.*, C. R. Phys. 25, 341 (2026), arXiv:2511.12339:
- L = 800 µm and N = 2048;
- ħω_p = 1473.85 meV;
- pump step at x_d − 7 µm, with k_up = 0.27 and k_down = 0.539 µm⁻¹;
- upstream sech boost to |F| = 9 at x = 0;
- absorber κ = 4.5 meV/ħ (width 20 µm) at x = L;
- Gaussian defect of 0.85 meV and width 0.75 µm at x_d = 400 µm.

Options:
- `defect`: `:attractive` (−0.85 meV, the paper), `:repulsive` (+0.85 meV), `:none`, or a
  number (V0 in meV).
- `support`: downstream density support. With `true`, F_down = F_sonic,down + `F_down_offset`.
  With `false`, F_down = 0.
- `gauge=:shifted`: wavevectors `k + 2π/L`. The FFT grid of the original simulations assigned k + δk to every bin,
  so those runs are *exactly* the correct-grid runs with pump wavevectors shifted by δk = 2π/L.
  This option reproduces the published data. `gauge=:nominal` uses k_up = 0.27 and
  k_down = 0.539 literally. Pump amplitudes always follow the original recipe
  (F_sonic from the nominal k).
"""
function burkhard2026(; defect=:attractive, support::Bool=true, gauge::Symbol=:shifted,
    F_up_offset=0.01, F_down_offset=0.13, N::Integer=2048, L=800.0)
    pol = Polariton()
    ħ = pol.ħ
    ωp = 1473.85 / ħ
    k_up, k_down = 0.27, 0.539
    F_up = above_sonic(pol, ωp, k_up, F_up_offset)
    F_down = support ? above_sonic(pol, ωp, k_down, F_down_offset) : 0.0
    shift = gauge === :shifted ? 2π / L : gauge === :nominal ? 0.0 : throw(ArgumentError("gauge must be :shifted or :nominal"))
    transcritical_setup(pol, Grid(L, N); ωp, x_d=400.0, k_up=k_up + shift, k_down=k_down + shift, F_up, F_down,
        boosts=[SechBoost(0.0, 20.0, 9 - F_up, k_up + shift)], defect,
        absorbers=(EdgeAbsorber(L, 20.0, 4.5 / ħ),),
        description="Burkhard et al. 2026 (arXiv:2511.12339); defect=$defect, support=$support, gauge=$gauge")
end

"""Probe of arXiv:2511.12339 (x_pr = 300 µm, σ = 5 µm, |F_pr| = F_up/1000) at `(k, ω)`."""
burkhard2026_probe(s::Setup, k, ω; x0=300.0, σ=5.0) =
    GaussianProbe(; x0, σ, k, ω, F=s.pump.F_up / 1000)

"""
    gildeoliveira2026(; defect=:repulsive, N=512, L=1024.0)

Geometry of Gil de Olivera *et al.*, Phys. Rev. A 114, L021701 (2026), arXiv:2512.17807 (Table I
of the SM). It reproduces
the pump, losses and potential of `momentum_correlations_polaritons/equations.jl`:
- m = 1/6 meV ps² µm⁻²; δ₀ = 0.49 meV/ħ; γ = 0.047 meV/ħ; g = 3e-4 meV µm/ħ;
- absorbers of 4.5 meV/ħ (width 20 µm) at x = 0 and x = L;
- repulsive defect of 0.85 meV (width 0.75 µm) at x_d = L/2;
- step pump at x_s = x_d − 7 µm: k_up = 0.148 and k_down = 0.614 µm⁻¹;
  F_up = F_sonic,up + 0.01 and F_down = F_sonic,down + 0.4;
- sech boosts to F_M = 20 within w_p = 20 µm of x = 0 (upstream) and x = L (downstream),
  each confined to its own region.

The runs use the start-up envelope h(t) = 1 + 6 e^{−t/50 ps}, returned as
`gildeoliveira2026_envelope`, with dt = 0.2 ps and T = 400 ps.

`defect`: `:repulsive` (the paper), `:attractive`, `:none`, or V0 in meV.
`support=false` removes the downstream pump (F_down = 0; the downstream boost at x = L is kept).
"""
function gildeoliveira2026(; defect=:repulsive, support::Bool=true, N::Integer=512, L=1024.0)
    ħ = 0.6582
    pol = Polariton(; ħ, m=1 / 6, γ=0.047 / ħ, g=3e-4 / ħ)
    δ₀ = 0.49 / ħ
    ωp = pol.ω0 + δ₀
    k_up, k_down = 0.148, 0.614
    δ(k) = δ₀ - kinetic(pol, k)                   # from δ₀ directly: (ω₀ + δ₀) − ω₀ ≠ δ₀ in floating point
    F_up = sonic_pump_amplitude(pol, δ(k_up)) + 0.01
    F_down = support ? sonic_pump_amplitude(pol, δ(k_down)) + 0.4 : 0.0
    F_M, w_p = 20.0, 20.0
    transcritical_setup(pol, Grid(L, N); ωp, x_d=L / 2, k_up, k_down, F_up, F_down,
        boosts=[SechBoost(0.0, w_p, F_M - F_up, k_up), SechBoost(L, w_p, F_M - F_down, k_down)],
        regional_boosts=true, defect,
        absorbers=(EdgeAbsorber(0.0, 20.0, 4.5 / ħ), EdgeAbsorber(L, 20.0, 4.5 / ħ)),
        description="Gil de Olivera et al. 2026 (arXiv:2512.17807, Table I); defect=$defect, support=$support")
end

"""Start-up envelope h(t) = 1 + 6 e^{−t/50} of arXiv:2512.17807."""
const gildeoliveira2026_envelope = (A=6.0, τ=50.0)
