# Stationary mean field and its hydrodynamic description.

"""
    steady_state(s; T=12000, dt=0.01, u0=zeros, envelope=(A=0, τ=1))

Noise-free evolution from `u0` (default: the empty cavity) for a time `T` (ps). It
returns the final field. Check convergence with [`stationarity`](@ref).
"""
function steady_state(s::Setup; T=12000.0, dt=0.01, u0=zeros(Complex{eltype(s.grid.x)}, s.grid.N),
    envelope=(A=0.0, τ=1.0))
    _, ψ = evolve(s, u0, (zero(T), T); dt, nsaves=1, envelope)
    ψ[:, end]
end

"""
Relative change of `ψ` over a further evolution time `t` (ps): a convergence check. Use
the same `dt` as for computing `ψ`: the fixed point of the scheme moves by O(dt²) with dt.
"""
function stationarity(s::Setup, ψ; t=1.0, dt=0.01)
    _, ψ2 = evolve(s, ψ, (0.0, t); dt)
    maximum(abs, ψ2[:, end] .- ψ) / maximum(abs, ψ)
end

"""Mean-field density |ψ|² (µm⁻¹). Wigner-shift free: for deterministic runs only."""
density(ψ) = abs2.(ψ)

"""
    wavevector(s, ψ)

Local carrier wavevector `∂ₓθ` (µm⁻¹), from a centred, unwrapping-free phase difference
`arg(ψ_{i+1} ψ*_{i-1})/(2dx)` on the periodic grid.
"""
function wavevector(s::Setup, ψ)
    N = length(ψ)
    [angle(ψ[mod1(i + 1, N)] * conj(ψ[mod1(i - 1, N)])) / (2s.grid.dx) for i in 1:N]
end

"""
    velocity(s, ψ)

Flow velocity: the group velocity at the local wavevector, `kinetic_d1(pol, ∂ₓθ)` (µm/ps);
`(ħ/m) ∂ₓθ` for the parabolic dispersion.
"""
velocity(s::Setup, ψ) = kinetic_d1.(Ref(s.pol), wavevector(s, ψ))

"""
    local_detuning(s, v)

Effective detuning `δ(v) = ω_p − ω₀ − m v²/(2ħ) + g s/dx` (ps⁻¹) of a fluid flowing at `v`.
In the paper notation δ(v₀) = δ(k_p) with v₀ = ħk_p/m. The last term is the Wigner shift
of the simulated equation, g(|ψ|² − 1/dx), which acts as an extra detuning
(1.2e-3 ps⁻¹ in `burkhard2026`); it is zero when `s.wigner_shift == false`.
"""
local_detuning(s::Setup, v) = detuning_at_k(s, wavevector_for_velocity(s.pol, v))

"""Effective detuning `ω_p − ω₀ − kinetic(k) + g s/dx` of a fluid with carrier wavevector `k`."""
detuning_at_k(s::Setup, k) = detuning0(s) - kinetic(s.pol, k) + (s.wigner_shift ? s.pol.g / s.grid.dx : zero(float(k)))

"""Speed `√(ħ g n / m)` (µm/ps), the `c_x` of arXiv:2511.12339 (k = 0 mass)."""
sound_speed_gn(s::Setup, n) = sqrt(s.pol.ħ * s.pol.g * n / s.pol.m)

"""
    bogoliubov_speed(s, n, v)

Bogoliubov sound speed `c_B = √(ħ(2gn − δ(v))/m)` (Eq. 6 of arXiv:2511.12339), with ħ/m replaced
by the local curvature `kinetic_d2` for a non-parabolic dispersion. It is `NaN` where 2gn < δ.
"""
function bogoliubov_speed(s::Setup, n, v)
    k = wavevector_for_velocity(s.pol, v)
    a = kinetic_d2(s.pol, k) * (2s.pol.g * n - detuning_at_k(s, k))
    a >= 0 ? sqrt(a) : oftype(a, NaN)
end

"""
    profiles(s, ψ)

NamedTuple of mean-field profiles on the grid: `x`, density `n`, velocity `v`,
`c = √(ħgn/m)`, Bogoliubov speed `cB = √(ħ(2gn − δ)/m)`, effective detuning `δ` (frozen
reservoir included) and Mach number `v/cB`.
"""
function profiles(s::Setup, ψ)
    n = density(ψ)
    k = wavevector(s, ψ)
    v = kinetic_d1.(Ref(s.pol), k)
    δ = detuning_at_k.(Ref(s), k) .- s.Vres                   # effective detuning (frozen reservoir included)
    a = kinetic_d2.(Ref(s.pol), k) .* (2s.pol.g .* n .- δ)
    cB = [x >= 0 ? sqrt(x) : oftype(x, NaN) for x in a]
    (; x=s.grid.x, n, v, c=sound_speed_gn.(Ref(s), n), cB, δ, mach=v ./ cB)
end

"""
    horizon(s, ψ; near, window=20)

Position (µm) where the flow turns supercritical, searched within `near ± window` along
increasing x. It is the first point that is supersonic (v ≥ c_B) or where c_B is undefined
(2gn < δ, e.g. in the density dip of a repulsive defect), right after a subsonic point.
The position is linearly interpolated when both neighbours have a finite c_B. Returns
`NaN` if the flow does not cross from subsonic to supersonic in the window.
"""
function horizon(s::Setup, ψ; near, window=20.0)
    p = profiles(s, ψ)
    f = p.v .- p.cB
    idx = findall(x -> abs(x - near) <= window, s.grid.x)
    for (a, b) in zip(idx[1:end-1], idx[2:end])
        if !isnan(f[a]) && f[a] < 0 && (isnan(f[b]) || f[b] >= 0)
            return isnan(f[b]) ? s.grid.x[b] : s.grid.x[a] + s.grid.dx * (-f[a]) / (f[b] - f[a])
        end
    end
    oftype(near, NaN)
end

# ---------------------------------------------------------------------------------------
# Homogeneous plane-wave states and optical bistability
# ---------------------------------------------------------------------------------------

"""
    pump_intensity(pol, δ, n)

Equation of state of a homogeneous driven fluid: |F|² = n[(gn − δ)² + γ²/4] (ps⁻² µm⁻¹),
with effective detuning `δ` (ps⁻¹). This is Eq. (2) of arXiv:2511.12339.
"""
pump_intensity(pol::Polariton, δ, n) = n * ((pol.g * n - δ)^2 + pol.γ^2 / 4)

"""
    sonic_pump_amplitude(pol, δ)

Pump amplitude |F| at `g n = δ`, i.e. `(γ/2)√(δ/g)`. Both codes use this point as the
reference ("turning point") for the operating amplitude.
"""
sonic_pump_amplitude(pol::Polariton, δ) = pol.γ / 2 * sqrt(δ / pol.g)

"""
    turning_points(pol, δ)

Densities at the turning points of the bistability loop (lower, upper), the roots of
d|F|²/dn = 0: `g n = (2δ ∓ √(δ² − 3γ²/4))/3`. Returns `nothing` when the loop does not
exist (δ < √3 γ/2).
"""
function turning_points(pol::Polariton, δ)
    disc = δ^2 - 3pol.γ^2 / 4
    disc < 0 && return nothing
    ((2δ - sqrt(disc)) / (3pol.g), (2δ + sqrt(disc)) / (3pol.g))
end
