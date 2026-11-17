# All-optical flow design: target mean fields and the exact pumps that sustain them.
#
# With control over both the amplitude and the phase of the pump, any smooth target field
# ψ_t(x) is an exact stationary solution of the driven-dissipative GPE for the pump
#
#     F(x) = −[e(−i∂ₓ) − δ₀ + V + V_res + g(|ψ_t|² − s/dx) − iκ] ψ_t,
#
# evaluated with the same spectral kinetic operator e(k) (`kinetic`: ħk²/2m by default) as the
# time stepper. A target is
# specified by the flow velocity v(x) and the Bogoliubov sound speed c(x):
#
#     k_p = m v/ħ,   θ = ∫ k_p dx,   c² = (ħ/m)(2gn − δ_eff),   δ_eff = δ(v) − V − g_r n_r,
#
# For a non-parabolic (lower-polariton) dispersion, v is the group velocity e′(k_p), δ(v) =
# δ₀ − e(k_p) and ħ/m is the local curvature e″(k_p) (`kinetic`, `kinetic_d1`, `kinetic_d2`).
# with a frozen exciton reservoir g_r n_r = β g n (Falque *et al.*, PRL 135, 023401 (2025),
# Appendix B of arXiv:2311.01392: β = 1.84). The reservoir is slow on the Bogoliubov time
# scales, so it enters the fluctuations as a static potential (Setup.Vres).

"""
    lkb_gaas_polariton(; g=3e-4)

GaAs microcavity of the LKB, as characterised in Falque *et al.*, PRL 135, 023401 (2025)
(arXiv:2311.01392): m* = 7.0×10⁻³⁵ kg,
ħγ = 80 µeV. The interaction constant is not given there; `g` (meV µm) only sets the
density scale (the Bogoliubov dynamics depend on gn).
"""
lkb_gaas_polariton(; g=3e-4, ħ=0.6582) =
    Polariton(; ħ, m=7.0e-35 / 1.602176634e-34, γ=0.080 / ħ, g=g / ħ)

"""Waterfall velocity profile (Eq. 5 of Falque *et al.*): vu → vd over a width wH around xH."""
waterfall(x; xH, vu, vd, wH) = (vd - vu) / 2 * tanh((x - xH) / wH) + (vd + vu) / 2

"""Gaussian bump of amplitude `A` and half width at half maximum `hwhm`, centred at `x0`."""
bump(x; x0, A, hwhm) = A * exp(-log(2) * (x - x0)^2 / hwhm^2)

"""
    design_flow(pol, grid; δ0, v, c, β=0, V=0, wigner_shift=true, fix=nothing)

Target mean field for a flow velocity `v(x)` and Bogoliubov sound speed `c(x)` (vectors on
the grid): ψ_t = √n e^{iθ}, with n from c² = e″(k)((2+β)gn − δ(v) + V) and θ = ∫ k dx, where
k(x) has group velocity v (k = m v/ħ and e″ = ħ/m for the parabolic dispersion).
`δ0` is ω_p − ω₀ (ps⁻¹). The total phase must wind by a multiple of 2π over the periodic box:
the wavevector is shifted by a constant on the indices `fix` (e.g. inside an absorber) to
enforce it. Returns `(; ψ, n, v, c, Vres, θ)`, with `Vres = β g n`.
"""
function design_flow(pol::Polariton{T}, grid::Grid{T}; δ0, v::AbstractVector, c::AbstractVector, β=0.0, V=0.0,
    wigner_shift::Bool=true, fix=nothing) where {T}
    ħ, m, g, dx = pol.ħ, pol.m, pol.g, grid.dx
    v = Vector{T}(v)
    k = wavevector_for_velocity.(Ref(pol), v)
    if fix !== nothing
        Φ = sum(k) * dx
        target = 2π * round(Φ / 2π)
        k[fix] .+= (target - Φ) / (length(fix) * dx)
        v[fix] .= kinetic_d1.(Ref(pol), k[fix])
    end
    shift = wigner_shift ? g / dx : zero(T)
    δ = δ0 .- kinetic.(Ref(pol), k) .+ shift
    gn = (c .^ 2 ./ kinetic_d2.(Ref(pol), k) .+ δ .- V) ./ (2 + β)
    any(<(0), gn) && throw(ArgumentError("target sound speed below the minimum allowed by δ(v) (gn < 0)"))
    θ = [zero(T); cumsum((k[1:end-1] .+ k[2:end]) ./ 2 .* dx)]
    n = gn ./ g
    (; ψ=sqrt.(n) .* cis.(θ), n, v, c=Vector{T}(c), Vres=β .* gn, θ)
end

"""
    exact_pump(s, ψ)

The static pump F(x) for which `ψ` is an exact stationary state of setup `s` (spectral
kinetic term, potential, frozen reservoir, losses and Wigner shift included).
"""
function exact_pump(s::Setup{T}, ψ::AbstractVector) where {T}
    g = s.pol.g
    shift = s.wigner_shift ? 1 / s.grid.dx : zero(T)
    Kψ = ifft(kinetic.(Ref(s.pol), s.grid.k) .* fft(ψ))
    -(Kψ .+ (-detuning0(s) .+ s.V .+ s.Vres .+ g .* (abs2.(ψ) .- shift) .- im .* s.κ) .* ψ)
end

"""
    lowpass_pump(s, F, kmax)

Pump band-limited to |k| ≤ `kmax` (µm⁻¹): the spatial frequencies an imaging system of
numerical aperture NA transmits are |k| ≤ 2π NA/λ.
"""
lowpass_pump(s::Setup, F, kmax) = ifft(fft(F) .* (abs.(s.grid.k) .<= kmax))

"""
    designed_setup(pol, grid; δ0, v, c, β=0, defects=(), absorbers=(), fix=nothing,
                   kmax=Inf, description="")

All-optically controlled setup: target flow `v(x)`, `c(x)` (see [`design_flow`](@ref)),
frozen reservoir `β g n`, and the exact pump that sustains it (band-limited to `kmax` if
finite). Returns `(s, ψ)`, with `ψ` the target field, to be used as the initial condition
of [`steady_state`](@ref) so that the evolution starts on the designed branch.
"""
function designed_setup(pol::Polariton{T}, grid::Grid{T}; δ0, v, c, β=0.0, defects=(), absorbers=(), fix=nothing,
    kmax=Inf, description::AbstractString="") where {T}
    base = make_setup(pol, grid; ωp=pol.ω0 + δ0, pump=zeros(Complex{T}, grid.N), defects, absorbers, description)
    d = design_flow(pol, grid; δ0, v, c, β, V=base.V, fix)
    s = remake(base; Vres=d.Vres)
    F = exact_pump(s, d.ψ)
    isfinite(kmax) && (F = lowpass_pump(s, F, kmax))
    remake(s; F, pump=(; kind=:designed, δ0, β, kmax)), d.ψ
end

"""
    reservoir_steady_state(s, ψ0; β, iters=20, T=2000, dt=0.01, tol=1e-9, mix=0.5)

Self-consistent stationary state with a frozen reservoir V_res = β g |ψ|²: alternately
evolve the GPE with V_res fixed and update V_res (with mixing `mix`) until it changes by
less than `tol` (relative). Returns `(s, ψ)` with the converged reservoir in `s.Vres`.
The fixed-point iteration diverges when the loop gain (the static density response to V_res)
exceeds 1 even if the state is stable; it then warns. Use [`reservoir_newton`](@ref).
"""
function reservoir_steady_state(s::Setup, ψ0; β, iters::Integer=20, T=2000.0, dt=0.01, tol=1e-9, mix=0.5)
    ψ = ψ0
    s = remake(s; Vres=β .* s.pol.g .* abs2.(ψ))
    δ = Inf
    for _ in 1:iters
        ψ = steady_state(s; T, dt, u0=ψ)
        Vn = β .* s.pol.g .* abs2.(ψ)
        δ = maximum(abs, Vn .- s.Vres) / maximum(abs, Vn)
        s = remake(s; Vres=(1 - mix) .* s.Vres .+ mix .* Vn)
        δ < tol && break
    end
    δ < tol || @warn "reservoir_steady_state: not converged after $iters iterations (relative V_res change $δ); use reservoir_newton"
    s, ψ
end

"""Jacobian of the stationary GPE in (δψ, δψ*) when the reservoir V_res = β g |ψ|² follows the
density adiabatically: the Bogoliubov matrix plus the reservoir's (Wigner-shift free) β g terms.
Its eigenvalues give the stability against perturbations slower than the reservoir lifetime."""
function reservoir_jacobian(s::Setup, ψ0::AbstractVector; β)
    J = bogoliubov_matrix(s, ψ0); N = s.grid.N; bg = β * s.pol.g
    for i in 1:N
        J[i, i] += bg * abs2(ψ0[i]); J[i, N+i] += bg * ψ0[i]^2
        J[N+i, i] -= bg * conj(ψ0[i])^2; J[N+i, N+i] -= bg * abs2(ψ0[i])
    end
    J
end

"""
    reservoir_newton(s, ψ0; β, tol=1e-12, maxit=40)

Self-consistent stationary state with V_res = β g |ψ|², by Newton's method on the stationary
equation `exact_pump(s, ψ) = s.F` (continuous-time fixed point, no time-step error), with the
[`reservoir_jacobian`](@ref) and a backtracking line search on the residual 2-norm. Returns
`(s, ψ)` with `s.Vres` set. Throws if it does not converge, which near ψ0 signals a fold of the
self-consistent branch (the slowest eigenvalue of the reservoir Jacobian goes to zero there).
"""
function reservoir_newton(s::Setup, ψ0::AbstractVector; β, tol=1e-12, maxit::Integer=40)
    withres(ψ) = remake(s; Vres=β .* s.pol.g .* abs2.(ψ))
    resid(ψ) = s.F .- exact_pump(withres(ψ), ψ)
    N = s.grid.N; ψ = Vector{ComplexF64}(ψ0); r = Inf
    for _ in 1:maxit
        G = resid(ψ); r = norm(G) / norm(s.F)
        r < tol && return withres(ψ), ψ
        δ = -(reservoir_jacobian(withres(ψ), ψ; β) \ vcat(G, -conj.(G)))[1:N]
        λ = 1.0
        while norm(resid(ψ .+ λ .* δ)) >= (1 - 1e-4λ) * norm(G)
            λ /= 2
            λ < 1e-4 && error("reservoir_newton: no descent at residual $r (fold of the self-consistent branch near ψ0?)")
        end
        ψ = ψ .+ λ .* δ
    end
    error("reservoir_newton: not converged after $maxit iterations (residual $r)")
end
