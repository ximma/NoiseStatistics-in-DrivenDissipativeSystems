# The driven-dissipative GPE in the frame rotating at the pump frequency ω_p:
#
#   i∂ₜψ = [ħk²/2m − δ₀ + V(x) − iκ(x) + g(|ψ|² − s/dx)] ψ + F(x, t) + √(κ(x)/dx) ξ(x, t)
#
# with δ₀ = ω_p − ω₀ and κ(x) = γ/2 + absorbers (amplitude damping). s = 1 subtracts the
# Wigner-ordering mean-field shift, as in the codes of arXiv:2511.12339 and arXiv:2512.17807. The last term is the
# truncated-Wigner noise, with ⟨ξξ*⟩ = δ_{xx'} δ(t−t'). It is switched on only for TWA runs.
# Integration uses Strang splitting from GeneralizedGrossPitaevskii.jl.

const GGP = GeneralizedGrossPitaevskii

"""
    Setup

A complete simulated geometry: polariton parameters, grid, pump frequency, real potential
`V(x)`, amplitude damping `κ(x)` (γ/2 included) and static pump profile `F(x)`.
Build one with [`make_setup`](@ref) or a paper configuration such as
[`burkhard2026`](@ref).
"""
struct Setup{T<:AbstractFloat}
    pol::Polariton{T}
    grid::Grid{T}
    ωp::T
    V::Vector{T}
    κ::Vector{T}
    F::Vector{Complex{T}}
    wigner_shift::Bool
    pump::Any                 # pump description, kept for reference
    description::String
    Vres::Vector{T}           # frozen reservoir potential g_r n_r(x) (ps⁻¹); zero by default
end
Setup{T}(pol, grid, ωp, V, κ, F, wigner_shift, pump, description) where {T} =
    Setup{T}(pol, grid, ωp, V, κ, F, wigner_shift, pump, description, zeros(T, grid.N))

"""
    remake(s; F=s.F, V=s.V, κ=s.κ, Vres=s.Vres, pump=s.pump, description=s.description)

Copy of setup `s` with some fields replaced: e.g. a designed pump profile `F`, or the frozen
reservoir potential `Vres` (see [`reservoir_steady_state`](@ref)).
"""
remake(s::Setup{T}; F=s.F, V=s.V, κ=s.κ, Vres=s.Vres, pump=s.pump, description=s.description) where {T} =
    Setup{T}(s.pol, s.grid, s.ωp, Vector{T}(V), Vector{T}(κ), Vector{Complex{T}}(F), s.wigner_shift, pump, String(description), Vector{T}(Vres))

"""
    make_setup(pol, grid; ωp, pump, defects=(), absorbers=(), wigner_shift=true, description="")

Assemble a [`Setup`](@ref). `pump` is a [`StepPump`](@ref) or an explicit vector `F(x)`.
"""
function make_setup(pol::Polariton{T}, grid::Grid{T}; ωp, pump, defects=(), absorbers=(),
    wigner_shift::Bool=true, description::AbstractString="") where {T}
    V = zeros(T, grid.N)
    for d in defects
        V .+= d.V0 .* gaussian.(grid.x, d.x0, d.w)
    end
    κ = fill(pol.γ / 2, grid.N)
    for a in absorbers
        κ .+= a.κ .* gaussian.(grid.x, a.x0, a.w)
    end
    F = pump isa AbstractVector ? Vector{Complex{T}}(pump) : Complex{T}[pump_amplitude(pump, x) for x in grid.x]
    Setup{T}(pol, grid, T(ωp), V, κ, F, wigner_shift, pump, String(description))
end

detuning0(s::Setup) = s.ωp - s.pol.ω0

# ---------------------------------------------------------------------------------------
# Pump callable with a fast, broadcast-based evaluation inside GGP
# ---------------------------------------------------------------------------------------

"""
    PolaritonPump(F, Fs, ω, A, τ, t_shift)

Time-dependent source in GGP's convention: GGP solves `i∂ₜu = … + iP`, so
`P = -i[F(x) h(t) + Fs(x) e^{-iωt}]`, with start-up envelope `h(t) = 1 + A e^{-t/τ}`.
`Fs` can be an `N × B` matrix with a vector `ω` of length B: each batch column then gets
its own probe frequency (batched probe scans). Probe phases e^{-iωt} are computed once per
column, in Float64, whatever the working precision.

An optional `extra(x, t)` returns an arbitrary additional source F_extra(x, t) (a complex
vector on the grid `x`, in the package convention `i∂ₜψ = … + F`). It is evaluated on the
host at every half-step, e.g. for moving wavepackets.

GGP's generic pump kernel is slow on CPUs for anything beyond an array lookup, so this
type overrides `evaluate_pump!`. On the CPU it fills columns with threads; on other array
types it uses one broadcast. The time loop (`_solve!`, `advance!`) passes `step!` the time
at the end of the step, as `GGP.solve!` v0.1.0 did, which puts the pump one step late;
`t_shift = -dt` compensates for it.
"""
struct PolaritonPump{AF,AS,P,T,E,EB}
    F::AF
    Fs::AS
    ω::Vector{Float64}          # probe frequencies (host, Float64)
    ph_host::Vector{ComplexF64}
    ph::P                       # phases e^{-iωt}, on the device of F, working precision
    A::T
    τ::T
    t_shift::T
    extra::E                    # nothing, or (x, t) -> Vector{<:Complex}
    x::Vector{Float64}
    ebuf::EB
end
function PolaritonPump(F, Fs, ω, A, τ, t_shift; extra=nothing, x=Float64[])
    ωv = ω === nothing ? Float64[] : Float64.(collect(ω isa Number ? [ω] : ω))
    ph = Fs === nothing ? nothing : similar(F, eltype(F), length(ωv))
    ebuf = extra === nothing ? nothing : similar(F)
    PolaritonPump(F, Fs, ωv, zeros(ComplexF64, length(ωv)), ph, A, τ, t_shift, extra, Float64.(x), ebuf)
end
(p::PolaritonPump)(r, param, t) = zero(eltype(p.F))        # only used by GGP to get the type

envelope(p::PolaritonPump, t) = iszero(p.A) ? one(t) : 1 + p.A * exp(-t / p.τ)

function GGP.get_pump_buffer(p::PolaritonPump, u, lengths, param, t)
    sz = p.Fs === nothing ? size(p.F) : size(p.Fs)
    similar(first(u), eltype(p.F), sz)
end

function GGP.evaluate_pump!(prob::GGP.GrossPitaevskiiProblem{N,M,T1,T2,T3,T4,T5,<:PolaritonPump}, dest, t) where {N,M,T1,T2,T3,T4,T5}
    p = prob.pump
    τ = Float64(t) + p.t_shift
    h = eltype(p.F)(envelope(p, τ))
    if p.Fs === nothing
        @. dest = -im * p.F * h
    else
        @. p.ph_host = cis(-p.ω * τ)
        copyto!(p.ph, p.ph_host)
        _fill_pump!(dest, p.F, p.Fs, p.ph, h)
    end
    if p.extra !== nothing
        copyto!(p.ebuf, p.extra(p.x, τ))
        dest .-= im .* p.ebuf
    end
    nothing
end

function _fill_pump!(dest::Array, F, Fs, ph, h)
    if ndims(Fs) == 1
        @inbounds @simd for i in eachindex(F)
            dest[i] = -im * (F[i] * h + Fs[i] * ph[1])
        end
    else
        Threads.@threads :static for b in axes(Fs, 2)
            phb = ph[b]
            @inbounds @simd for i in axes(Fs, 1)
                dest[i, b] = -im * (F[i] * h + Fs[i, b] * phb)
            end
        end
    end
end
# generic arrays (e.g. GPU): one broadcast
function _fill_pump!(dest::AbstractArray, F, Fs, ph, h)
    if ndims(Fs) == 1
        @. dest = -im * (F * h + Fs * $(Array(ph))[1])
    else
        @. dest = -im * (F * h + Fs * $(transpose(ph)))
    end
    nothing
end

# ---------------------------------------------------------------------------------------
# Equation terms for GGP (array lookups by grid index; work on CPU and GPU)
# ---------------------------------------------------------------------------------------

@inline _idx(r, p) = unsafe_trunc(Int, r[1] * p.invdx + 0.5) + 1

@inline function gpe_dispersion(k, p)
    k2 = k[1]^2
    if p.lp                                        # lower polariton (see `kinetic`)
        εC = p.dxc + p.aph * k2
        return (εC - sqrt(εC^2 + p.Ω2)) / 2 - p.e0 - p.δ0
    end
    p.ħ2m * k2 - p.δ0
end
gpe_potential(r, p) = (i = _idx(r, p); @inbounds p.V[i] - im * p.κ[i])
gpe_nonlinearity(ψ, p) = p.g * (abs2(first(ψ)) - p.shift)
gpe_noise(ψ, r, p) = (i = _idx(r, p); @inbounds sqrt(p.κ[i] * p.invdx))

function gpe_param(s::Setup, u0=s.F)
    R = real(eltype(u0))
    lp = !is_parabolic(s.pol); c = lp ? _lpconst(s.pol) : (; Ω=0.0, d=0.0, aph=0.0, e0=0.0)
    (; ħ2m=R(s.pol.ħ / (2s.pol.m)), lp, dxc=R(c.d), aph=R(c.aph), Ω2=R(c.Ω^2), e0=R(c.e0),
        δ0=R(detuning0(s)), g=R(s.pol.g), invdx=R(1 / s.grid.dx),
        shift=R(s.wigner_shift ? 1 / s.grid.dx : 0), V=_like(u0, s.V .+ s.Vres), κ=_like(u0, s.κ))
end

"""
    effective_dt(dt, ΔT)

Largest time step ≤ `dt` (up to rounding) that divides the save interval `ΔT`, and the
number of steps per save. GGP rounds the step count up, so the step passed on to it is
nudged to make sure it uses exactly this step.
"""
function effective_dt(dt, ΔT)
    n = max(1, round(Int, ΔT / dt))
    ΔT / n, n
end

"""
    evolve(s, u0, tspan; dt, nsaves=1, save_start=false, probe=nothing, noise=false,
           envelope=(A=0, τ=1), rng=nothing, show_progress=false)

Integrate the GPE of setup `s` from `u0` (an `N` vector, or an `N × B` matrix holding B
independent trajectories) over `tspan` (ps) with Strang splitting. It returns `(ts, ψ)`,
where `ψ` has size `(size(u0)..., nsaves + save_start)`.

- `probe`: a [`GaussianProbe`](@ref), or a vector of them (one per column of `u0`).
- `noise=true`: add truncated-Wigner noise √(κ/dx) ξ.
- `envelope`: start-up boost h(t) = 1 + A e^{-t/τ} of the static pump.
- `extra`: additional source `(x, t) -> F_extra` (see [`PolaritonPump`](@ref)).
- `pump_timing=:ggp` evaluates the pump one step late, like plain GGP v0.1.0. Use it only to
  reproduce runs made with it (e.g. arXiv:2512.17807).
"""
function evolve(s::Setup, u0::AbstractArray, tspan; dt, nsaves::Integer=1, save_start::Bool=false,
    probe=nothing, noise::Bool=false, envelope=(A=0.0, τ=1.0), extra=nothing, pump_timing::Symbol=:corrected,
    rng=nothing, show_progress::Bool=false)
    T = eltype(s.grid.x)
    ΔT = (tspan[2] - tspan[1]) / nsaves
    _dt, _ = effective_dt(dt, ΔT)
    prob = gpe_problem(s, u0, _dt; probe, noise, envelope, extra, pump_timing)
    iter = GGP.init(prob, GGP.StrangSplitting(), T.(tspan); dt=T(_dt * (1 + 1e-10)), nsaves,
        save_start, show_progress, rng)
    ts, sol = _solve!(iter)
    ts, first(sol)
end

# Fixed-step time loop, in place of `GGP.solve!`. GGP v0.1.0 advanced t before each step,
# calling `step!(iter, t, dt)` for the step from t − dt to t, so the pump was evaluated at
# t + dt/2 and t + dt, one step late; GGP ≥ 0.2 calls `step!(iter, t − dt, dt)`. This loop
# keeps the v0.1.0 sequence whatever the GGP version, and `PolaritonPump.t_shift` (−dt for
# :corrected, 0 for :ggp) sets the time at which the pump is actually evaluated. Results are
# bit-identical to those of `GGP.solve` v0.1.0.
function _solve!(iter)
    dt, ts, s0 = iter.dt, iter.ts, iter.save_start
    nd = ndims(first(iter.result))
    t = ts[1]
    for n in 1:(size(first(iter.result), nd) - s0)
        for _ in 1:iter.steps_per_save
            t += dt
            GGP.step!(iter, t, dt)
            GGP._next!(iter.progress)
        end
        for (r, u) in zip(iter.result, iter.u)
            copy!(selectdim(r, nd, n + s0), u)
        end
        ts[n+1] = t
    end
    GGP._finish!(iter.progress, iter.given_progress)
    ts[begin+1-s0:end], iter.result
end

"""
    gpe_problem(s, u0, dt; probe=nothing, noise=false, envelope=(A=0, τ=1))

The `GrossPitaevskiiProblem` of setup `s` for the exact time step `dt`. The pump-time
correction depends on dt.
"""
function gpe_problem(s::Setup, u0::AbstractArray, dt; probe=nothing, noise::Bool=false, envelope=(A=0.0, τ=1.0), extra=nothing,
    pump_timing::Symbol=:corrected)
    T = eltype(s.grid.x)
    Fs, ω = _probe_arrays(s, probe, u0)
    pump_timing in (:corrected, :ggp) || throw(ArgumentError("pump_timing must be :corrected or :ggp"))
    t_shift = pump_timing === :ggp ? 0.0 : Float64(-dt)        # :ggp reproduces runs made with plain GGP v0.1.0
    pump = PolaritonPump(_like(u0, s.F), _like(u0, Fs), ω, T(envelope.A), T(envelope.τ), t_shift; extra, x=s.grid.x)
    param = gpe_param(s, u0)
    u = (complex.(u0),)
    kw = noise ? (; position_noise_func=gpe_noise, noise_prototype=similar.(u)) : (;)
    GGP.GrossPitaevskiiProblem(u, (real(eltype(u0))(s.grid.L),); dispersion=gpe_dispersion, potential=gpe_potential,
        nonlinearity=gpe_nonlinearity, pump, param, kw...)
end

# move setup arrays to the array type (CPU/GPU) and element type of u0
_like(u0, ::Nothing) = nothing
_like(u0, x::Number) = real(eltype(u0))(x)
_like(u0, x::AbstractArray{<:Real}) = copyto!(similar(u0, real(eltype(u0)), size(x)), x)
_like(u0, x::AbstractArray{<:Complex}) = copyto!(similar(u0, complex(eltype(u0)), size(x)), x)

"""
    Stepper(s, u0, t0; dt, probe=nothing, noise=false, envelope=(A=0, τ=1), rng=nothing)

Time-stepping iterator for loops that process the field on the fly, without storing
snapshots. `advance!(st, n)` takes `n` steps of exactly `st.dt`, and `state(st)` is the
current field (a view of GGP's buffer: copy it if you need to keep it). `st.dt` is `dt`
itself; make it divide your sampling interval (see [`effective_dt`](@ref)).
"""
mutable struct Stepper{I,R}
    iter::I
    t::Float64          # time is always accumulated in Float64 (probe phases over ~10⁶ steps)
    dt::R               # step in the working precision of the field
end
function Stepper(s::Setup, u0::AbstractArray, t0; dt, probe=nothing, noise::Bool=false, envelope=(A=0.0, τ=1.0), extra=nothing, rng=nothing)
    R = real(eltype(u0))
    prob = gpe_problem(s, u0, dt; probe, noise, envelope, extra)
    # init on (0, dt), so that GGP derives exactly `dt`; then set the pump buffer to time t0.
    # PolaritonPump evaluates at t + t_shift = t − dt, hence the argument t0 + dt.
    iter = GGP.init(prob, GGP.StrangSplitting(), (zero(R), R(dt)); dt=R(dt), nsaves=1,
        save_start=false, show_progress=false, rng)
    @assert iter.dt == R(dt)
    GGP.evaluate_pump!(prob, iter.pump_buffer_next, Float64(t0) + dt)
    Stepper(iter, Float64(t0), R(dt))
end
function advance!(st::Stepper, n::Integer)
    for _ in 1:n
        st.t += st.dt
        GGP.step!(st.iter, st.t, st.dt)            # same sequence as `_solve!`
    end
    st
end
state(st::Stepper) = first(st.iter.u)

_probe_arrays(s, ::Nothing, u0) = (nothing, nothing)
_probe_arrays(s, p::GaussianProbe, u0) = (probe_profile(s.grid, p), p.ω)
function _probe_arrays(s, ps::AbstractVector{<:GaussianProbe}, u0)
    size(u0, 2) == length(ps) || throw(ArgumentError("one probe per column of u0 is required"))
    reduce(hcat, (probe_profile(s.grid, p) for p in ps)), [p.ω for p in ps]
end
