# Truncated-Wigner (TWA) runs: many independent noisy trajectories, one sample per
# trajectory at the final time, accumulated into online moments.

"""
    TWAProtocol(; T=400, dt=0.2, envelope=(A=0, τ=1), pump_timing=:corrected, initial=:state,
                nsamples=1, Δt_sample=0)

One TWA trajectory: noisy evolution over `(0, T)` ps with time step `dt` and pump
envelope `envelope`, starting from the given state. `initial=:vacuum` also adds Wigner
vacuum noise `ξ/√(2dx)`.

With `nsamples = 1` (default) only the field at `T` is sampled, as in arXiv:2512.17807
([`gildeoliveira2026_protocol`](@ref)). With `nsamples > 1`, the trajectory continues and is sampled
`nsamples` times, every `Δt_sample` (starting at T). This is valid for stationary
geometries, and much cheaper per sample, but successive samples are correlated over
~1/γ. Choose Δt_sample accordingly; the statistical estimates must count trajectories,
not samples, for the worst case. The sampled path always uses the corrected pump timing.
"""
Base.@kwdef struct TWAProtocol
    T::Float64 = 400.0
    dt::Float64 = 0.2
    envelope::NamedTuple{(:A, :τ),Tuple{Float64,Float64}} = (A=0.0, τ=1.0)
    pump_timing::Symbol = :corrected
    initial::Symbol = :state
    nsamples::Int = 1
    Δt_sample::Float64 = 0.0
end

"""Protocol of arXiv:2512.17807: 400 ps from the steady state at dt = 0.2, start-up boost, GGP pump timing."""
gildeoliveira2026_protocol() = TWAProtocol(; T=400.0, dt=0.2, envelope=gildeoliveira2026_envelope, pump_timing=:ggp)

"""
    twa_batch(s, ψ0, B; protocol, rng=nothing, array=identity, T=Float64)

Run `B` independent trajectories in a single GGP batch and return their final fields
(N × B, on the device of `array`).
"""
function twa_batch(s::Setup, ψ0::AbstractVector, B::Integer; protocol::TWAProtocol=TWAProtocol(),
    rng=nothing, array=identity, T::Type{<:AbstractFloat}=Float64)
    pr = protocol
    u0 = array(repeat(Complex{T}.(ψ0), 1, B))
    if pr.initial === :vacuum
        ξ = similar(u0)
        rng === nothing ? randn!(ξ) : randn!(rng, ξ)
        u0 .+= ξ ./ T(sqrt(2s.grid.dx))
    end
    _, u = evolve(s, u0, (0.0, pr.T); dt=pr.dt, nsaves=1, noise=true, envelope=pr.envelope,
        pump_timing=pr.pump_timing, rng)
    dropdims(u; dims=3)
end

"""
    TWAAccumulator(s, window_pairs; position=true, array=identity, T=Float64)

Online moments for:
- the field in position space: `position=true` (all N points, N × N moments), a range of
  grid indices (moments over that range only, which is cheaper for large N), or `false`;
- each `(w1, w2)` pair of [`SpatialWindow`](@ref)s in momentum space.
"""
struct TWAAccumulator{P,M,W,R}
    position::P
    momentum::M
    windows::W
    prange::R
    rejected::Base.RefValue{Int}     # non-finite trajectories dropped (should stay 0)
    nupdates::Base.RefValue{Int}
end
function TWAAccumulator(s::Setup, window_pairs=(); position=true, array=identity, T::Type=Float64)
    N = s.grid.N
    prange = position === true ? (1:N) : position === false ? nothing : position
    pos = prange === nothing ? nothing : Moments(length(prange), length(prange); array, T)
    mom = [Moments(length(a), length(b); array, T) for (a, b) in window_pairs]
    TWAAccumulator(pos, mom, [(a, b) for (a, b) in window_pairs], prange, Ref(0), Ref(0))
end

function update!(acc::TWAAccumulator, X::AbstractMatrix)
    acc.nupdates[] += 1
    good = vec(all(isfinite, X; dims=1))
    if !all(good)
        nbad = count(!, Array(good))
        acc.rejected[] += nbad
        @warn "TWA: dropping $nbad non-finite trajectory sample(s) at update $(acc.nupdates[]) (total rejected $(acc.rejected[]))"
        any(good) || return acc
        X = X[:, findall(Array(good))]
    end
    if acc.position !== nothing
        Xp = acc.prange == axes(X, 1) ? X : X[acc.prange, :]
        update!(acc.position, Xp, Xp)
    end
    for (m, (a, b)) in zip(acc.momentum, acc.windows)
        a_, b_ = _on(X, a), _on(X, b)
        update!(m, windowed_fft(X, a_), windowed_fft(X, b_))
    end
    acc
end

"""
    twa_run!(acc, s, ψ0; ntraj, batch=1000, protocol, rng=nothing, array=identity, T=Float64, progress=false)

Accumulate `ntraj` trajectories into `acc`, in batches of `batch`. Returns `acc`. The run
can be resumed, and runs on different machines can be combined with [`merge`](@ref).
"""
function twa_run!(acc::TWAAccumulator, s::Setup, ψ0::AbstractVector; ntraj::Integer, batch::Integer=1000,
    protocol::TWAProtocol=TWAProtocol(), rng=nothing, array=identity, T::Type{<:AbstractFloat}=Float64, progress::Bool=false)
    done = 0
    while done < ntraj
        B = min(batch, ntraj - done)
        t = @elapsed if protocol.nsamples == 1
            update!(acc, twa_batch(s, ψ0, B; protocol, rng, array, T))
        else
            _twa_sampled!(acc, s, ψ0, B; protocol, rng, array, T)
        end
        done += B
        progress && @info "TWA: $done / $ntraj trajectories ($(round(t, digits=1)) s per batch of $B)"
    end
    acc
end

# time-sampled trajectories (stationary geometries): one Stepper, samples every Δt_sample
function _twa_sampled!(acc, s, ψ0, B; protocol, rng, array, T)
    pr = protocol
    pr.Δt_sample > 0 || throw(ArgumentError("nsamples > 1 requires Δt_sample > 0"))
    dt, nsub = effective_dt(pr.dt, pr.Δt_sample)
    u0 = array(repeat(Complex{T}.(ψ0), 1, B))
    if pr.initial === :vacuum
        ξ = similar(u0)
        rng === nothing ? randn!(ξ) : randn!(rng, ξ)
        u0 .+= ξ ./ T(sqrt(2s.grid.dx))
    end
    st = Stepper(s, u0, 0.0; dt, noise=true, envelope=pr.envelope, rng)
    advance!(st, round(Int, pr.T / dt))
    update!(acc, state(st))
    for _ in 2:pr.nsamples
        advance!(st, nsub)
        update!(acc, state(st))
    end
    acc
end
