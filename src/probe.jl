# Stimulated scattering: a weak monochromatic probe is injected into the steady flow, and
# its frequency-resolved, windowed spatial spectrum is read out on both sides of the horizon.

"""
    ProbeProtocol(; t_relax=400, nsamples=3000, Δt=0.75, dt=0.01)

Timing of a probe run: relax for `t_relax` ps after switching the probe on, then record
`nsamples` snapshots every `Δt` ps (frequency resolution 2π/(nsamples Δt)). `t_relax`,
`nsamples` and `Δt` are those of arXiv:2511.12339. The paper used dt = 0.005 ps with a
first-order scheme. With Strang splitting, dt = 0.01 changes the readouts by ≤ 1e-3, 3×
less than the original's own time-step error.
"""
Base.@kwdef struct ProbeProtocol{T}
    t_relax::T = 400.0
    nsamples::Int = 3000
    Δt::T = 0.75
    dt::T = 0.01
end

frequency_resolution(p::ProbeProtocol) = 2π / (p.nsamples * p.Δt)

"""
    probe_samples(s, ψss, probe; protocol=ProbeProtocol())

Start from the steady state `ψss` and switch on `probe` (one [`GaussianProbe`](@ref), or a
vector of probes run as independent batch columns). Return the deviations `ψ(t_j) − ψss`
as an `N × nsamples` matrix, or `N × B × nsamples` for B probes.
"""
function probe_samples(s::Setup, ψss::AbstractVector, probe; protocol::ProbeProtocol=ProbeProtocol())
    pr = protocol
    B = probe isa AbstractVector ? length(probe) : 1
    u0 = B == 1 && !(probe isa AbstractVector) ? copy(ψss) : repeat(ψss, 1, B)
    _, u1 = evolve(s, u0, (0.0, pr.t_relax); dt=pr.dt, nsaves=1, probe)
    u1 = selectdim(u1, ndims(u1), 1)
    t0 = pr.t_relax
    _, samples = evolve(s, u1, (t0, t0 + pr.nsamples * pr.Δt); dt=pr.dt, nsaves=pr.nsamples, probe)
    samples .- ψss
end

"""
    probe_readout(grid, samples, ω, Δt; windows=(up=(70, 370), down=(401, 601)), convention=:grid)

For each spatial window, the windowed spatial spectra of the components oscillating at
`+ω` (e^{−iωt}: the probed frequency) and `−ω` (e^{+iωt}: the conjugate, v-type response).
The time series is Hann-windowed before demodulation. It returns a NamedTuple
`(k, up=(plus, minus), down=(plus, minus))` of |·|² spectra on `kaxis(grid)`.
"""
function probe_readout(grid::Grid, samples::AbstractMatrix, ω, Δt;
    windows=(up=(70.0, 370.0), down=(401.0, 601.0)), convention::Symbol=:grid)
    ψp = demodulate(samples, ω, Δt)
    ψm = demodulate(samples, -ω, Δt)
    out = map(windows) do (a, b)
        w = hann_window(grid, a, b; convention)
        (plus=abs2.(spatial_spectrum(ψp, w)), minus=abs2.(spatial_spectrum(ψm, w)))
    end
    (; k=kaxis(grid), out...)
end

# ---------------------------------------------------------------------------------------
# Online (memory-free) and batched probe runs
# ---------------------------------------------------------------------------------------

"""
    probe_demodulated(s, ψss, probes; protocol=ProbeProtocol(), T=Float64)

Same physics as [`probe_samples`](@ref) followed by [`demodulate`](@ref) at ±ω, but:
- all `probes` (a vector of [`GaussianProbe`](@ref)) run at once, as columns of a single
  `N × B` batch, each with its own frequency;
- the demodulation sums `Σ_j w_j (ψ_j − ψss) e^{±iω t_j}` are accumulated while stepping,
  so no snapshots are stored (memory O(N·B) instead of O(N·B·nsamples));
- `T=Float32` runs the field in single precision. Time and phases stay Float64.
  **Not suitable for weak probes**: Float32 rounding errors, re-injected at every step,
  accumulate over a damping time to |δψ|/|ψ| ~ ε₃₂/(κ dt) ~ 1e-3. In `burkhard2026()` this is
  comparable to the probe response, and readouts are off by 10–15%. Use Float64 (the default).

Returns `(plus, minus)`, two `N × B` matrices (column b: probe b demodulated at ±ω_b).
"""
function probe_demodulated(s::Setup, ψss::AbstractVector, probes::AbstractVector{<:GaussianProbe};
    protocol::ProbeProtocol=ProbeProtocol(), T::Type{<:AbstractFloat}=Float64, array=identity)
    pr = protocol
    B = length(probes)
    T === Float32 && @warn "Float32 probe runs are unreliable for weak probes (see docstring)" maxlog = 1
    dt, nsub = effective_dt(pr.dt, pr.Δt)
    ψ0 = array(Complex{T}.(ψss))
    u = repeat(ψ0, 1, B)
    st = Stepper(s, u, 0.0; dt, probe=probes)
    nrelax = round(Int, pr.t_relax / dt)
    advance!(st, nrelax)
    ω = Float64[p.ω for p in probes]
    plus = fill!(similar(u), 0)
    minus = fill!(similar(u), 0)
    w = hann(pr.nsamples)
    for j in 1:pr.nsamples
        advance!(st, nsub)
        ψ = state(st)
        tj = j * pr.Δt                                   # same phase origin as `demodulate`
        php = array(transpose(Complex{T}.(w[j] .* cis.(ω .* tj))))    # phases in Float64, then cast
        phm = array(transpose(Complex{T}.(w[j] .* cis.(-ω .* tj))))
        @. plus += (ψ - ψ0) * php
        @. minus += (ψ - ψ0) * phm
    end
    plus, minus
end

"""
    probe_scan(s, ψss, probes; batch=16, windows=(up=(70,370), down=(401,601)),
               convention=:grid, protocol=ProbeProtocol(), T=Float64)

Probe scan over `probes` in batches of `batch` columns, demodulated online. Returns
`(k, up=(plus, minus), down=(plus, minus))`, where each entry is an `N × length(probes)`
matrix of |·|² spatial spectra on `kaxis(grid)`. These are the rows of Fig. 2a–c of
arXiv:2511.12339 before normalisation. The time-phase origin is the one of
[`demodulate`](@ref), so `|·|²` is identical to `probe_readout(probe_samples(…))`.
"""
function probe_scan(s::Setup, ψss::AbstractVector, probes::AbstractVector{<:GaussianProbe};
    batch::Integer=16, windows=(up=(70.0, 370.0), down=(401.0, 601.0)), convention::Symbol=:grid,
    protocol::ProbeProtocol=ProbeProtocol(), T::Type{<:AbstractFloat}=Float64, progress::Bool=false)
    N, P = s.grid.N, length(probes)
    W = map(((a, b),) -> hann_window(s.grid, a, b; convention), windows)
    out = map(_ -> (plus=zeros(N, P), minus=zeros(N, P)), windows)
    for r in Iterators.partition(1:P, batch)
        t = @elapsed (plus, minus) = probe_demodulated(s, ψss, probes[r]; protocol, T)
        for (name, w) in pairs(W)
            out[name].plus[:, r] .= abs2.(fftshift(fft(Array(plus) .* w, 1), 1))
            out[name].minus[:, r] .= abs2.(fftshift(fft(Array(minus) .* w, 1), 1))
        end
        progress && @info "probes $(first(r))–$(last(r)) of $P in $(round(t, digits=1)) s"
    end
    (; k=kaxis(s.grid), out...)
end

"""
    incident_probes(s, up, ωs; x0, σ=5.0, F, krange=(-3, 3))

Probes exactly on resonance with the incoming upstream mode `in`. For each frequency ω in
`ωs` (ps⁻¹), the probe wavevector is the root of the upstream positive-norm LDA branch
(`up`, a [`LocalFluid`](@ref)) with positive group velocity, towards the horizon at larger
x, closest to the fluid's own wavevector. Frequencies below the upstream gap (no such
root) are skipped. Returns a vector of [`GaussianProbe`](@ref).
"""
function incident_probes(s::Setup, up::LocalFluid, ωs; x0, σ=5.0, F, krange=(-3.0, 3.0))
    probes = GaussianProbe{Float64}[]
    for ω in ωs
        ks = filter(k -> group_velocity(up, k, +1) > 0, lda_roots(up, ω, +1; krange))
        isempty(ks) && continue
        k = ks[argmin(abs.(ks .- up.k0))]
        push!(probes, GaussianProbe(; x0, σ, k, ω, F))
    end
    probes
end
