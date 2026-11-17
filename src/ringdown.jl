# Quasinormal-mode ringdown: a wavepacket is sent towards the horizon, and the free decay
# of the density at the defect is analysed. This follows the protocol of arXiv:2511.12339,
# with the nonlinear fit replaced by a matrix-pencil extraction of the
# complex frequencies.

"""
    WavePacketSource(; x_stop, k0, wk, F, c, t_center)

Moving source of `simulation_QNM_wavepacket.m`:

    WP(x, t) = F exp(−(wk/2)² ξ²) e^{i k0 ξ} Θ(x_stop − x),   ξ = x − x_stop − c (t − t_center)

The envelope moves at speed `c` and its centre reaches `x_stop` at `t = t_center`. The
source exists only upstream of `x_stop`. The MATLAB code adds `δt·WP` to ψ, so in the
package convention (i∂ₜψ = … + F) the source is `F = i WP`. Calling the source as
`src(x, t)` returns that vector; pass it to `evolve(...; extra=src)`.
"""
Base.@kwdef struct WavePacketSource
    x_stop::Float64
    k0::Float64
    wk::Float64
    F::Float64
    c::Float64
    t_center::Float64
end
function (w::WavePacketSource)(x::AbstractVector, t)
    map(x) do xi
        ξ = xi - w.x_stop - w.c * (t - w.t_center)
        xi <= w.x_stop ? im * w.F * exp(-(w.wk / 2)^2 * ξ^2) * cis(w.k0 * ξ) : zero(ComplexF64)
    end
end

"""Half-length (µm) of the packet at 1% of its maximum, and the time it takes to pass at `c`."""
packet_halfwidth(w::WavePacketSource) = 2 / w.wk * sqrt(log(100))
packet_front_time(w::WavePacketSource) = packet_halfwidth(w) / w.c

"""
    ringdown(s, ψss, src; t_end, Δt=0.1, dt=0.01, x_probe=src.x_stop)

Evolve from the steady state with the extra source `src` over `[0, t_end]`. It returns
`(t, δn)`, the relative density change `(|ψ(x_probe, t)|² − |ψss(x_probe)|²)/n̄` sampled
every `Δt`, where n̄ is the mean density. The probe point is MATLAB's `floor(x_d/dx)`
(1-based).
"""
function ringdown(s::Setup, ψss::AbstractVector, src; t_end, Δt=0.1, dt=0.01, x_probe=src.x_stop)
    nsaves = round(Int, t_end / Δt)
    ts, ψ = evolve(s, ψss, (0.0, nsaves * Δt); dt, nsaves, extra=src)
    i = floor(Int, x_probe / s.grid.dx)
    nbar = mean(abs2, ψss)
    ts, (abs2.(ψ[i, :]) .- abs2(ψss[i])) ./ nbar
end

"""
    matrix_pencil(y, Δt; nmodes=nothing, tol=1e-6, L=length(y)÷3)

Matrix-pencil method (Hua & Sarkar 1990): fit `y_n ≈ Σ_j a_j e^{−iω_j nΔt}` with complex
frequencies `ω_j`, with no initial guess. The model order is `nmodes`, or else the number
of singular values above `tol × σ_max`. Returns `(ω, a)`, sorted by decreasing |a|.
A real signal gives pairs `ω, −ω*`; decay corresponds to `Im ω < 0`.
"""
function matrix_pencil(y::AbstractVector, Δt; nmodes=nothing, tol=1e-6, L::Integer=length(y) ÷ 3)
    n = length(y)
    Y = [complex(y[i+j-1]) for i in 1:n-L, j in 1:L+1]
    F = svd(Y)
    M = nmodes === nothing ? count(>(tol * F.S[1]), F.S) : nmodes
    V = F.V[:, 1:M]
    V1, V2 = V[1:end-1, :], V[2:end, :]
    z = eigvals(pinv(V1) * V2)
    ω = im .* log.(z) ./ Δt                      # z = e^{−iωΔt}
    Z = [zj^(k - 1) for k in 1:n, zj in z]
    a = Z \ complex.(y)
    p = sortperm(abs.(a); rev=true)
    ω[p], a[p]
end
