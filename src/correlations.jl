# Second-order correlations from truncated-Wigner samples (SM of arXiv:2512.17807).
#
# Wigner averages are symmetrically ordered. For two (windowed) field operators a, b with
# commutators [a_j, a_j†] = c_aa(j), [b_k, b_k†] = c_bb(k) and [a_j, b_k†] = c_ab(j, k):
#
#   n_a(j)       = ⟨|a_j|²⟩_W − c_aa/2                     (normally ordered population)
#   G2(j,k) − n_a n_b = Cov_W(|a_j|², |b_k|²) − Re[c_ab* ⟨a_j b_k*⟩_W] + |c_ab|²/4
#
# and g2 − 1 = [G2 − n_a n_b]/(n_a n_b). This is SM Eq. (18). In position space,
# a = b = ψ(x), c = δ_{xx'}/dx (SM Eq. (12)).

"""
    SpatialWindow(w, first_idx)
    SpatialWindow(grid, x_begin, x_end; shape=:hann)

A window of length `length(w)` starting at grid index `first_idx`. The windowed transform
is `ψ_w(k_m) = Σ_n w_n ψ(x_{first_idx+n}) e^{−i k_m n dx}`, with `k_m = fftfreq(Nw, 2π/dx)`
and x measured from the window start, as in the code of arXiv:2512.17807. The geometric
constructor picks the grid points nearest to `x_begin` and `x_end`. It uses the periodic
Hann shape `sin²(πn/Nw)`, n = 0…Nw−1, with Nw = last − first index (as `windows.jl`).
"""
struct SpatialWindow{V<:AbstractVector}
    w::V
    first_idx::Int
end
function SpatialWindow(grid::Grid, x_begin, x_end; shape::Symbol=:hann)
    i1 = argmin(abs.(grid.x .- x_begin))
    i2 = argmin(abs.(grid.x .- x_end))
    Nw = i2 - i1
    w = shape === :hann ? [sinpi(n / Nw)^2 for n in 0:Nw-1] : ones(Nw)
    SpatialWindow(w, i1)
end
Base.length(w::SpatialWindow) = length(w.w)
window_k(w::SpatialWindow, dx) = fftfreq(length(w), 2π / dx)
_on(x::AbstractArray, w::SpatialWindow) = (w2 = similar(x, real(eltype(x)), length(w.w)); copyto!(w2, w.w); SpatialWindow(w2, w.first_idx))

"""Windowed FFT along the first dimension of `X` (N × B): returns Nw × B, unnormalised."""
windowed_fft(X::AbstractMatrix, w::SpatialWindow) = fft(view(X, w.first_idx:w.first_idx+length(w)-1, :) .* w.w, 1)

"""
    momentum_commutators(w1, w2, dx)

`(c11, c22, c12)` for two windowed transforms (SM Eqs. (15)–(16)):
- `c11 = Σ|w1|²/dx` and `c22 = Σ|w2|²/dx` (scalars);
- `c12[k, k′] = Σ_x w1 w2* e^{−ik(x−x₁)} e^{+ik′(x−x₂)}/dx` over the overlap of the windows,
  where x₁ and x₂ are the window starts.
"""
function momentum_commutators(w1::SpatialWindow, w2::SpatialWindow, dx)
    c11 = sum(abs2, w1.w) / dx
    c22 = sum(abs2, w2.w) / dx
    N1, N2 = length(w1), length(w2)
    c12 = zeros(ComplexF64, N1, N2)
    for n′ in 1:N2
        n = n′ + w2.first_idx - w1.first_idx
        1 <= n <= N1 && (c12[n, n′] = w1.w[n] * conj(w2.w[n′]) / dx)
    end
    fft!(c12, 1)
    bfft!(c12, 2)
    c11, c22, c12
end

"""Position-space commutators: `(1/dx, 1/dx, I/dx)`."""
position_commutators(N, dx) = (1 / dx, 1 / dx, Matrix{Float64}(I, N, N) ./ dx)

"""
    Moments(na, nb; array=identity)

Online Wigner moments of two fields a (length `na`) and b (length `nb`):
- `μa = ⟨|a|²⟩` and `μb = ⟨|b|²⟩`;
- `G1 = ⟨a b*⟩`;
- `M2`, the co-moment sum of (|a|², |b|²), so that Cov = M2/n.

Batches are merged with Chan's parallel update. It is exact, numerically stable, and uses
only matrix products, so it works on GPU arrays (`array=CuArray`).
"""
mutable struct Moments{V,C,M}
    n::Int
    μa::V
    μb::V
    G1::C
    M2::M
end
function Moments(na::Integer, nb::Integer; array=identity, T::Type=Float64)
    Moments(0, array(zeros(T, na)), array(zeros(T, nb)), array(zeros(Complex{T}, na, nb)), array(zeros(T, na, nb)))
end

"""Add a batch: columns of `A` (na × B) and `B` (nb × B) are paired samples of a and b."""
function update!(m::Moments, A::AbstractMatrix, Bm::AbstractMatrix)
    nb_ = size(A, 2)
    Ia, Ib = abs2.(A), abs2.(Bm)
    μa_b = vec(sum(Ia; dims=2)) ./ nb_
    μb_b = vec(sum(Ib; dims=2)) ./ nb_
    Ca, Cb = Ia .- μa_b, Ib .- μb_b
    n = m.n + nb_
    δa, δb = μa_b .- m.μa, μb_b .- m.μb
    m.M2 .+= Ca * transpose(Cb) .+ (δa * transpose(δb)) .* (m.n * nb_ / n)
    m.μa .+= δa .* (nb_ / n)
    m.μb .+= δb .* (nb_ / n)
    m.G1 .+= (A * Bm' ./ nb_ .- m.G1) .* (nb_ / n)
    m.n = n
    m
end

"""Merge two independent moment sets (e.g. from different machines)."""
function merge(m1::Moments, m2::Moments)
    n = m1.n + m2.n
    δa, δb = m2.μa .- m1.μa, m2.μb .- m1.μb
    Moments(n, m1.μa .+ δa .* (m2.n / n), m1.μb .+ δb .* (m2.n / n),
        m1.G1 .+ (m2.G1 .- m1.G1) .* (m2.n / n), m1.M2 .+ m2.M2 .+ (δa * transpose(δb)) .* (m1.n * m2.n / n))
end

"""Normally ordered populations `(n_a, n_b)`."""
populations(m::Moments, (c11, c22, c12)) = (Array(m.μa) .- c11 / 2, Array(m.μb) .- c22 / 2)

"""
    g2m1(m, commutators)

`g2 − 1` (SM Eq. (18) divided by n_a n_b) from Wigner moments. Use [`momentum_commutators`](@ref)
for windowed transforms and [`position_commutators`](@ref) for g2(x, x′).
"""
function g2m1(m::Moments, (c11, c22, c12))
    na, nb = populations(m, (c11, c22, c12))
    G1, σ² = Array(m.G1), Array(m.M2) ./ m.n
    (σ² .- real.(conj.(c12) .* G1) .+ abs2.(c12) ./ 4) ./ (na * transpose(nb))
end

"""Normally ordered connected correlation G2 − n_a n_b (not normalised)."""
function connected_G2(m::Moments, (c11, c22, c12))
    Array(m.M2) ./ m.n .- real.(conj.(c12) .* Array(m.G1)) .+ abs2.(c12) ./ 4
end
