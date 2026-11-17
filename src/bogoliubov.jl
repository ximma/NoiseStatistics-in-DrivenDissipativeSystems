# Bogoliubov spectrum of a stationary state.
#
# Linearising the GPE of `model.jl` around a stationary ψ₀, with
# δψ = u e^{−iωt} + v* e^{+iω*t}, gives (Eq. (4) of arXiv:2511.12339, written in the lab
# frame, without factoring out the phase of ψ₀):
#
#   ω u =  L u + g ψ₀² v
#   ω v = −L* v − g ψ₀*² u,      L = ħk²/2m − δ₀ + V − iκ + 2g|ψ₀|² − g s/dx
#
# L uses the same spectral kinetic operator and the same κ(x) and V(x) as the time
# stepper. The eigenvalues are therefore exactly the small-amplitude frequencies (in the
# pump frame) of the simulated model. Uniform losses give Im ω = −γ/2 for all modes.
# Deviations come from the absorbers, and from pairs of complex frequencies of the
# conservative problem: dynamical instabilities, or zero-norm modes such as quasinormal modes.

"""
    BogoliubovSpectrum

Eigenmodes of the Bogoliubov operator:
- `ω`: complex frequencies (ps⁻¹);
- `u`, `v`: mode functions as columns, normalised to Σ(|u|² + |v|²) dx = 1;
- `norm`: symplectic norm η = Σ(|u|² − |v|²) dx ∈ [−1, 1]. η > 0 means positive norm,
  η < 0 negative norm, η ≈ 0 zero norm;
- `x`: the grid.
"""
struct BogoliubovSpectrum{T}
    ω::Vector{Complex{T}}
    u::Matrix{Complex{T}}
    v::Matrix{Complex{T}}
    norm::Vector{T}
    x::Vector{T}
end

"""Dense spectral kinetic operator `kinetic(pol, −i∂ₓ)` (ħ(−i∂ₓ)²/2m by default) on the periodic grid (circulant)."""
function kinetic_matrix(s::Setup{T}) where {T}
    N = s.grid.N
    c = ifft(complex.(kinetic.(Ref(s.pol), s.grid.k)))
    [c[mod(i - j, N)+1] for i in 1:N, j in 1:N]
end

"""
    bogoliubov_matrix(s, ψ0)

The 2N × 2N matrix `[L gψ₀²; −gψ₀*² −L*]` acting on `(u, v)`.
"""
function bogoliubov_matrix(s::Setup{T}, ψ0::AbstractVector) where {T}
    N = s.grid.N
    g = s.pol.g
    shift = s.wigner_shift ? 1 / s.grid.dx : zero(T)
    L = kinetic_matrix(s)
    d = -detuning0(s) .+ s.V .+ s.Vres .- im .* s.κ .+ 2g .* abs2.(ψ0) .- g * shift
    for i in 1:N
        L[i, i] += d[i]
    end
    M = zeros(Complex{T}, 2N, 2N)
    M[1:N, 1:N] .= L
    M[N+1:2N, N+1:2N] .= -conj.(L)
    for i in 1:N
        M[i, N+i] = g * ψ0[i]^2
        M[N+i, i] = -g * conj(ψ0[i])^2
    end
    M
end

"""
    bogoliubov_spectrum(s, ψ0)

Full dense diagonalisation of the Bogoliubov operator (2N × 2N; about 1 min for
N = 2048). It returns a [`BogoliubovSpectrum`](@ref). The spectrum is symmetric under
(ω, u, v) ↦ (−ω*, v*, u*); usually only modes with Re ω ≥ 0 are needed.
"""
function bogoliubov_spectrum(s::Setup{T}, ψ0::AbstractVector) where {T}
    N, dx = s.grid.N, s.grid.dx
    F = eigen(bogoliubov_matrix(s, ψ0))
    u = F.vectors[1:N, :]
    v = F.vectors[N+1:2N, :]
    nu = vec(sum(abs2, u; dims=1)) .* dx
    nv = vec(sum(abs2, v; dims=1)) .* dx
    tot = nu .+ nv
    u ./= sqrt.(tot')
    v ./= sqrt.(tot')
    BogoliubovSpectrum{T}(F.values, u, v, (nu .- nv) ./ tot, copy(s.grid.x))
end

"""
    mode_weight(sp, j, x_beg, x_end)

Fraction of the weight Σ(|u|² + |v|²) of mode `j` located in `x_beg ≤ x ≤ x_end`.
"""
function mode_weight(sp::BogoliubovSpectrum, j::Integer, x_beg, x_end)
    r = findall(x -> x_beg <= x <= x_end, sp.x)
    w = abs2.(sp.u[:, j]) .+ abs2.(sp.v[:, j])
    sum(w[r]) / sum(w)
end

"""
    find_qnm(sp, x_H; window=30, ωmin=0, damping=γ/2, ...)

Candidate quasinormal modes: modes with Re ω > `ωmin` whose weight within `x_H ± window`
exceeds `loc` (default 0.2) and whose symplectic norm satisfies |η| < `zero_norm` (default 0.3).
Returns the indices, sorted by decreasing localisation.
"""
function find_qnm(sp::BogoliubovSpectrum, x_H; window=30.0, ωmin=0.0, loc=0.2, zero_norm=0.3)
    idx = [j for j in eachindex(sp.ω) if real(sp.ω[j]) > ωmin && abs(sp.norm[j]) < zero_norm &&
                                          mode_weight(sp, j, x_H - window, x_H + window) > loc]
    sort(idx; by=j -> -mode_weight(sp, j, x_H - window, x_H + window))
end
