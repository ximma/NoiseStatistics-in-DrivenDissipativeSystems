# Quantum fluctuations in the Bogoliubov (linearised) approximation: input–output noise
# ports, frequency-resolved and equal-time covariances, and second-order correlations.
#
# Langevin model (amplitude damping κ = κ_r + κ_nr + κ_abs; see docs/methods.md):
#
#   dψ/dt = −i[H[ψ] − iκ]ψ − iF + Σ_c √(2κ_c) c_in(x, t)
#
# with one input port per loss channel c:
#   r   radiative, through the cavity mirror: vacuum (the pump and probe are its coherent part);
#   nr  non-radiative (excitonic) reservoir: thermal occupation n_x;
#   abs numerical absorbers at the box edges: vacuum (they stand in for the rest of an
#       infinite cavity).
# Output field through the mirror: A_out = √(2κ_r) ψ − A_in.
#
# Linearising around the stationary state, Φ = (δψ, δψ†) obeys dΦ/dt = −iMΦ + Ξ, with
# M = `bogoliubov_matrix(s, ψ₀)` and, in symmetric (Wigner) ordering,
#   ⟨Ξ(t) Ξ(t′)†⟩ = D δ(t − t′),  D = diag(d, d)/dx,  d = κ_r + κ_nr (1 + 2n_x) + κ_abs.
# For n_x = 0, d = κ: this is the truncated-Wigner noise √(κ/dx) ξ of the time stepper.
#
#   spectral covariance   S(ω) = G(ω) D G(ω)†,   G(ω) = (ω − M)⁻¹
#   equal-time covariance C = ∫ S(ω) dω/2π,      the solution of A C + C A† + D = 0, A = −iM
#
# C = ⟨Φ Φ†⟩_sym gives the normally ordered N(x, x′) = ⟨δψ†(x) δψ(x′)⟩ = C₁₁ᵀ − 1/(2dx)
# and M(x, x′) = ⟨δψ(x) δψ(x′)⟩ = C₁₂.

"""
    NoiseModel(s; radiative_fraction=1.0, n_x=0.0)

Loss channels and their input noise. The intrinsic damping γ/2 is split into a
radiative part `radiative_fraction·γ/2` (vacuum input through the mirror) and a
non-radiative part (input with thermal occupation `n_x`, scalar or vector). The
absorbers, `s.κ − γ/2`, take vacuum input. With the defaults, the noise is exactly the
truncated-Wigner noise of [`evolve`](@ref).
"""
struct NoiseModel{T}
    κr::Vector{T}
    κnr::Vector{T}
    nx::Vector{T}
    κabs::Vector{T}
end
function NoiseModel(s::Setup{T}; radiative_fraction=1.0, n_x=0.0) where {T}
    N = s.grid.N
    κi = s.pol.γ / 2
    NoiseModel{T}(fill(T(radiative_fraction * κi), N), fill(T((1 - radiative_fraction) * κi), N),
        n_x isa AbstractVector ? Vector{T}(n_x) : fill(T(n_x), N), s.κ .- κi)
end

"""Symmetric-order diffusion `d(x) = κ_r + κ_nr(1 + 2n_x) + κ_abs` of each grid point."""
diffusion(nm::NoiseModel) = nm.κr .+ nm.κnr .* (1 .+ 2 .* nm.nx) .+ nm.κabs

"""Diagonal of the Nambu noise matrix D (length 2N): `(d, d)/dx`."""
noise_diagonal(s::Setup, nm::NoiseModel=NoiseModel(s)) = (d = diffusion(nm) ./ s.grid.dx; vcat(d, d))

"""
    spectral_covariance(s, ψ0, ω; noise=NoiseModel(s), M=bogoliubov_matrix(s, ψ0))

Symmetrically ordered spectral covariance `S(ω) = G(ω) D G(ω)†` (2N × 2N) of
Φ = (δψ, δψ†), normalised so that `∫ S(ω) dω/2π` is the equal-time covariance.
`S[1:N, 1:N]` is the spectrum of δψ at +ω (lab frequency ω_p + ω); `S[N+1:2N, N+1:2N]` that
of δψ† at +ω, i.e. of δψ at −ω; the off-diagonal block holds the ±ω pair correlations.
One LU factorisation: O((2N)³).
"""
function spectral_covariance(s::Setup, ψ0::AbstractVector, ω; noise::NoiseModel=NoiseModel(s),
    M::AbstractMatrix=bogoliubov_matrix(s, ψ0))
    sd = sqrt.(noise_diagonal(s, noise))
    X = lu(ω * I - M) \ Matrix(Diagonal(complex.(sd)))
    X * X'
end

"""
    stationary_covariance(s, ψ0; noise=NoiseModel(s), M=bogoliubov_matrix(s, ψ0))

Equal-time symmetrically ordered covariance `C = ⟨Φ Φ†⟩` (2N × 2N), solution of the
Lyapunov equation `A C + C A† + D = 0` with A = −iM (Bartels–Stewart, Schur based: it
does not use the ill-conditioned eigenvector basis). Requires a stable operator (all
Im λ < 0).
"""
function stationary_covariance(s::Setup, ψ0::AbstractVector; noise::NoiseModel=NoiseModel(s),
    M::AbstractMatrix=bogoliubov_matrix(s, ψ0))
    C = lyap(-im .* Matrix(M), Matrix(Diagonal(complex.(noise_diagonal(s, noise)))))
    (C .+ C') ./ 2                                   # Hermitian up to round-off
end

"""
    Fluctuations(ψ0, N, M)

Gaussian (Bogoliubov) fluctuation state around the mean field ψ0, with normally ordered
`N[i, j] = ⟨δψᵢ† δψⱼ⟩` and `M[i, j] = ⟨δψᵢ δψⱼ⟩`. Build it with
`Fluctuations(s, ψ0, C)` from an equal-time covariance.
"""
struct Fluctuations{V,T}
    α::V
    N::T
    M::T
end
function Fluctuations(s::Setup, ψ0::AbstractVector, C::AbstractMatrix)
    n, dx = s.grid.N, s.grid.dx
    Fluctuations(Vector{ComplexF64}(ψ0), Matrix(transpose(C[1:n, 1:n])) - I / (2dx), C[1:n, n+1:2n])
end

"""
    window_matrix(w, N, dx)

Matrix `F` (Nw × N) of the windowed transform of [`windowed_fft`](@ref):
`ψ_w = F ψ`.
"""
function window_matrix(w::SpatialWindow, N::Integer, dx)
    Nw = length(w)
    E = zeros(ComplexF64, Nw, N)
    for n in 1:Nw
        E[n, w.first_idx+n-1] = w.w[n]
    end
    fft(E, 1)
end

"""
    transform(fl, Fa, Fb)

Mean fields and cross moments of the linear transforms `a = Fa ψ`, `b = Fb ψ`:
`(αa, αb, Nab, Mab, Naa_diag, Nbb_diag)` with `Nab[k, k′] = ⟨δa_k† δb_k′⟩`,
`Mab[k, k′] = ⟨δa_k δb_k′⟩`.
"""
function transform(fl::Fluctuations, Fa::AbstractMatrix, Fb::AbstractMatrix)
    αa, αb = Fa * fl.α, Fb * fl.α
    Nab = conj(Fa) * fl.N * transpose(Fb)
    Mab = Fa * fl.M * transpose(Fb)
    naa = real.(vec(sum((conj(Fa) * fl.N) .* Fa; dims=2)))
    nbb = real.(vec(sum((conj(Fb) * fl.N) .* Fb; dims=2)))
    αa, αb, Nab, Mab, naa, nbb
end

"""
    bogoliubov_g2m1(fl; Fa=I, Fb=I)

Normally ordered `g2 − 1` of two linear transforms of the field (identity: position space;
[`window_matrix`](@ref): windowed momentum space) for the Gaussian state `fl`, by Wick's
theorem:

    G2 − n_a n_b = 2Re[αa αb* Nab] + 2Re[αa* αb* Mab] + |Nab|² + |Mab|²,   n = |α|² + ⟨δ†δ⟩

Returns `(g2m1, na, nb)`. Comparable to [`g2m1`](@ref) of truncated-Wigner moments.
"""
function bogoliubov_g2m1(fl::Fluctuations; Fa=I, Fb=I)
    n = length(fl.α)
    Fa === I && (Fa = Matrix{ComplexF64}(I, n, n))
    Fb === I && (Fb = Matrix{ComplexF64}(I, n, n))
    αa, αb, Nab, Mab, naa, nbb = transform(fl, Fa, Fb)
    G = 2 .* real.((αa * αb') .* Nab) .+ 2 .* real.((conj.(αa) * transpose(conj.(αb))) .* Mab) .+ abs2.(Nab) .+ abs2.(Mab)
    na, nb = abs2.(αa) .+ naa, abs2.(αb) .+ nbb
    G ./ (na * transpose(nb)), na, nb
end
