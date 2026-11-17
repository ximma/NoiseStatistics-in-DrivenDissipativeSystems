# Intensity-noise spectra of the light scattered from a coherent probe (self-homodyne
# detection), in the linearised theory of noise.jl. Derivation: docs/methods.md.
#
# Coherent output (transmission through a mirror of rate κ_d): B(x, t) = √(2κ_d) β(x, t),
# with three frequency components in the pump frame,
#   ψ₀ at ν = 0,   u at ν = +ω_s (probe band),   v* at ν = −ω_s (idler band),
# where (u, v) = `respond(lr, probe)`. Detector j is an orthogonal projector Π_j (aperture,
# then Fourier-plane slit). To first order in the fluctuations, its photocurrent is
#   δI_j(Ω) = Σ_ν w_jν† Φ_out(ν + Ω),
#   w_jν = dx·(Π_j B_c, 0) for each component c at ν = ω_c, and dx·(0, (Π_j B_c)*) at ν = −ω_c,
# with Φ_out = K Φ − Ξ_d (K = √(2κ_d)), whose symmetrised spectrum is
#   S_out(ω) = K G D G† K + [1 − iKGK + iKG†K]/(2dx),   G = (ω − M)⁻¹.
# Fluctuations at different frequencies are uncorrelated, so
#   S_jk(Ω) = Σ_ν w_jν† S_out(ν + Ω) w_kν
# and only adjoint solves X = G(ν + Ω)† K W are needed. Shot noise is S_jj = I_j.

"""
    Detector(name, s, (xa, xb), (ka, kb))

Ideal detection channel:
- a hard real-space aperture x ∈ [xa, xb] (length L_a);
- then a Fourier-plane slit that keeps the aperture's own Fourier modes
  e_q(x) = e^{ik_q x}/√L_a, k_q = 2πq/L_a, with ka ≤ k_q ≤ kb.

The modes are orthonormal (Σ_x e_q* e_q′ dx = δ_qq′), so the filter is an orthogonal
projector. The detected photon flux is Σ_q |c_q|², with c_q = Σ_x e_q* A_out dx.
Detectors with disjoint apertures, or disjoint k-bands in the same aperture, are
orthogonal; no vacuum enters at the filters.
"""
struct Detector
    name::Symbol
    idx::UnitRange{Int}
    k::Vector{Float64}
    E::Matrix{ComplexF64}
    dx::Float64
end
function Detector(name, s::Setup, xr, kr)
    x, dx = s.grid.x, s.grid.dx
    idx = findfirst(>=(xr[1]), x):findlast(<=(xr[2]), x)
    La = length(idx) * dx
    qs = ceil(Int, kr[1] * La / 2π):floor(Int, kr[2] * La / 2π)
    length(qs) <= length(idx) || throw(ArgumentError("k-band wider than the aperture's Brillouin zone"))
    k = 2π .* collect(qs) ./ La
    Detector(Symbol(name), idx, k, [cis(kq * x[i]) / sqrt(La) for i in idx, kq in k], dx)
end
"""Mode coefficients c_q = Σ_x e_q*(x) f(x) dx of a grid field in detector `d`."""
coefficients(d::Detector, f::AbstractVector) = d.E' * f[d.idx] .* d.dx
"""Orthogonal projection Π f of a grid field onto the modes of detector `d` (grid vector)."""
function project(d::Detector, f::AbstractVector)
    p = zeros(ComplexF64, length(f))
    p[d.idx] .= d.E * coefficients(d, f)
    p
end

"""
    OutputNoise(s, ψ0; noise=NoiseModel(s), output_fraction=0.5, M=bogoliubov_matrix(s, ψ0))

Output-noise solver around the stationary state `ψ0`. The detected mirror has rate
κ_d = `output_fraction`·κ_r. For a symmetric cavity seen in transmission this is 1/2; with
1, all radiative loss is detected. One O(N³) Hessenberg reduction of M†; each frequency
then costs O(N²) per detector.
"""
struct OutputNoise{T,H}
    s::Setup{T}
    ψ0::Vector{Complex{T}}
    K::Vector{T}
    D::Vector{T}
    hadj::H
end
function OutputNoise(s::Setup, ψ0::AbstractVector; noise::NoiseModel=NoiseModel(s), output_fraction=0.5,
    M=bogoliubov_matrix(s, ψ0))
    κd = output_fraction .* noise.κr
    OutputNoise(s, Vector{ComplexF64}(ψ0), sqrt.(2 .* vcat(κd, κd)), noise_diagonal(s, noise),
        hessenberg(Matrix(M')))
end

"""
    coherent_components(on, u, v, ωs; pump=false)

Coherent output fields `(ν, B)` with B = √(2κ_d)·β: `(+ωs, K u)` and `(−ωs, K v*)`, plus
`(0, K ψ₀)` if `pump`. Leaving the pump out models a detection path that rejects light at
the pump frequency (an etalon or notch at ω_p).
"""
function coherent_components(on::OutputNoise, u, v, ωs; pump::Bool=false)
    N = on.s.grid.N
    k = on.K[1:N]
    c = [(float(ωs), k .* u), (-float(ωs), k .* conj.(v))]
    pump && push!(c, (0.0, k .* on.ψ0))
    c
end

"""
    self_homodyne(on, dets, components; Ω=0)

Mean photon fluxes `I` (photons/ps) and the cross-spectral matrix `S` (photons/ps,
two-sided) of the photocurrents of detectors `dets` at analysis frequency Ω (ps⁻¹).
`components` comes from [`coherent_components`](@ref).

- Linear order in the fluctuations: valid when the coherent flux in each detector
  dominates its spontaneous (incoherent) flux.
- Shot noise is `S[j, j] = I[j]`.
- Beat notes between components (at ω_s, 2ω_s, tens of GHz) are outside the
  photodetection band and drop out of `I`.
"""
function self_homodyne(on::OutputNoise, dets::AbstractVector{Detector}, components; Ω=0.0)
    N, dx = on.s.grid.N, on.s.grid.dx
    J = length(dets)
    flux = [sum(sum(abs2, coefficients(d, B)) for (_, B) in components) for d in dets]
    νs = unique(ν for (ω, _) in components for ν in (ω, -ω))
    proj = [[dx .* project(d, B) for (_, B) in components] for d in dets]
    S = zeros(ComplexF64, J, J)
    for ν in νs
        W = zeros(ComplexF64, 2N, J)
        for j in 1:J, (c, (ωc, _)) in enumerate(components)
            isapprox(ωc, ν; atol=1e-12) && (W[1:N, j] .+= proj[j][c])
            isapprox(-ωc, ν; atol=1e-12) && (W[N+1:2N, j] .+= conj.(proj[j][c]))
        end
        KW = on.K .* W
        X = -((on.hadj - conj(ν + Ω) * I) \ KW)                 # G(ν + Ω)† K W
        S .+= X' * (on.D .* X) .+ (W' * W .- im .* (X' * KW) .+ im .* (KW' * X)) ./ (2dx)
    end
    flux, S
end

"""
    spontaneous_flux(on, fl, d)

Normally ordered photon flux (photons/ps) of the spontaneous emission (Hawking pairs and
other incoherent scattering) in detector `d`, all frequencies, from the equal-time
moments `fl::Fluctuations`. The linear self-homodyne regime needs a coherent flux well
above this.
"""
function spontaneous_flux(on::OutputNoise, fl::Fluctuations, d::Detector)
    E = d.E .* on.K[d.idx] .* d.dx                               # rows: grid points, cols: modes
    real(tr(transpose(E) * fl.N[d.idx, d.idx] * conj(E)))
end

"""
    with_efficiency(I, S, η)

Detection efficiency η (scalar or vector over the detectors), modelled as a beamsplitter
that mixes in vacuum: I → ηI and S → η_jη_k S + δ_jk η_j(1 − η_j) I_j.
"""
function with_efficiency(I, S, η)
    e = η isa AbstractVector ? η : fill(η, length(I))
    e .* I, (e * e') .* S .+ Diagonal(e .* (1 .- e) .* I)
end

"""Fano factors S_jj/I_j: 1 is shot noise, < 1 is sub-Poissonian."""
fano(I, S) = real.(diag(S)) ./ I

"""Normalised intensity correlation C_jk = S_jk/√(S_jj S_kk) at Ω → 0 (real part)."""
intensity_correlation(S, j, k) = real(S[j, k]) / sqrt(real(S[j, j]) * real(S[k, k]))

"""
    conditional_variance(I, S, j, ks)

Noise of detector `j` after optimal linear subtraction of detectors `ks`, normalised to
the shot noise of `j`: `(S_jj − S_jK S_KK⁻¹ S_Kj)/I_j`. A value below 1 means quantum
correlations (a quantum-non-demolition-type criterion).
"""
function conditional_variance(I, S, j, ks)
    Sr = real.(S)
    (Sr[j, j] - (Sr[j:j, ks] * pinv(Sr[ks, ks]) * Sr[ks, j:j])[1]) / I[j]
end

"""
    gemellity(I, S, j, k)

Intensity-difference noise of beams j and k after balancing (the brighter beam is
attenuated, which mixes in vacuum), normalised to the shot noise of the sum (Karr et al.,
PRA 69, 063807 (2004)). A value below 1 means twin-beam (quantum) correlation.
"""
function gemellity(I, S, j, k)
    a, b = I[j] >= I[k] ? (j, k) : (k, j)
    τ = I[b] / I[a]
    Saa = τ^2 * real(S[a, a]) + τ * (1 - τ) * I[a]
    (Saa + real(S[b, b]) - 2τ * real(S[a, b])) / (2I[b])
end

# ---------------------------------------------------------------------------------------
# Spontaneous (incoherent) light in the detectors: the terms quadratic in the fluctuations.
#
# The detected modes c_q = Σ_x e_q* A_out dx are Gaussian with zero mean, with normally
# ordered spectra (output fields are free fields: [c, c†] = δ, [c, c] = 0)
#   ⟨c_q†(ω) c_q′(ω′)⟩ = 2π δ(ω − ω′) n_qq′(ω),   ⟨c_q(ω) c_q′(ω′)⟩ = 2π δ(ω + ω′) m_qq′(ω),
# read off the normally ordered output spectrum 𝒩(ω) = S_out(ω) − 1/(2dx):
#   𝒩(ω) = K G D G† K + i K (G† − G) K/(2dx),
#   n_qq′(ω) = p_q′† 𝒩 p_q,  m_qq′(ω) = p_q† 𝒩 p̃_q′,  p_q = dx (e_q, 0),  p̃_q = dx (0, e_q*).
# The photocurrent I_j = Σ_{q∈j} :(α_q + c_q)†(α_q + c_q): then has
#   mean    ⟨I_j⟩ = I_coh,j + Σ_{q∈j} ∫ n_qq dω/2π,
#   noise   S_jk  = S_lin,jk + δ_jk I_sp,j + Σ_{q∈j, q′∈k} ∫ (|n_qq′|² + |m_qq′|²) dω/2π
# at Ω → 0 (Wick; the cross terms linear × quadratic are odd moments and vanish). The
# spectra vary on the scale κ ≫ Ω, so Ω → 0 is exact to O(Ω²/κ²) here too.
#
# Evaluation: Schur form M = Z T Z† (unitary Z: no ill-conditioned eigenvectors). With
# B = Z† K P, Y = (ω − T)^{-†} B (one triangular solve) and X = Z Y = G† K P,
#   P† 𝒩 P = X† D X + i (B† Y − Y† B)/(2dx).
# ---------------------------------------------------------------------------------------

_modekey(d::Detector, kq) = (first(d.idx), last(d.idx), round(Int, kq * length(d.idx) * d.dx / 2π))

"""
    DetectionBasis(s, dets)

The aperture Fourier modes used by any of the detectors `dets` (e.g. all detectors of a
probe-frequency scan), without duplicates. The spontaneous terms are computed once in
this basis; each detector is a subset of its modes.
"""
struct DetectionBasis
    keys::Vector{Tuple{Int,Int,Int}}
    k::Vector{Float64}
    E::Matrix{ComplexF64}                       # grid vectors e_q (N × Q), zero outside the aperture
    lookup::Dict{Tuple{Int,Int,Int},Int}
end
function DetectionBasis(s::Setup, dets)
    x, dx, N = s.grid.x, s.grid.dx, s.grid.N
    modes = Dict{Tuple{Int,Int,Int},Float64}()
    for d in dets, kq in d.k
        modes[_modekey(d, kq)] = kq
    end
    ks = sort(collect(keys(modes)))
    E = zeros(ComplexF64, N, length(ks))
    for (c, key) in enumerate(ks)
        idx = key[1]:key[2]
        E[idx, c] .= cis.(modes[key] .* x[idx]) ./ sqrt(length(idx) * dx)
    end
    DetectionBasis(ks, [modes[q] for q in ks], E, Dict(q => c for (c, q) in enumerate(ks)))
end
Base.length(b::DetectionBasis) = length(b.keys)
"""Columns of basis `b` that make up detector `d`."""
modes_of(b::DetectionBasis, d::Detector) = [b.lookup[_modekey(d, kq)] for kq in d.k]

"""
    SpontaneousNoise

Spontaneous-light moments in a [`DetectionBasis`](@ref), integrated over all
frequencies (trapezoid on a uniform grid):
- `flux[q]  = ∫ n_qq dω/2π` (photons/ps);
- `power[q, q′] = ∫ (|n_qq′|² + |m_qq′|²) dω/2π` (photons/ps);
- `flux2`, `power2`: the same with every other grid point (2dω), a quadrature check;
- `tail`: largest integrand of Σ_q n_qq at the two ends of the ω grid, relative to its
  maximum (truncation check).
"""
struct SpontaneousNoise
    basis::DetectionBasis
    ω::Vector{Float64}
    flux::Vector{Float64}
    power::Matrix{Float64}
    flux2::Vector{Float64}
    power2::Matrix{Float64}
    tail::Float64
end

"""
    spontaneous_noise(on, basis, ω; F=schur(Matrix{ComplexF64}(bogoliubov_matrix(on.s, on.ψ0))))

Normally ordered output moments of the fluctuations in the modes of `basis`, integrated
over the uniform frequency grid `ω` (ps⁻¹, pump frame), which must cover every
resonance of those modes with margin (≳ 20κ). `F` is the Schur factorisation of the
Bogoliubov matrix (O(N³), reuse it). Each frequency costs one triangular solve and one
product with 2Q right-hand sides, Q = `length(basis)`.
"""
function spontaneous_noise(on::OutputNoise, basis::DetectionBasis, ω::AbstractRange;
    F::Schur=schur(Matrix{ComplexF64}(bogoliubov_matrix(on.s, on.ψ0))))
    N, dx = on.s.grid.N, on.s.grid.dx
    Q = length(basis)
    P = zeros(ComplexF64, 2N, 2Q)
    P[1:N, 1:Q] .= dx .* basis.E
    P[N+1:2N, Q+1:2Q] .= dx .* conj.(basis.E)
    B = F.Z' * (on.K .* P)
    L = -Matrix(F.T')                                    # lower triangular; diagonal set per ω
    λc = conj.(diag(F.T))
    Y, X, DX = similar(B), similar(B), similar(B)
    flux, flux2 = zeros(Q), zeros(Q)
    power, power2 = zeros(Q, Q), zeros(Q, Q)
    trace = zeros(length(ω))
    for (i, w) in enumerate(ω)
        @inbounds for a in 1:2N
            L[a, a] = conj(w) - λc[a]
        end
        copyto!(Y, B)
        ldiv!(LowerTriangular(L), Y)                     # Y = (ω − T)^{-†} B
        mul!(X, F.Z, Y)                                  # X = G(ω)† K P
        DX .= on.D .* X
        Nq = view(X, :, 1:Q)' * DX .+ im .* (view(B, :, 1:Q)' * Y .- view(Y, :, 1:Q)' * B) ./ (2dx)
        n, m = view(Nq, :, 1:Q), view(Nq, :, Q+1:2Q)     # n[q′, q] = n_qq′(ω); m[q, q′] = m_qq′(ω)
        wt = (i == 1 || i == length(ω) ? 0.5 : 1.0) * step(ω) / 2π
        dflux = real.(diag(n))
        dpow = abs2.(n) .+ abs2.(m)
        flux .+= wt .* dflux
        power .+= wt .* dpow
        trace[i] = sum(dflux)
        if isodd(i)
            wt2 = (i == 1 || i == length(ω) - iseven(length(ω)) ? 1.0 : 2.0) * step(ω) / 2π
            flux2 .+= wt2 .* dflux
            power2 .+= wt2 .* dpow
        end
    end
    sym(A) = (A .+ A') ./ 2                              # |n_qq′| = |n_q′q|; m_q′q(ω) = m_qq′(−ω)
    tail = max(abs(trace[1]), abs(trace[end])) / maximum(abs, trace)
    SpontaneousNoise(basis, collect(ω), flux, sym(power), flux2, sym(power2), tail)
end

"""
    spontaneous_terms(sn, dets; output_scale=1)

Mean flux `I_sp` and photocurrent noise `S_sp` (incl. its own shot noise) of the
spontaneous light in detectors `dets`, at Ω → 0. They add to the coherent-probe terms of
[`self_homodyne`](@ref): `I = I_coh + I_sp`, `S = S_lin + S_sp`, so with probe power
scaled by p, `I = p I_coh + I_sp` and `S = p S_lin + S_sp`. `output_scale` rescales the
detected mirror rate κ_d (n, m ∝ κ_d exactly): I_sp ∝ c, the excess noise ∝ c².
"""
function spontaneous_terms(sn::SpontaneousNoise, dets::AbstractVector{Detector}; output_scale=1.0)
    cols = [modes_of(sn.basis, d) for d in dets]
    Isp = [output_scale * sum(sn.flux[c]) for c in cols]
    S = [output_scale^2 * sum(sn.power[a, b]) for a in cols, b in cols]
    Isp, S .+ Diagonal(Isp)
end
