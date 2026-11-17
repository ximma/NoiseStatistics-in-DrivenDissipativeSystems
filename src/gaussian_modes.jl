# Frequency-resolved Gaussian state of chosen output modes, and entanglement criteria.
# Derivation: docs/methods.md.
#
# A detected mode i is a spatial profile f_i (grid vector, Σ|f_i|²dx = 1) at the pump-frame
# frequency σ_i ω (σ_i = ±1): A_i = Σ_x dx f_i* A_out(x, σ_i ω). In the stationary state,
# fields at different frequencies are uncorrelated, so the only moments are
#   n_ij = ⟨A_i† A_j⟩ = p_j† 𝒩(σω) p_i                (σ_i = σ_j = σ),
#   m_ij = ⟨A_i A_j⟩  = p_i† 𝒩(σ_i ω) p̃_j            (σ_j = −σ_i),
# spectral densities from the normally ordered output spectrum 𝒩 of noise_spectra.jl
# (p = dx (f, 0), p̃ = dx (0, f*)). They are the covariances of one spectral–temporal mode:
# what a balanced homodyne detector with a local oscillator at ω_p + σ_i ω in profile f_i
# measures at Ω → 0. Profiles at the same σ must be orthonormal (independent modes,
# [A_i, A_j†] = δ_ij); `orthonormalize_modes` makes them so.

"""
    aperture_mode(s, (xa, xb), k)

Normalised profile e^{ikx} on the aperture [xa, xb] (zero outside), Σ|f|²dx = 1. `k` may
be complex (the spatial damping of an outgoing channel).
"""
function aperture_mode(s::Setup, xr, k)
    x, dx = s.grid.x, s.grid.dx
    f = zeros(ComplexF64, length(x))
    idx = findfirst(>=(xr[1]), x):findlast(<=(xr[2]), x)
    f[idx] .= exp.(im .* k .* x[idx])
    f ./ sqrt(sum(abs2, f) * dx)
end

"""
    orthonormalize_modes(modes, dx)

Symmetric (Löwdin) orthonormalisation of the profiles that share a frequency sign σ:
F → F O^{-1/2}, O = F†F dx. It changes each profile as little as possible. Modes at +ω
and −ω are independent whatever their profiles.
"""
function orthonormalize_modes(modes, dx)
    out = collect(modes)
    for σ in (1, -1)
        ix = findall(x -> x[2] == σ, modes)
        isempty(ix) && continue
        F = reduce(hcat, [modes[i][1] for i in ix])
        F2 = F * inv(sqrt(Hermitian(F' * F .* dx)))
        for (c, i) in enumerate(ix)
            out[i] = (F2[:, c], σ)
        end
    end
    out
end

"""
    SchurResolvent(M)

Schur form M = Z T Z† of the Bogoliubov matrix, kept to apply G(ω)† = Z (ω − T)^{-†} Z† by
triangular solves: cheaper than the Hessenberg solves of `OutputNoise` when many modes or
many frequencies are needed. Pass it as `schur=` to `mode_state` and `pixel_state`.
"""
struct SchurResolvent
    Z::Matrix{ComplexF64}
    L::Matrix{ComplexF64}                 # T†, lower triangular; its diagonal is shifted per call
    d::Vector{ComplexF64}                 # conj(diag T)
end
function SchurResolvent(M::AbstractMatrix)
    F = schur(Matrix{ComplexF64}(M))
    SchurResolvent(F.Z, Matrix(F.T'), conj.(diag(F.T)))
end

# 𝒩(z) projected on the columns of P (2N × c): X = G(z)† K P, then
# X† D X + i((KP)† X − X† KP)/(2dx) (normally ordered output spectrum, noise_spectra.jl).
function _projected_spectrum(on::OutputNoise, P, z, R)
    dx = on.s.grid.dx
    KP = on.K .* P
    if R === nothing
        X = -((on.hadj - conj(z) * I) \ KP)                  # (z − M)^{-†} K P
        return X' * (on.D .* X) .+ im .* (KP' * X .- X' * KP) ./ (2dx)
    end
    B = R.Z' * KP
    for i in eachindex(R.d)
        R.L[i, i] = R.d[i] - conj(z)
    end
    Y = ldiv!(LowerTriangular(R.L), -B)                      # (z − T)^{-†} B
    X = R.Z * Y
    X' * (on.D .* X) .+ im .* (B' * Y .- Y' * B) ./ (2dx)
end

# moments of the modes with profiles Fp[:, i] at σs[i] ω (profiles at equal σ orthonormal)
function _mode_moments(on::OutputNoise, Fp, σs, ω, output_scale, R)
    N, dx = on.s.grid.N, on.s.grid.dx
    ip, iq = findall(==(1), σs), findall(==(-1), σs)
    a, b = length(ip), length(iq)
    P = zeros(ComplexF64, 2N, a + b)                         # at +ω: p of the + modes, p̃ of the − modes
    P[1:N, 1:a] .= dx .* Fp[:, ip]
    P[N+1:2N, a+1:a+b] .= dx .* conj.(Fp[:, iq])
    N1 = _projected_spectrum(on, P, ω, R)
    Pm = zeros(ComplexF64, 2N, b)                            # at −ω: p of the − modes
    Pm[1:N, :] .= dx .* Fp[:, iq]
    N2 = b == 0 ? zeros(ComplexF64, 0, 0) : _projected_spectrum(on, Pm, -ω, R)
    Mm = length(σs)
    n = zeros(ComplexF64, Mm, Mm)
    m = zeros(ComplexF64, Mm, Mm)
    n[ip, ip] .= transpose(N1[1:a, 1:a])                      # n_ij = p_j† 𝒩 p_i
    n[iq, iq] .= transpose(N2)
    m[ip, iq] .= N1[1:a, a+1:a+b]                             # m_ij = p_i† 𝒩(ω) p̃_j
    m[iq, ip] .= transpose(m[ip, iq])
    output_scale .* (n + n') ./ 2, output_scale .* m
end

"""
    mode_state(on, modes, ω; output_scale=1, schur=nothing)

Normally ordered moments `(n, m)` (M × M) of the output modes `modes = [(f_i, σ_i), …]` at
pump-frame frequency ±ω (spectral densities, as for one spectral–temporal mode).
`output_scale` multiplies n and m, which models a smaller detected mirror rate or a
detection efficiency η (n, m ∝ η). The resolvent is applied by Hessenberg solves, or by
triangular solves with a precomputed `schur::SchurResolvent`.
"""
function mode_state(on::OutputNoise, modes, ω; output_scale=1.0, schur=nothing)
    N, dx = on.s.grid.N, on.s.grid.dx
    for σ in (1, -1)                         # independent modes: orthonormal at equal σ
        F = reduce(hcat, [f for (f, σi) in modes if σi == σ]; init=zeros(ComplexF64, N, 0))
        maximum(abs, F' * F .* dx - I; init=0.0) < 1e-8 ||
            throw(ArgumentError("profiles at the same frequency must be orthonormal; use orthonormalize_modes"))
    end
    Fp = reduce(hcat, [f for (f, _) in modes])
    _mode_moments(on, Fp, [σ for (_, σ) in modes], ω, output_scale, schur)
end

"""
    pixel_state(on, idx, ω; output_scale=1, schur=nothing)

Moments `(n, m)` of the output field in the pixel basis: modes 1:P are the grid points
`idx` at +ω, modes P+1:2P the same points at −ω (profile 1/√dx on one cell). Every
detectable mode in that region is a combination of these (`transform_modes`), so the
entanglement of this state bounds that of any choice of LO profiles there.
"""
function pixel_state(on::OutputNoise, idx, ω; output_scale=1.0, schur=nothing)
    N, dx = on.s.grid.N, on.s.grid.dx
    Pn = length(idx)
    Fp = zeros(ComplexF64, N, 2Pn)
    for (c, j) in enumerate(idx)
        Fp[j, c] = Fp[j, Pn+c] = 1 / sqrt(dx)
    end
    _mode_moments(on, Fp, [fill(1, Pn); fill(-1, Pn)], ω, output_scale, schur)
end

"""
    transform_modes(n, m, c)

Moments of the new modes B_k whose profiles are g_k = Σ_i c_ik f_i (columns of `c`, in the
basis of the modes of `(n, m)`): n′ = cᵀ n c̄, m′ = c† m c̄. A column must not mix modes of
opposite σ (such a mode is not stationary). New modes are independent when the columns
of `c` are orthonormal within each σ.
"""
transform_modes(n, m, c) = (transpose(c) * n * conj(c), c' * m * conj(c))

"""
    principal_modes(n, groups; rtol=1e-9)

Eigenmodes of the occupations within each group of modes (a group: one region at one σ),
keeping those with occupation > rtol × the largest of the group. Returns the coefficient
matrix `c` (for `transform_modes`), whose new modes are diagonal in n within each group,
and the new groups. Modes with n ≈ 0 are near vacuum and carry no correlation
(|m_ij|² ≤ n_i(n_j + 1)), so discarding them leaves the entanglement unchanged.
"""
function principal_modes(n, groups; rtol=1e-9)
    cols = Vector{Vector{ComplexF64}}()
    newgroups = Vector{Vector{Int}}()
    for g in groups
        e = eigen(Hermitian(Matrix(n[g, g])))
        keep = findall(>(rtol * max(maximum(e.values), eps())), e.values)
        keep = keep[sortperm(e.values[keep]; rev=true)]
        ng = Int[]
        for k in keep
            v = zeros(ComplexF64, size(n, 1))
            v[g] .= conj.(e.vectors[:, k])                   # n′ = cᵀ n c̄ = V† n V
            push!(cols, v)
            push!(ng, length(cols))
        end
        push!(newgroups, ng)
    end
    reduce(hcat, cols; init=zeros(ComplexF64, size(n, 1), 0)), newgroups
end

# Löwdin-orthonormal k columns from a real parameter vector
function _unpack_modes(θ, r, k)
    Zc = reshape(complex.(θ[1:r*k], θ[r*k+1:2r*k]), r, k)
    Zc * inv(sqrt(Hermitian(Zc' * Zc)))
end

"""
    optimise_modes(n, m, groups, A; k=1, inits=[], restarts=4, rng=Random.default_rng())

Chooses k modes in each group (orthonormal combinations of its modes, so one frequency
sign and one region per group) to maximise the logarithmic negativity between the groups
`A` (indices into `groups`) and the others. The objective is −log₂ 2ν̃₁ − Σ_{i>1} min(0,
log₂ 2ν̃_i) over the sorted symplectic eigenvalues ν̃ of the partial transpose: E_N when
ν̃₁ < ½, and still informative (no flat region) when not. It is maximised by BFGS from each initial
guess in `inits` (vectors of coefficient matrices, one r_g × k per group) and from
`restarts` random ones. Returns `(EN, c, groups′)`: E_N of the selected modes, their
coefficient matrix for `transform_modes`, and their grouping.
"""
function optimise_modes(n, m, groups, A; k=1, inits=[], restarts=4, rng=Random.default_rng(), iters=2000)
    Mm = size(n, 1)
    rs = [min(k, length(g)) for g in groups]
    nr = [length(g) for g in groups]
    off = cumsum([0; 2 .* nr .* rs])
    sel = [sum(rs[1:j-1]) .+ (1:rs[j]) for j in eachindex(groups)]
    Asel = reduce(vcat, sel[A])
    function coeffs(θ)
        c = zeros(ComplexF64, Mm, sum(rs))
        for (j, g) in enumerate(groups)
            c[g, sel[j]] .= _unpack_modes(view(θ, off[j]+1:off[j+1]), nr[j], rs[j])
        end
        c
    end
    function objective(θ)
        n2, m2 = transform_modes(n, m, coeffs(θ))
        σ = quadrature_covariance(n2, m2)
        T = ones(size(σ, 1)); T[2 .* Asel] .= -1
        ν = sort(symplectic_eigenvalues(T .* σ .* T'))
        log2(2ν[1]) + sum(x -> min(0.0, log2(2x)), ν[2:end]; init=0.0)   # −E_N when ν₁ < ½; minimise
    end
    starts = Vector{Vector{Float64}}()
    for c0 in [collect(inits); _correlation_inits(n, m, groups, A, rs)]
        push!(starts, reduce(vcat, [[vec(real(c0[j][:, 1:rs[j]])); vec(imag(c0[j][:, 1:rs[j]]))] for j in eachindex(groups)]))
    end
    for _ in 1:restarts
        push!(starts, randn(rng, off[end]))
    end
    best = (Inf, Float64[])
    for θ0 in starts
        θ, f = _bfgs(objective, θ0; iters)
        f < best[1] && (best = (f, θ))
    end
    c = coeffs(best[2])
    n2, m2 = transform_modes(n, m, c)
    log_negativity(quadrature_covariance(n2, m2), Asel), c, sel
end

# Starting points from the pair correlations across the cut: each group g in A is paired
# with the group h outside A where ‖m_gh‖ is largest, and the top-k singular vectors of
# m_gh (plain, and whitened by the occupations: the pairs with the largest Cauchy–Schwarz
# ratio |m_ab|²/(n_a n_b)) give its modes and h's. Other groups start from their most
# occupied modes. Random starts tend to stall on near-vacuum modes (ν̃ ≈ ½, flat objective).
function _correlation_inits(n, m, groups, A, rs)
    ng = length(groups)
    isqrt(H) = (e = eigen(Hermitian(Matrix(H))); λ = max.(e.values, 1e-6 * max(maximum(e.values), eps()));
                e.vectors * Diagonal(1 ./ sqrt.(λ)) * e.vectors')
    top(g, k) = (e = eigen(Hermitian(Matrix(n[g, g]))); conj.(e.vectors[:, sortperm(e.values; rev=true)[1:k]]))
    out = []
    for whiten in (false, true)
        c0 = Vector{Any}(nothing, ng)
        for a in A
            hs = setdiff(1:ng, A)
            isempty(hs) && continue
            h = hs[argmax([norm(m[groups[a], groups[b]]) for b in hs])]
            c0[h] === nothing || continue
            Wa = whiten ? isqrt(transpose(n[groups[a], groups[a]])) : I
            Wh = whiten ? isqrt(n[groups[h], groups[h]]) : I
            F = svd(Matrix(Wa * m[groups[a], groups[h]] * Wh))
            k = min(rs[a], rs[h])
            k == rs[a] == rs[h] || continue
            c0[a] = Wa * F.U[:, 1:k]                                     # m′_ab = a† m b̄
            c0[h] = conj.(Wh * F.V[:, 1:k])
        end
        for j in 1:ng
            c0[j] === nothing && (c0[j] = top(groups[j], rs[j]))
        end
        push!(out, c0)
    end
    out
end

# quasi-Newton minimisation with central-difference gradients and Armijo backtracking
function _bfgs(f, x0; iters=2000, h=1e-6, gtol=1e-6)
    grad(x) = [(f(x .+ h .* (1:length(x) .== i)) - f(x .- h .* (1:length(x) .== i))) / 2h for i in eachindex(x)]
    x = copy(x0); fx = f(x); g = grad(x); H = Matrix(1.0I, length(x), length(x))
    for _ in 1:iters
        d = -H * g
        dot(d, g) < 0 || (H = Matrix(1.0I, length(x), length(x)); d = -g)
        t = 1.0; xn = x; fn = fx
        while true
            xn = x .+ t .* d; fn = f(xn)
            fn <= fx + 1e-4 * t * dot(g, d) && break
            t /= 2
            t < 1e-10 && return x, fx
        end
        gn = grad(xn)
        sv, y = xn - x, gn - g
        if dot(sv, y) > 1e-14
            ρ = 1 / dot(y, sv)
            H = (I - ρ * sv * y') * H * (I - ρ * y * sv') + ρ * sv * sv'
        end
        x, fx, g = xn, fn, gn
        norm(g) < gtol && break
    end
    x, fx
end

"""
    quadrature_covariance(n, m)

Real symmetric covariance σ (2M × 2M, order x₁, p₁, x₂, p₂, …) of the quadratures
x = (A + A†)/√2, p = (A − A†)/(i√2), symmetrically ordered: the vacuum is I/2.
"""
function quadrature_covariance(n, m)
    Mm = size(n, 1)
    C = transpose(n) + I / 2                                 # C_ij = ½⟨{A_i, A_j†}⟩
    σ = zeros(2Mm, 2Mm)
    for i in 1:Mm, j in 1:Mm
        σ[2i-1, 2j-1] = real(C[i, j]) + real(m[i, j])
        σ[2i, 2j] = real(C[i, j]) - real(m[i, j])
        σ[2i-1, 2j] = imag(m[i, j]) - imag(C[i, j])
        σ[2i, 2j-1] = imag(m[j, i]) - imag(C[j, i])
    end
    (σ + σ') / 2
end

"""Symplectic eigenvalues of a quadrature covariance (≥ 1/2 for a physical state)."""
function symplectic_eigenvalues(σ)
    Mm = size(σ, 1) ÷ 2
    Ω = kron(Matrix(I, Mm, Mm), [0 1; -1 0])
    ν = sort(abs.(eigvals(im .* Ω * σ)))
    ν[1:2:end]                                               # each appears twice
end

"""
    log_negativity(σ, A)

Logarithmic negativity E_N = Σ max(0, −log₂ 2ν̃) of the bipartition (modes `A` | the rest),
from the symplectic eigenvalues ν̃ of the partially transposed covariance (p → −p on `A`).
E_N > 0 is a sufficient condition for entanglement (necessary and sufficient when one side
is a single mode).
"""
function log_negativity(σ, A)
    T = ones(size(σ, 1))
    for a in A
        T[2a] = -1
    end
    sum(max(0.0, -log2(2ν)) for ν in symplectic_eigenvalues(T .* σ .* T'))
end

"""
    pair_ratio(n, m, i, j)

|m_ij|²/(n_ii n_jj): the two-mode (Cauchy–Schwarz) non-separability criterion. A value > 1
means modes i and j are entangled; the other correlations of the pair are zero by
stationarity when σ_i = −σ_j.
"""
pair_ratio(n, m, i, j) = abs2(m[i, j]) / real(n[i, i] * n[j, j])

"""
    max_pair_ratio(n, m, ia, ib; rtol=1e-6)

Largest pair ratio |m_AB|²/(n_A n_B) over one mode A = Σ_{i∈ia} a_i (mode i) and one mode
B = Σ_{j∈ib} b_j (mode j), where the modes `ia` share one σ, the modes `ib` the other, and
each set is orthonormal. With x = ā, y = b̄: n_A = x†n_aa x, n_B = y†n_bb y, m_AB = xᵀ m_ab y.
Whitening x = n_aa^{-1/2} p, y = n_bb^{-1/2} q turns the ratio into |pᵀ C q|²/(|p|²|q|²) with
C = (n_aa^{-1/2})ᵀ m_ab n_bb^{-1/2}, so the maximum is σ₁(C)², at the leading singular pair.
Directions with occupation below `rtol`·max are dropped. Returns `(R, a, b)`: a and b are
unit coefficient vectors (columns for `transform_modes` on ia and ib).
"""
function max_pair_ratio(n, m, ia, ib; rtol=1e-6)
    function whiten(N)
        e = eigen(Hermitian(Matrix(N)))
        keep = e.values .> rtol * maximum(e.values)
        e.vectors[:, keep] * Diagonal(1 ./ sqrt.(e.values[keep]))
    end
    P, Q = whiten(n[ia, ia]), whiten(n[ib, ib])
    F = svd(transpose(P) * m[ia, ib] * Q)
    x = P * conj.(F.U[:, 1]); y = Q * F.V[:, 1]
    F.S[1]^2, conj.(x) ./ norm(x), conj.(y) ./ norm(y)
end
