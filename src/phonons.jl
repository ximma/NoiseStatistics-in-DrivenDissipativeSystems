# Lattice-phonon bath of the excitonic fraction (Frérot et al., PRX 13, 041058 (2023)).
#
# The condensate's excitons couple to longitudinal acoustic phonons through the deformation
# potential, H = ħ ∫dx ψ†ψ ξ(x, t), with ξ a Hermitian bath field. Linearised around ψ₀ and
# integrated over the bath (Markov in time apart from the frequency dependence of the rate,
# Kramers–Kronig shift neglected), it adds to the Bogoliubov problem, in Nambu form
# Φ = (δψ, δψ†):
#
#   noise:   Ξ_ph = −i ξ (ψ₀, −ψ₀*),   ∫dτ e^{iωτ}⟨ξ(x,τ)ξ(x′,0)⟩ = δ(x − x′) R(x, |ω|)/|ψ₀|² · {n_T + 1 (ω > 0), n_T (ω < 0)}
#   damping: Σ(ω) = −(i/2) sgn(ω) R(x, |ω|) [[1, e^{2iφ₀}], [−e^{−2iφ₀}, −1]]   (local, per site)
#
# R(x, ω) is the local phonon scattering rate of the fluid, n_T the Bose occupation of
# phonons at ħ|ω|. The frequency ω is the pump-frame (lab-frame) energy exchanged with the
# lattice, so a negative-norm mode at ω > 0 (supersonic flow, ergoregion) is anti-damped:
# creating it emits a phonon (the Landau criterion with respect to the lattice at rest).
# The damping is independent of T; the noise carries T. In symmetric ordering the noise is
#
#   D_ph(ω) = R (n_T(|ω|) + ½)/dx · [[1, −e^{2iφ₀}], [−e^{−2iφ₀}, 1]]  (per site)
#
# which with Σ keeps the output commutators (fluctuation–dissipation; tested).
#
# Rate (Frérot App. A9): γ_xp(ω) ∝ ω exp(−2ω²/(v_s q_cut)²), maximal at ω_m = v_s q_cut/2
# (ħω_m ≈ 1.03 meV for v_s = 4.7 nm/ps, q_cut = 0.9·2π/8.5 nm, their fitted GaAs well).
# Their scattering rate between the condensate's excitons and the phonon vacuum is
# ħγ_xp^M n_x ≈ 2.5 µeV at a blueshift of 0.19 meV. We scale it with the local blueshift
# g|ψ₀(x)|²: R(x, ω) = strength · (2.5 µeV/0.19 meV) · g|ψ₀(x)|² · shape(ω), shape(ω_m) = 1.
# `strength` absorbs the material ratio γ_xp/g_x and sample constants (1: same ratio of phonon
# rate to blueshift as in Frérot et al.; the excitonic fraction cancels at fixed blueshift).

const KB_MEV = 0.08617333          # Boltzmann constant [meV/K]

"""
    PhononBath(; T=4.0, strength=1.0, ħωm=1.03, ratio=2.5e-3/0.19, ħ=0.6582)

Lattice-phonon bath of temperature `T` [K]. Local rate R(x, ω) = strength · ratio ·
g|ψ₀(x)|² · shape(ω), with shape(ω) = (|ω|/ω_m) exp((1 − ω²/ω_m²)/2) (maximal, = 1, at
ħω_m [meV]). Defaults follow Frérot et al., PRX 13, 041058 (2023).
"""
struct PhononBath
    T::Float64
    strength::Float64
    ωm::Float64
    ratio::Float64
    ħ::Float64
end
PhononBath(; T=4.0, strength=1.0, ħωm=1.03, ratio=2.5e-3 / 0.19, ħ=0.6582) = PhononBath(T, strength, ħωm / ħ, ratio, ħ)

"""Frequency dependence of the exciton–phonon rate, normalised to 1 at ω_m."""
phonon_shape(b::PhononBath, ω) = (x = abs(ω) / b.ωm; x * exp((1 - x^2) / 2))

"""Bose occupation n_T(|ω|) of lattice phonons at ħ|ω|."""
phonon_occupation(b::PhononBath, ω) = b.T <= 0 ? 0.0 : 1 / expm1(b.ħ * abs(ω) / (KB_MEV * b.T))

"""Local phonon scattering rates R(x, ω) [ps⁻¹] of the fluid ψ₀ (length N)."""
phonon_rates(b::PhononBath, s::Setup, ψ0::AbstractVector, ω) =
    (b.strength * b.ratio * phonon_shape(b, ω) * s.pol.g) .* abs2.(ψ0)

_phase2(ψ0) = [abs(z) > 0 ? (z / abs(z))^2 : one(ComplexF64) for z in ψ0]

"""
    phonon_self_energy(b, s, ψ0, ω)

Damping (2N × 2N, local 2 × 2 blocks) added to the Bogoliubov matrix by the phonon bath at
frequency ω: the poles of G(ω) = (ω − M − Σ(ω))⁻¹. Independent of the temperature.
"""
function phonon_self_energy(b::PhononBath, s::Setup, ψ0::AbstractVector, ω)
    N = s.grid.N
    R = phonon_rates(b, s, ψ0, ω); e2 = _phase2(ψ0); sg = sign(real(ω))
    Σ = zeros(ComplexF64, 2N, 2N)
    for i in 1:N
        c = -0.5im * sg * R[i]
        Σ[i, i] = c;            Σ[i, N+i] = c * e2[i]
        Σ[N+i, i] = -c * conj(e2[i]);  Σ[N+i, N+i] = -c
    end
    Σ
end

# X ↦ D_ph X (symmetric-order phonon noise, local 2 × 2 blocks)
function _apply_phonon_noise(R, e2, nT, dx, X)
    N = length(R)
    c = R .* (nT + 0.5) ./ dx
    top = @view X[1:N, :]; bot = @view X[N+1:2N, :]
    vcat(c .* (top .- e2 .* bot), c .* (bot .- conj.(e2) .* top))
end

"""
    PhononNoise(on, b; M=bogoliubov_matrix(on.s, on.ψ0))

Output-noise solver with the phonon bath `b` on top of the loss ports of `on`. Each
frequency costs one dense LU of (ω − M − Σ(ω))† (the damping depends on ω).
"""
struct PhononNoise{O}
    on::O
    bath::PhononBath
    Madj::Matrix{ComplexF64}
end
PhononNoise(on::OutputNoise, b::PhononBath; M=bogoliubov_matrix(on.s, on.ψ0)) = PhononNoise(on, b, Matrix{ComplexF64}(M'))

# projected normally ordered output spectra P†𝒩(z)P for each temperature in Ts
function _projected_spectrum_phonon(pn::PhononNoise, P, z, Ts)
    on, b = pn.on, pn.bath; s = on.s; N, dx = s.grid.N, s.grid.dx
    A = -pn.Madj .- phonon_self_energy(b, s, on.ψ0, z)'
    for i in 1:2N
        A[i, i] += conj(z)
    end
    KP = on.K .* P
    X = lu!(A) \ KP                                            # (z − M − Σ)^{-†} K P
    base = X' * (on.D .* X) .+ im .* (KP' * X .- X' * KP) ./ (2dx)
    R = phonon_rates(b, s, on.ψ0, z); e2 = _phase2(on.ψ0)
    map(Ts) do T
        nT = phonon_occupation(PhononBath(T, b.strength, b.ωm, b.ratio, b.ħ), z)
        base .+ X' * _apply_phonon_noise(R, e2, nT, dx, X)
    end
end

"""
    phonon_mode_state(pn, modes, ω; Ts=[pn.bath.T], output_scale=1.0)

As [`mode_state`](@ref), with the phonon bath: normally ordered moments `(n, m)` of the
output modes `modes = [(f_i, σ_i), …]` at ±ω, one pair per temperature in `Ts` (the damping,
and hence the LU, is shared).
"""
function phonon_mode_state(pn::PhononNoise, modes, ω; Ts=[pn.bath.T], output_scale=1.0)
    on = pn.on; N, dx = on.s.grid.N, on.s.grid.dx
    for σ in (1, -1)
        F = reduce(hcat, [f for (f, σi) in modes if σi == σ]; init=zeros(ComplexF64, N, 0))
        maximum(abs, F' * F .* dx - I; init=0.0) < 1e-8 ||
            throw(ArgumentError("profiles at the same frequency must be orthonormal; use orthonormalize_modes"))
    end
    Fp = reduce(hcat, [f for (f, _) in modes]); σs = [σ for (_, σ) in modes]
    ip, iq = findall(==(1), σs), findall(==(-1), σs); a, b = length(ip), length(iq)
    P = zeros(ComplexF64, 2N, a + b)
    P[1:N, 1:a] .= dx .* Fp[:, ip]; P[N+1:2N, a+1:a+b] .= dx .* conj.(Fp[:, iq])
    N1s = _projected_spectrum_phonon(pn, P, ω, Ts)
    Pm = zeros(ComplexF64, 2N, b); Pm[1:N, :] .= dx .* Fp[:, iq]
    N2s = b == 0 ? [zeros(ComplexF64, 0, 0) for _ in Ts] : _projected_spectrum_phonon(pn, Pm, -ω, Ts)
    map(zip(N1s, N2s)) do (N1, N2)
        Mm = length(σs); n = zeros(ComplexF64, Mm, Mm); m = zeros(ComplexF64, Mm, Mm)
        n[ip, ip] .= transpose(N1[1:a, 1:a]); n[iq, iq] .= transpose(N2)
        m[ip, iq] .= N1[1:a, a+1:a+b]; m[iq, ip] .= transpose(m[ip, iq])
        (output_scale .* (n + n') ./ 2, output_scale .* m)
    end
end

"""
    phonon_covariance(s, ψ0, b; noise=NoiseModel(s), M=bogoliubov_matrix(s, ψ0), ωs=nothing, pad=30, dω=κ/8)

Equal-time symmetric covariance C = ⟨ΦΦ†⟩_sym = ∫ G(ω)(D + D_ph(ω))G(ω)† dω/2π with the phonon
bath `b` (the Lyapunov equation does not apply: Σ and D_ph depend on ω). The default grid
covers the real parts of the Bogoliubov spectrum padded by `pad`·κ, with step κ/8 (κ = γ/2).
Pass the result to [`Fluctuations`](@ref) for normally ordered moments and g₂.
"""
function phonon_covariance(s::Setup, ψ0::AbstractVector, b::PhononBath; noise::NoiseModel=NoiseModel(s),
    M=bogoliubov_matrix(s, ψ0), ωs=nothing, pad=30.0, dωfrac=1 / 8)
    N, dx = s.grid.N, s.grid.dx
    κ0 = s.pol.γ / 2
    if ωs === nothing
        lo, hi = extrema(real.(eigvals(Matrix(M))))
        ωs = range(lo - pad * κ0, hi + pad * κ0; step=dωfrac * κ0)
    end
    D = noise_diagonal(s, noise); e2 = _phase2(ψ0)
    C = zeros(ComplexF64, 2N, 2N)
    Mc = Matrix{ComplexF64}(M)
    for ω in ωs
        G = inv(ω * I - Mc - phonon_self_energy(b, s, ψ0, ω))
        Gt = Matrix(G')
        DG = D .* Gt .+ _apply_phonon_noise(phonon_rates(b, s, ψ0, ω), e2, phonon_occupation(b, ω), dx, Gt)
        mul!(C, G, DG, step(ωs) / 2π, 1.0)
    end
    (C + C') / 2
end
