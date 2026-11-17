# Mode-resolved analysis of stimulated scattering. In each homogeneous region, the
# demodulated response of a probe run is fitted to the local Bogoliubov modes (LDA), with
# complex wavevectors that include the damping. This gives complex mode amplitudes,
# Bogoliubov norm fluxes, and scattering coefficients normalised to the incident mode.
#
# In a region with background ψ₀ = √n e^{ik₀x}, a mode of real frequency ω is
#   δψ = a [U e^{i(kx − ωt)} + V* e^{−i(k* x − ωt)} e^{2ik₀x}]
# Demodulating at +ω gives a U e^{ikx} (lab wavevector k); demodulating at −ω gives
# a* V* e^{i(2k₀ − k*)x} (mirror wavevector 2k₀ − k). (U, V) solve the 2×2 lab-frame
# Bogoliubov problem at k. The norm flux of a mode is J = |a|²(|U|² − |V|²) v_g.

"""
    bogoliubov_uv(f, k)

Lab-frame 2×2 Bogoliubov eigenproblem of the homogeneous fluid `f` at real wavevector `k`
(u at k, v at k − 2k₀). Returns `((ω₊, U₊, V₊), (ω₋, U₋, V₋))`, with symplectic norm
|U|² − |V|² = +1 for the positive-norm branch and −1 for the negative-norm branch
(Wigner shift included through `f.δ`).
"""
function bogoliubov_uv(f::LocalFluid, k)
    ħ, m, g = f.pol.ħ, f.pol.m, f.pol.g
    gn = g * f.n
    if is_parabolic(f.pol)
        δ0eff = f.δ + m * f.v^2 / (2ħ)             # = δ₀ (+ Wigner shift), the lab-frame detuning
        a = ħ * k^2 / (2m) - δ0eff + 2gn
        b = ħ * (2f.k0 - k)^2 / (2m) - δ0eff + 2gn
    else
        δ0eff = f.δ + kinetic(f.pol, f.k0)
        a = kinetic(f.pol, k) - δ0eff + 2gn
        b = kinetic(f.pol, 2f.k0 - k) - δ0eff + 2gn
    end
    M = [a gn; -gn -b]
    F = eigen(complex.(M))
    out = map(1:2) do j
        U, V = F.vectors[1, j], F.vectors[2, j]
        η = abs2(U) - abs2(V)
        s = sqrt(abs(η))
        (real(F.values[j]), U / s, V / s, sign(η))
    end
    p = findfirst(o -> o[4] > 0, out)
    p === nothing && return nothing                # complex (unstable/gapped) wavevector
    q = 3 - p
    ((out[p][1], out[p][2], out[p][3]), (out[q][1], out[q][2], out[q][3]))
end

"""
    ScatteringMode(name, region, branch, k, kc, U, V, vg)

One Bogoliubov channel at fixed ω:
- `region`: `:up` or `:down`; `branch`: ±1;
- `k`: real lab wavevector; `kc`: complex wavevector including damping, k + iκ/v_g;
- `U`, `V`: mode components; `vg`: group velocity.
"""
struct ScatteringMode
    name::Symbol
    region::Symbol
    branch::Int
    k::Float64
    kc::ComplexF64
    U::ComplexF64
    V::ComplexF64
    vg::Float64
end
norm_flux(m::ScatteringMode, a) = abs2(a) * (abs2(m.U) - abs2(m.V)) * m.vg

"""
    complex_root(f, ω, b, k; κ)

Complex wavevector k_c with `ω + iκ = Ω_b(k_c)`, where Ω_b is the analytic continuation of
the LDA branch `b` (amplitude damping κ). Newton iteration from the real root `k`. At
first order k_c ≈ k + iκ/v_g, which fails when v_g → 0 (band edges).
"""
function complex_root(f::LocalFluid, ω, b::Integer, k; κ, iters=60)
    ħ, m, g = f.pol.ħ, f.pol.m, f.pol.g
    gn = g * f.n
    pol, k0 = f.pol, f.k0
    # ε(q) = [e(k₀+q) + e(k₀−q)]/2 − e(k₀), Doppler D(q) = [e(k₀+q) − e(k₀−q)]/2 (vq, ħq²/2m for the parabola)
    parts(kc) = is_parabolic(pol) ? (ħ * (kc - k0)^2 / (2m), ħ * (kc - k0) / m, f.v * (kc - k0), f.v) :
        ((kinetic(pol, kc) + kinetic(pol, 2k0 - kc)) / 2 - kinetic(pol, k0), (kinetic_d1(pol, kc) - kinetic_d1(pol, 2k0 - kc)) / 2,
         (kinetic(pol, kc) - kinetic(pol, 2k0 - kc)) / 2, (kinetic_d1(pol, kc) + kinetic_d1(pol, 2k0 - kc)) / 2)
    ε0 = parts(k)[1]
    s0 = sign(real(sqrt(complex((ε0 + gn - f.δ) * (ε0 + 3gn - f.δ)))))
    function Ω(kc)
        ε, dε, D, dD = parts(kc)
        P = (ε + gn - f.δ) * (ε + 3gn - f.δ)
        dP = dε * (2ε + 4gn - 2f.δ)
        r = sqrt(complex(P)) * (s0 == 0 ? 1 : s0)
        D + b * r, dD + b * dP / (2r)
    end
    vg = group_velocity(f, k, b)
    kc = complex(k) + (abs(vg) > 1e-3 ? im * κ / vg : 0.0im)
    for _ in 1:iters
        F, dF = Ω(kc)
        δ = (F - ω - im * κ) / dF
        kc -= δ
        abs(δ) < 1e-13 && break
    end
    kc
end

"""
    channel_modes(f, ω, region; κ, kmax=3)

All propagating LDA modes of fluid `f` at lab frequency ω (both branches). Each is
labelled by branch and direction of propagation (sign of v_g). `κ` is the amplitude
damping; the complex wavevector solves ω + iκ = Ω_b(k_c) exactly ([`complex_root`](@ref)).
"""
function channel_modes(f::LocalFluid, ω, region::Symbol; κ, kmax=3.0)
    modes = ScatteringMode[]
    for b in (1, -1), k in lda_roots(f, ω, b; krange=(-kmax, kmax))
        vg = group_velocity(f, k, b)
        uv = bogoliubov_uv(f, k)
        uv === nothing && continue
        _, U, V = b == 1 ? uv[1] : uv[2]
        name = Symbol(region, b == 1 ? "+" : "-", vg > 0 ? "→" : "←")
        push!(modes, ScatteringMode(name, region, b, k, complex_root(f, ω, b, k; κ), U, V, vg))
    end
    modes
end

"""
    fit_modes(x, ψp, ψm, ψ0, modes; xref, W=1)

Least-squares fit of a probe's demodulated response in one region. Inputs:
- `ψp`: demodulation at +ω, so ψp/W = u(x);
- `ψm`: demodulation at −ω, so conj(ψm)/W = v(x);
- `ψ0`: the background on the same points, which fixes the phase of v.

`W` is the sum of the time-window weights. The model is

    u(x) = Σ_j a_j U_j e^{ik_j(x−xref)},   v(x) e^{2iφ₀(x)} = Σ_j a_j V_j e^{ik_j(x−xref)}

with complex k_j (damping included). Returns `(a, relative residual)`, where a_j is the
amplitude of mode j at `xref`.
"""
function fit_modes(x, ψp, ψm, ψ0, modes::Vector{ScatteringMode}; xref, W=1.0)
    isempty(modes) && return (ComplexF64[], NaN)
    E = [exp(im * m.kc * (xi - xref)) for xi in x, m in modes]
    A = vcat(E .* transpose([m.U for m in modes]), E .* transpose([m.V for m in modes]))
    ph2 = (ψ0 ./ abs.(ψ0)) .^ 2
    y = vcat(ψp ./ W, conj.(ψm) ./ W .* ph2)
    a = A \ y
    a, norm(A * a - y) / norm(y)
end
