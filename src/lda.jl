# Local-density approximation (LDA) of the Bogoliubov dispersion in each homogeneous region.

"""
    LocalFluid(n, v, δ, k0, pol)

Homogeneous region of the flow: density `n` (µm⁻¹), velocity `v` (µm/ps), effective
detuning `δ` (ps⁻¹), carrier wavevector `k0 = m v/ħ` (µm⁻¹). A frozen exciton reservoir
(`Setup.Vres`) lowers the effective detuning: δ = δ(v) − g_r n_r.
"""
struct LocalFluid{T}
    n::T
    v::T
    δ::T
    k0::T
    pol::Polariton{T}
end

"""
    local_fluid(s, ψ, x_beg, x_end; method=:mean)

Average the mean field over `x_beg ≤ x ≤ x_end`. `method=:legacy` reproduces the
estimator of arXiv:2511.12339 exactly: forward phase differences on `ψ[2:N]`, density from the mean of
`√(ħgn/m)`, and indices `floor(x/dx)+1 : floor(x/dx)` (1-based).
"""
function local_fluid(s::Setup{T}, ψ, x_beg, x_end; method::Symbol=:mean) where {T}
    pol, dx = s.pol, s.grid.dx
    if method === :legacy
        θ = _unwrap(angle.(ψ[2:end]))
        vx = pol.ħ / pol.m * diff(θ) / dx
        cx = sqrt.(pol.ħ * pol.g * abs2.(ψ) / pol.m)
        r = (floor(Int, x_beg / dx)+1):floor(Int, x_end / dx)
        n = pol.m * mean(cx[r])^2 / (pol.ħ * pol.g)
        v = mean(vx[r])
    else
        r = findall(x -> x_beg <= x <= x_end, s.grid.x)
        n = mean(density(ψ)[r])
        if !is_parabolic(pol)                       # lower polariton: average the wavevector
            k0 = mean(wavevector(s, ψ)[r])
            return LocalFluid{T}(n, kinetic_d1(pol, k0), detuning_at_k(s, k0) - mean(s.Vres[r]), k0, pol)
        end
        v = mean(velocity(s, ψ)[r])
    end
    rr = method === :legacy ? ((floor(Int, x_beg / dx)+1):floor(Int, x_end / dx)) : r
    LocalFluid{T}(n, v, local_detuning(s, v) - mean(s.Vres[rr]), pol.m * v / pol.ħ, pol)
end

function _unwrap(θ)
    y = collect(θ)
    for i in 2:length(y)
        d = y[i] - y[i-1]
        abs(d) > π && (y[i] -= floor((d + π) / 2π) * 2π)
    end
    y
end

"""
    lda_omega(f, k, branch)

Laboratory-frame Bogoliubov frequency (pump-rotating frame, ps⁻¹) at wavevector `k`:

    ω±(k) = v q ± Re √[(ħq²/2m + gn − δ)(ħq²/2m + 3gn − δ)],   q = k − k0

where `branch = +1` is the positive-norm branch and `-1` the negative-norm one
(Eq. (2) of arXiv:2512.17807, Eq. (5) of arXiv:2511.12339). Where the square root is
imaginary (dynamically unstable or gapped wavevectors), only the real part `v q` remains.

For a non-parabolic dispersion e(k) (`pol.ΩR` finite), v q → [e(k) − e(2k₀ − k)]/2 and
ħq²/2m → [e(k) + e(2k₀ − k)]/2 − e(k₀).
"""
function lda_omega(f::LocalFluid, k, branch::Integer)
    if !is_parabolic(f.pol)                         # e(k₀ ± q): Doppler and curvature from the full dispersion
        e = kinetic(f.pol, k); e2 = kinetic(f.pol, 2f.k0 - k)
        ε = (e + e2) / 2 - kinetic(f.pol, f.k0)
        gn = f.pol.g * f.n
        return (e - e2) / 2 + branch * real(sqrt(complex((ε + gn - f.δ) * (ε + 3gn - f.δ))))
    end
    q = k - f.k0
    ε = f.pol.ħ * q^2 / (2f.pol.m)
    gn = f.pol.g * f.n
    f.v * q + branch * real(sqrt(complex((ε + gn - f.δ) * (ε + 3gn - f.δ))))
end

"""Bogoliubov sound speed √(e″(k₀)(2gn − δ)) of the fluid (√(ħ(2gn − δ)/m) for the parabola); 0 if negative."""
sound_speed(f::LocalFluid) = sqrt(max(kinetic_d2(f.pol, f.k0) * (2f.pol.g * f.n - f.δ), zero(f.n)))

"""`true` when `gn ≥ δ`: the square root in [`lda_omega`](@ref) is then real for every q."""
is_gapless_stable(f::LocalFluid) = f.pol.g * f.n >= f.δ

"""
    omega_max(down; krange=(-3, 3), npts=60001)

Upper edge of the Hawking window: the maximum laboratory frequency of the downstream
negative-norm branch.
"""
function omega_max(down::LocalFluid; krange=(-3.0, 3.0), npts::Integer=60001)
    ks = range(krange..., length=npts)
    maximum(k -> lda_omega(down, k, -1), ks)
end

"""
    omega_min(up; krange=(-3, 3), npts=60001)

Lower edge of the Hawking window: the minimum of the upstream positive-norm branch (its
effective mass gap).
"""
function omega_min(up::LocalFluid; krange=(-3.0, 3.0), npts::Integer=60001)
    ks = range(krange..., length=npts)
    max(zero(up.v), minimum(k -> lda_omega(up, k, +1), ks))
end

"""
    lda_roots(f, ω, branch; krange=(-3, 3), npts=60001)

All real wavevectors with `lda_omega(f, k, branch) == ω`, found by locating sign changes
on a fine grid and refining by bisection.
"""
function lda_roots(f::LocalFluid, ω, branch::Integer; krange=(-3.0, 3.0), npts::Integer=60001)
    ks = range(krange..., length=npts)
    h(k) = lda_omega(f, k, branch) - ω
    roots = eltype(ks)[]
    hp = h(ks[1])
    for i in 2:npts
        hc = h(ks[i])
        if hp == 0
            push!(roots, ks[i-1])
        elseif hp * hc < 0
            a, b, fa = ks[i-1], ks[i], hp
            for _ in 1:60
                c = (a + b) / 2
                fc = h(c)
                (fa * fc <= 0) ? (b = c) : (a = c; fa = fc)
            end
            push!(roots, (a + b) / 2)
        end
        hp = hc
    end
    roots
end

"""Group velocity ∂ω/∂k (µm/ps) on a branch, by central difference."""
group_velocity(f::LocalFluid, k, branch::Integer; h=1e-6) =
    (lda_omega(f, k + h, branch) - lda_omega(f, k - h, branch)) / 2h
