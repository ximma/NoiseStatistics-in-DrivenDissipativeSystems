# Exact linear response to a weak monochromatic probe, and the Bogoliubov norm balance.
#
# For a probe F_pr(x) e^{−iωt} (package convention i∂ₜψ = … + F), the deviation
# δψ = u e^{−iωt} + v* e^{+iωt} solves, to first order in F_pr,
#
#   (ω − M) (u, v) = (F_pr, 0)
#
# where M is `bogoliubov_matrix(s, ψ₀)`. This is the same discretised operator as the time
# stepper, so the result is the probe simulation without its time window,
# demodulation or O(dt²) splitting error.
#
# Norm continuity, from the linearised equations (κ(x) = amplitude damping):
#   ∂ₓJ = −2κ(x) ρ(x) + source,  J = (ħ/m) Im(u* ∂ₓu + v* ∂ₓv),  ρ = |u|² − |v|²
# The probe is the only source.

"""
    LinearResponse(s, ψ0)

Precomputes the Hessenberg form of the Bogoliubov operator of `s` around `ψ0` (one
O(N³) step). Each frequency is then an O(N²) shifted solve: see [`respond`](@ref).
"""
struct LinearResponse{T,H}
    s::Setup{T}
    ψ0::Vector{Complex{T}}
    hess::H
end
LinearResponse(s::Setup, ψ0::AbstractVector) = LinearResponse(s, Vector{ComplexF64}(ψ0), hessenberg(bogoliubov_matrix(s, ψ0)))

"""
    respond(lr, F, ω)

`(u, v)` solving `(ω − M)(u, v) = (F, 0)`. `F` is the probe profile on the grid (e.g.
`probe_profile(grid, probe)`) and ω is in ps⁻¹. u is the response at +ω (lab wavevectors
of `F`), v* the response at −ω.
"""
function respond(lr::LinearResponse, F::AbstractVector, ω)
    N = lr.s.grid.N
    b = vcat(ComplexF64.(F), zeros(ComplexF64, N))
    x = -((lr.hess - ω * I) \ b)
    x[1:N], x[N+1:2N]
end
respond(lr::LinearResponse, p::GaussianProbe) = respond(lr, probe_profile(lr.s.grid, p), p.ω)

_dx(s::Setup, f) = ifft(im .* s.grid.k .* fft(f))          # spectral derivative

"""Bogoliubov norm density ρ = |u|² − |v|²."""
norm_density(u, v) = abs2.(u) .- abs2.(v)

"""Bogoliubov norm current J(x) = (ħ/m) Im(u* ∂ₓu + v* ∂ₓv)  (spectral derivative). Parabolic dispersion
only: for a non-parabolic `kinetic` the current is not local; use the mode-resolved fluxes (`norm_flux`)."""
norm_current(s::Setup, u, v) = s.pol.ħ / s.pol.m .* imag.(conj.(u) .* _dx(s, u) .+ conj.(v) .* _dx(s, v))

"""
    dissipation(s, u, v, xa, xb)

`D = 2∫_{xa}^{xb} κ(x) ρ(x) dx` (trapezoidal rule on the grid). The exact balance between
two points with no source in between is `J(xb) − J(xa) = −D`.
"""
function dissipation(s::Setup, u, v, xa, xb)
    r = findall(x -> xa <= x <= xb, s.grid.x)
    f = 2 .* s.κ[r] .* norm_density(u[r], v[r])
    s.grid.dx * (sum(f) - (f[1] + f[end]) / 2)
end
