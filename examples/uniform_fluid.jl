# A homogeneous driven polariton fluid, flowing at constant velocity in a periodic box.
#
# What are the collective excitations of the fluid, and how many quanta does each of them hold in the stationary
# state when every input of the cavity is accounted for? The fluid is designed exactly (designed_setup), its
# Bogoliubov spectrum is computed on the grid and compared with the dispersion of a uniform fluid (LDA). The
# equal-time occupation n(k) of the fluctuations is computed from the linearised theory with
#   - the vacuum of the cavity mirror (the only input of a perfect cavity),
#   - lattice phonons at 4 K, 2 K and 0.3 K (Frérot et al., PRX 13, 041058 (2023)),
#   - the non-radiative excitonic port, per unit of its thermal occupation n_th (its value is sample-dependent),
# and the vacuum part is checked against truncated-Wigner trajectories of the full nonlinear equation.
#
#   julia -t 4 --project=examples examples/uniform_fluid.jl          (a few minutes on a laptop)
using PolaritonHorizons, CairoMakie, FFTW, LinearAlgebra, Printf, Random, Statistics

out = mkpath(joinpath(@__DIR__, "output"))
pol = lkb_gaas_polariton()                         # parabolic lower polariton, m* = 7.0e-35 kg, ħγ = 80 µeV
grid = Grid(100.0, 128)                            # L = 100 µm, dx = 0.78 µm, periodic
ħ, N, dx, L = pol.ħ, grid.N, grid.dx, grid.L

# Working point: subsonic flow, gapped spectrum on the upper branch of the bistability loop.
v, c = 0.4, 0.9                                    # velocity and Bogoliubov sound speed, µm/ps
δv = 0.30 / ħ                                      # effective detuning δ(v) = ω_p − ω_0 − e(k_p), ps⁻¹
δ0 = δv + kinetic(pol, wavevector_for_velocity(pol, v))
s, ψ0 = designed_setup(pol, grid; δ0, v=fill(v, N), c=fill(c, N), fix=1:N)   # fix: phase winds by 2πn over L
f = local_fluid(s, ψ0, 0.0, L)
M = bogoliubov_matrix(s, ψ0)
@printf("fluid: v = %.3f µm/ps, ħgn = %.3f meV, c_B = %.3f µm/ps, Mach %.2f; max Im ω/γ = %.4f (stable if < 0)\n",
    f.v, ħ * pol.g * f.n, bogoliubov_speed(s, f.n, f.v), f.v / bogoliubov_speed(s, f.n, f.v), maximum(imag, eigvals(M)) / pol.γ)

# Bogoliubov spectrum on the grid: each mode is a plane wave; its wavevector is that of the u component.
sp = bogoliubov_spectrum(s, ψ0)
kmode = [grid.k[argmax(abs.(fft(sp.u[:, j])))] for j in eachindex(sp.ω)]
kk = range(-2.5, 3.0, 400) .+ f.k0

# Occupation of the fluctuations in the lab momentum basis: a_k = (dx/√L) Σ_x e^{−ikx} δψ(x), [a_k, a_k†] = 1.
E = cis.(-grid.k * grid.x') .* (dx / sqrt(L))
occupation(C) = real.(diag(conj(E) * Fluctuations(s, ψ0, C).N * transpose(E)))
C0 = stationary_covariance(s, ψ0; M)                                           # vacuum of the mirror
n0 = occupation(C0)
nph = Dict(T => occupation(phonon_covariance(s, ψ0, PhononBath(; T); M)) .- n0 for T in (4.0, 2.0, 0.3))
nx = occupation(stationary_covariance(s, ψ0; M, noise=NoiseModel(s; radiative_fraction=0.0, n_x=1.0))) .- n0

# Truncated Wigner, vacuum inputs only, on the setup with V lowered by g/dx: the Wigner noise adds g/dx to the
# interaction energy on average, and the compensation makes the simulated flow the one of the linear theory.
stw = remake(s; V=s.V .- pol.g / dx)
ntraj, batch, Tburn, nsamp, Δts = 512, 128, 150.0, 40, 20.0
rng = Xoshiro(1)
acc = zeros(N); acc2 = zeros(N); nsamples = 0
t = @elapsed for b in 1:ntraj÷batch
    u0 = repeat(ψ0, 1, batch) .+ randn(rng, ComplexF64, N, batch) ./ sqrt(2dx)   # Wigner vacuum of the field
    _, ψb = evolve(stw, u0, (0.0, Tburn + nsamp * Δts); dt=0.02, nsaves=round(Int, (Tburn + nsamp * Δts) / Δts),
        noise=true, rng)
    for j in size(ψb, 3)-nsamp+1:size(ψb, 3), i in 1:batch
        a2 = abs2.(fft(ψb[:, i, j] .- ψ0) .* (dx / sqrt(L)))
        acc .+= a2; acc2 .+= a2 .^ 2; global nsamples += 1
    end
end
nW = acc ./ nsamples .- 0.5                                                    # normal order: ⟨|a|²⟩_W − 1/2
seW = sqrt.((acc2 ./ nsamples .- (acc ./ nsamples) .^ 2) ./ (ntraj * nsamp / 2))   # samples ~2 per correlation time
@printf("TWA: %d trajectories × %d samples in %.0f s\n", ntraj, nsamp, t)
z = (nW .- n0) ./ seW
@printf("TWA − linear (vacuum), |k − k_p| ≤ 2.5 µm⁻¹: median |z| = %.2f, max |z| = %.2f\n",
    median(abs.(z[abs.(grid.k .- f.k0) .<= 2.5])), maximum(abs.(z[abs.(grid.k .- f.k0) .<= 2.5])))

# Figure
q = fftshift(grid.k) .- f.k0; o = sortperm(grid.k); sel = abs.(q) .<= 2.5
fig = Figure(size=(1500, 430))
ax = Axis(fig[1, 1]; xlabel="k − k_p (µm⁻¹)", ylabel="ħ Re ω (meV)", title="collective excitations")
for (b, col) in ((1, :black), (-1, :red))
    lines!(ax, kk .- f.k0, ħ .* lda_omega.(Ref(f), kk, b); color=col)
end
scatter!(ax, kmode .- f.k0, ħ .* real.(sp.ω); color=[η > 0 ? :black : :red for η in sp.norm], markersize=5)
xlims!(ax, -2.5, 2.5); ylims!(ax, -1.5, 1.5)
text!(ax, -2.4, 0.0; text="lines: uniform-fluid dispersion\ndots: spectrum on the grid\nblack/red: positive/negative norm", fontsize=11, align=(:left, :center))
ax = Axis(fig[1, 2]; xlabel="k − k_p (µm⁻¹)", ylabel="occupation n(k)", yscale=log10, title="quanta per mode: loss ports")
lines!(ax, q[sel], n0[o][sel]; color=:black, label="vacuum of the mirror (linear)")
rangebars!(ax, q[sel], max.(nW[o][sel] .- 2seW[o][sel], 1e-8), nW[o][sel] .+ 2seW[o][sel]; color=(:gray, 0.6))
scatter!(ax, q[sel], max.(nW[o][sel], 1e-8); color=:gray, markersize=5, label="vacuum of the mirror (TWA, ±2σ)")
lines!(ax, q[sel], nx[o][sel]; color=:steelblue, linestyle=:dash, label="excitonic port, per unit n_th")
ylims!(ax, 1e-3, 20); axislegend(ax; position=:lb, labelsize=10)
ax = Axis(fig[1, 3]; xlabel="k − k_p (µm⁻¹)", ylabel="relative change of n(k)", title="lattice phonons (damping and noise)")
for (T, col) in ((4.0, :firebrick), (2.0, :darkorange), (0.3, :goldenrod))
    lines!(ax, q[sel], nph[T][o][sel] ./ n0[o][sel]; color=col, label="$T K")
end
hlines!(ax, [0.0]; color=:gray, linestyle=:dot); axislegend(ax; position=:rt)
save(joinpath(out, "uniform_fluid.png"), fig)
println("figure: ", joinpath(out, "uniform_fluid.png"))
