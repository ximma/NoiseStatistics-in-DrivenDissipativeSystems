# Momentum correlations of the light emitted by a polariton horizon, from truncated-Wigner trajectories: Fig. 2(c) of
# Gil de Olivera et al., Phys. Rev. A 114, L021701 (2026) (arXiv:2512.17807).
#
# The geometry and the protocol are those of the paper (`gildeoliveira2026`, `gildeoliveira2026_protocol`): each
# trajectory starts from the stationary state with the vacuum noise of the loss ports, runs 400 ps at dt = 0.2 ps,
# and is sampled once at the end. The density-density correlations g2(k, k′) − 1 of the field in one Hann window
# centred on the horizon show the Hawking–partner correlation lines. The paper used 5.5×10⁸ trajectories; the
# structure of Fig. 2(c) is already visible with 10⁵ (about half an hour on a 4-core laptop), and the default here is
# 2×10⁴ (environment NTRAJ). As in the paper, the TWA runs on the same setup as the mean field; the Wigner term g/dx
# (see docs/methods.md) changes this flow by less than 0.3 %.
#
#   NTRAJ=20000 julia -t 4 --project=examples examples/reproduce_momentum_correlations.jl
using DrivenDissipativeNoise, CairoMakie, FFTW, Printf, Random

out = mkpath(joinpath(@__DIR__, "output"))
s = gildeoliveira2026()
dx, ntraj = s.grid.dx, parse(Int, get(ENV, "NTRAJ", "20000"))
_, ψT = evolve(s, zeros(ComplexF64, s.grid.N), (0.0, 400.0); dt=0.2, nsaves=1,
    envelope=gildeoliveira2026_envelope, pump_timing=:ggp)              # the paper's start-up protocol
ψ0 = ψT[:, end]
w = SpatialWindow(s.grid, 262.0, 762.0)        # the window of Fig. 2(c): 250 points from x = 262 µm, centred on the horizon
acc = TWAAccumulator(s, [(w, w)]; position=false)
t = @elapsed twa_run!(acc, s, ψ0; ntraj, batch=2000, protocol=gildeoliveira2026_protocol(), rng=Xoshiro(1))
@printf("%d trajectories in %.1f min\n", ntraj, t / 60)

g = g2m1(acc.momentum[1], momentum_commutators(w, w, dx))
k = fftshift(window_k(w, dx))
fig = Figure(size=(560, 500))
ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="k (µm⁻¹)", ylabel="k′ (µm⁻¹)", title="g2(k, k′) − 1  (×10³), $ntraj trajectories")
hm = heatmap!(ax, k, k, 1e3 .* fftshift(g); colorrange=(-2, 2), colormap=:inferno)
Colorbar(fig[1, 2], hm)
save(joinpath(out, "momentum_correlations.png"), fig)
println("figure: ", joinpath(out, "momentum_correlations.png"))
