# One quasinormal mode of an acoustic horizon, obtained three independent ways.
#
# The flow is the published geometry of Burkhard et al., C. R. Phys. 25, 341 (2026) (arXiv:2511.12339): an attractive
# Gaussian defect makes the flow transcritical, with downstream density support. Its Bogoliubov spectrum has a mode
# localised at the horizon, of mixed norm, at ħΩ ≈ 0.47 meV. The example finds it
#   (1) by diagonalising the Bogoliubov operator on the grid (`bogoliubov_spectrum`, `find_qnm`);
#   (2) in the time domain: a wavepacket is sent at the horizon in the full nonlinear equation, and the free decay of
#       the density at the defect is fitted by the matrix-pencil method (`ringdown`, `matrix_pencil`);
#   (3) in the stationary driven response: a weak probe on the incoming upstream branch is swept in frequency
#       (`LinearResponse`, `respond`), and the response at the defect is fitted by a smooth background plus a pole.
#
#   julia -t 4 --project=examples examples/qnm_three_ways.jl          (about ten minutes on a laptop)
using PolaritonHorizons, CairoMakie, LinearAlgebra, Printf

out = mkpath(joinpath(@__DIR__, "output"))
s = burkhard2026()
ħ, γ, x = s.pol.ħ, s.pol.γ, s.grid.x
t = @elapsed ψ = steady_state(s)
xH = horizon(s, ψ; near=400.0)
up, dn = local_fluid(s, ψ, 70, 365), local_fluid(s, ψ, 401, 745)
@printf("stationary state (%.0f s): x_H = %.2f µm; Hawking window ħω ∈ [%.3f, %.3f] meV\n", t, xH, ħ * omega_min(up), ħ * omega_max(dn))

# (1) diagonalisation
t = @elapsed sp = bogoliubov_spectrum(s, ψ)
jq = first(find_qnm(sp, xH; zero_norm=0.5))
λ1 = sp.ω[jq]
@printf("(1) Bogoliubov spectrum (%.0f s): ħΩ = %.4f meV, Im ω/γ = %.4f, norm η = %+.2f\n", t, ħ * real(λ1), imag(λ1) / γ, sp.norm[jq])

# (2) ringdown after a wavepacket, in the full nonlinear equation
c = 1.4922                                                   # speed of the moving source (µm/ps)
src = WavePacketSource(; x_stop=400.0, k0=0.5, wk=0.2, F=s.pump.F_up / 100, c, t_center=(400 + 80) / c)
tf = packet_front_time(src)
t = @elapsed ts, δn = ringdown(s, ψ, src; t_end=src.t_center + 2tf + 200, Δt=0.1, dt=0.005)
r = findall(τ -> src.t_center + 2tf <= τ <= src.t_center + 2tf + 120, ts)
ωs2, a2 = matrix_pencil(δn[r] .- sum(δn[r]) / length(r), 0.1; nmodes=6)
λ2 = ωs2[argmax([real(w) > 0.05 ? abs(a2[i]) : 0.0 for (i, w) in enumerate(ωs2)])]
@printf("(2) ringdown (%.0f s): ħΩ = %.4f meV, Im ω/γ = %.4f\n", t, ħ * real(λ2), imag(λ2) / γ)

# (3) driven linear response to a probe on the incoming upstream branch. In each uniform region the response is
# fitted to the local Bogoliubov modes (`channel_modes`, `fit_modes`); the transmission t(ω) from the incoming
# upstream mode at x_a to the transmitted downstream mode at x_b, with the propagation phases removed, is fitted by a
# smooth background plus one pole.
t = @elapsed lr = LinearResponse(s, ψ)
x0, σ, κ = 150.0, 5.0, γ / 2
probes = incident_probes(s, up, range(0.30, 0.56, 131) ./ ħ; x0, σ, F=s.pump.F_up / 1000)
xa, xb = xH - 45, xH + 40
ru = findall(xi -> x0 + 4σ <= xi <= xa, x); rd = findall(xi -> xb <= xi <= xb + 240, x)
upf, dnf = local_fluid(s, ψ, x[ru[1]], x[ru[end]]), local_fluid(s, ψ, x[rd[1]], x[rd[end]])
function forward_mode(f, region, r, xref, u, v, ω)        # amplitude at xref of the positive-norm mode moving to +x
    ms = channel_modes(f, ω, region; κ)
    a, _ = fit_modes(x[r], u[r], conj.(v[r]), ψ[r], ms; xref)
    j = findfirst(m -> m.branch == 1 && m.vg > 0, ms)
    a[j], real(ms[j].k)
end
ω3 = [p.ω for p in probes]
tr = map(probes) do p
    u, v = respond(lr, p)
    ain, kin = forward_mode(upf, :up, ru, xa, u, v, p.ω)
    adn, kdn = forward_mode(dnf, :down, rd, xb, u, v, p.ω)
    adn / ain * cis(-(kin * (xH - xa) + kdn * (xb - xH)))
end
basis(λ; deg=4) = hcat([(ω3 .- real(λ1)) .^ n for n in 0:deg]..., 1 ./ (ω3 .- λ))
fitpole(λ) = (cf = basis(λ) \ tr; (cf, norm(basis(λ) * cf - tr) / norm(tr)))
grid_re = real(λ1) .+ range(-0.06, 0.06, 121); grid_im = imag(λ1) .* range(0.5, 1.5, 101)
res = [fitpole(complex(a, b))[2] for a in grid_re, b in grid_im]
ij = argmin(res); λ3 = complex(grid_re[ij[1]], grid_im[ij[2]])
@printf("(3) driven response (%.0f s for the operator, %d frequencies): ħΩ = %.4f meV, Im ω/γ = %.4f (fit residual %.1e; %.1e without the pole)\n",
    t, length(ω3), ħ * real(λ3), imag(λ3) / γ, res[ij], fitpole(complex(real(λ1), 1e3))[2])

fig = Figure(size=(1300, 420))
ax = Axis(fig[1, 1]; xlabel="ħ Re ω (meV)", ylabel="Im ω / γ", title="(1) Bogoliubov spectrum")
pos = findall(j -> 0 <= real(sp.ω[j]) <= ħ^-1, eachindex(sp.ω))
scatter!(ax, ħ .* real.(sp.ω[pos]), imag.(sp.ω[pos]) ./ γ; color=[η > 0 ? :black : :red for η in sp.norm[pos]], markersize=4)
scatter!(ax, [ħ * real(λ1)], [imag(λ1) / γ]; color=:transparent, strokecolor=:blue, strokewidth=2, markersize=14)
vlines!(ax, ħ .* [omega_min(up), omega_max(dn)]; color=:gray, linestyle=:dash)
xlims!(ax, 0, 1); ylims!(ax, -0.6, -0.4)
ax = Axis(fig[1, 2]; xlabel="t − t_c (ps)", ylabel="δn/n̄ at the defect", title="(2) ringdown")
lines!(ax, ts .- src.t_center, δn; color=:black); vspan!(ax, 2tf, 2tf + 120; color=(:blue, 0.1)); xlims!(ax, -40, 2tf + 150)
ax = Axis(fig[1, 3]; xlabel="ħω (meV)", ylabel="|t(ω)|²", title="(3) driven response: transmission")
cf, _ = fitpole(λ3)
lines!(ax, ħ .* ω3, abs2.(tr); color=:black, linewidth=2)
lines!(ax, ħ .* ω3, abs2.(basis(λ3) * cf); color=:orange, linestyle=:dash)
lines!(ax, ħ .* ω3, abs2.(basis(λ3)[:, 1:end-1] * cf[1:end-1]); color=:steelblue, linestyle=:dot)
vlines!(ax, [ħ * real(λ1)]; color=:blue, linestyle=:dot)
text!(ax, 0.305, 0.0; text="black: exact response\norange: background + pole\nblue: background alone", fontsize=11, align=(:left, :bottom))
save(joinpath(out, "qnm_three_ways.png"), fig)
println("figure: ", joinpath(out, "qnm_three_ways.png"))
