# All-optical design of a flow: choose the velocity v(x) and the Bogoliubov sound speed c(x), compute the pump that
# sustains exactly that flow, and check that the driven-dissipative equation, integrated in time from the target,
# stays on it.
#
# With control of both the amplitude and the phase of the pump (a spatial light modulator), any smooth target field
# is an exact stationary state of the driven-dissipative equation for one pump profile (`exact_pump`). The example
# designs a subsonic flow that accelerates over a 20 µm wide region and slows down again, with a frozen exciton
# reservoir (β = 1.84, Falque et al., PRL 135, 023401 (2025)), and limits the pump to the spatial frequencies an
# imaging system of numerical aperture 0.4 transmits. It then checks the linear stability of the designed state and
# integrates the full equation from it.
#
#   julia -t 4 --project=examples examples/inverse_design.jl          (about a minute)
using DrivenDissipativeNoise, CairoMakie, LinearAlgebra, Printf

out = mkpath(joinpath(@__DIR__, "output"))
pol = lkb_gaas_polariton()
grid = Grid(200.0, 512)
ħ, x, N = pol.ħ, grid.x, grid.N
β = 1.84                                                     # frozen reservoir g_r n_r = β g n

vt = [0.4 + bump(xi; x0=100.0, A=0.3, hwhm=10.0) for xi in x]  # target velocity (µm/ps), peak 0.7
ct = fill(0.9, N)                                            # target Bogoliubov sound speed (µm/ps)
δ0 = 0.30 / ħ + kinetic(pol, wavevector_for_velocity(pol, 0.4))
λ, NA = 0.835, 0.4                                           # wavelength (µm) and numerical aperture
kmax = 2π * NA / λ
# fix=1:N: the wavevector is shifted by a constant everywhere so that the phase winds by 2πn over the periodic box.
s, ψt = designed_setup(pol, grid; δ0, v=vt, c=ct, β, fix=1:N, kmax,
    description="inverse-design example")

M = bogoliubov_matrix(s, ψt)
@printf("pump band limit |k| ≤ %.2f µm⁻¹; max Im ω/γ of the designed state = %.4f (stable if < 0)\n",
    kmax, maximum(imag, eigvals(M)) / pol.γ)

ψ = steady_state(s; u0=ψt, T=3000.0)                         # time integration, starting on the designed branch
@printf("stationarity of the integrated state: %.1e\n", stationarity(s, ψ))
pt, ps = profiles(s, ψt), profiles(s, ψ)
@printf("integrated vs target: max |Δn|/n = %.1e, max |Δv| = %.1e µm/ps, max |Δc_B| = %.1e µm/ps\n",
    maximum(abs.(ps.n ./ pt.n .- 1)), maximum(abs.(ps.v .- pt.v)), maximum(abs.(ps.cB .- pt.cB)))

fig = Figure(size=(1150, 420))
ax = Axis(fig[1, 1]; xlabel="x (µm)", ylabel="|F| (ps⁻¹ µm⁻¹ᐟ²)", title="pump (|k| ≤ $(round(kmax; digits=2)) µm⁻¹)")
lines!(ax, x, abs.(s.F); color=:black)
ax2 = Axis(fig[1, 1]; ylabel="local wavevector of the pump (µm⁻¹)", yaxisposition=:right, ylabelcolor=:steelblue)
hidexdecorations!(ax2)
lines!(ax2, x[2:end], rem2pi.(diff(angle.(s.F)), RoundNearest) ./ grid.dx; color=:steelblue)
ax = Axis(fig[1, 2]; xlabel="x (µm)", ylabel="µm/ps", title="flow: target (dashed) and integrated (solid)")
lines!(ax, x, pt.v; color=:black, linestyle=:dash, label="v target")
lines!(ax, x, ps.v; color=:black, label="v")
lines!(ax, x, pt.cB; color=:firebrick, linestyle=:dash, label="c_B target")
lines!(ax, x, ps.cB; color=:firebrick, label="c_B")
axislegend(ax; position=:rb)
save(joinpath(out, "inverse_design.png"), fig)
println("figure: ", joinpath(out, "inverse_design.png"))
