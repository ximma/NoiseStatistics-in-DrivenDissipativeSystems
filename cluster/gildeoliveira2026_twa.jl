# TWA statistics for arXiv:2512.17807 on a CUDA GPU (one job = `ntraj` trajectories).
#
#   julia --project=cluster cluster/gildeoliveira2026_twa.jl OUTDIR JOB_ID NTRAJ [BATCH=10000] [Float64|Float32]
#
# Each job writes OUTDIR/moments_JOB_ID.jls (the merge-able `TWAAccumulator`). Combine them
# with cluster/merge_moments.jl. The noise RNG is CUDA's default RNG, seeded with JOB_ID,
# so jobs are independent. Runs started from the stored steady state of the paper
# (STEADY_STATE env var, a JLD2 file with key "steady_state") or from a fresh mean field.
using PolaritonHorizons, CUDA, Random, Serialization, Printf, JLD2
const PH = PolaritonHorizons

outdir, job, ntraj = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3])
batch = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10_000
T = length(ARGS) >= 5 && ARGS[5] == "Float32" ? Float32 : Float64
mkpath(outdir)
CUDA.functional() || error("no functional CUDA device")
Random.seed!(CUDA.default_rng(), job)

s = gildeoliveira2026()
ψ0 = if haskey(ENV, "STEADY_STATE")
    jldopen(f -> f["steady_state"][1], ENV["STEADY_STATE"]) .* (-im)       # data files of arXiv:2512.17807: pump convention +iF
else
    evolve(s, zeros(ComplexF64, s.grid.N), (0.0, 400.0); dt=0.2, nsaves=1, envelope=gildeoliveira2026_envelope, pump_timing=:ggp)[2][:, end]
end
# window pairs of the paper (Fig. 2): separated [−26, 172]/[−176, 22] µm and identical [−250, 248] µm around x_d
xd = 512.0
pairs = [(SpatialWindow(s.grid, xd - 26, xd + 174), SpatialWindow(s.grid, xd - 176, xd + 24)),
         (SpatialWindow(s.grid, xd - 250, xd + 250), SpatialWindow(s.grid, xd - 250, xd + 250))]
acc = TWAAccumulator(s, pairs; array=CuArray, T)
file = joinpath(outdir, @sprintf("moments_%05d.jls", job))
t = @elapsed twa_run!(acc, s, ψ0; ntraj, batch, protocol=gildeoliveira2026_protocol(), rng=nothing, array=CuArray, T, progress=true)
host(m::Moments) = Moments(m.n, Array(m.μa), Array(m.μb), Array(m.G1), Array(m.M2))
serialize(file, (; position=host(acc.position), momentum=host.(acc.momentum), pairs, T=string(T), job, ntraj))
@printf("job %d: %d trajectories (%s) in %.1f min → %s\n", job, ntraj, T, t / 60, file)
