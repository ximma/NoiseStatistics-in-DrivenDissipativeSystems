# Merge the per-job moment files written by gildeoliveira2026_twa.jl into one file.
#   julia --project=cluster cluster/merge_moments.jl OUTDIR [merged.jls]
using PolaritonHorizons, Serialization, Printf
const PH = PolaritonHorizons
outdir = ARGS[1]
files = filter(f -> startswith(f, "moments_") && endswith(f, ".jls"), readdir(outdir))
isempty(files) && error("no moments_*.jls in $outdir")
parts = [deserialize(joinpath(outdir, f)) for f in files]
pos = reduce(PH.merge, [p.position for p in parts])
mom = [reduce(PH.merge, [p.momentum[i] for p in parts]) for i in eachindex(parts[1].momentum)]
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(outdir, "merged.jls")
serialize(out, (; position=pos, momentum=mom, pairs=parts[1].pairs, njobs=length(parts)))
@printf("merged %d jobs, %d trajectories → %s\n", length(parts), pos.n, out)
