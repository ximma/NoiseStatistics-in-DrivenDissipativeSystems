# Running the TWA statistics of arXiv:2512.17807 on a CUDA cluster

One-time setup on the cluster (needs network access):

```bash
julia --project=cluster -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["CUDA", "JLD2"]); Pkg.instantiate()'
julia --project=cluster -e 'using CUDA; CUDA.versioninfo()'
```

A short test, then the array job:

```bash
julia --project=cluster cluster/gildeoliveira2026_twa.jl results/test 1 20000 10000
sbatch cluster/gildeoliveira2026_twa.slurm
julia --project=cluster cluster/merge_moments.jl results/gildeoliveira2026
```

Notes:
- Seeds: each job seeds CUDA's default RNG with its job id. A CPU `Xoshiro` cannot fill GPU arrays.
- Throughput reference: the 5.5×10⁸ trajectories of arXiv:2512.17807 took days on one GPU at batch 10⁴.
- Precision: Float64 by default, as in the paper. Float32 is allowed for TWA but must first be validated against Float64 on a short run of the same size.
- Combining runs: moments from different machines combine exactly (`DrivenDissipativeNoise.merge`), so CPU and GPU runs can be pooled.
- Untested: the GPU path has only been reviewed, not run. The first run should be the short test above, compared with a CPU run of the same size.

