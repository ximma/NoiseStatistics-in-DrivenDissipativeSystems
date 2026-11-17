# Windows and Fourier analysis.
#
# Conventions: a field component ∝ e^{i(kx − ωt)} shows up at (+k, +ω). Spatial
# transforms are `fftshift(fft(·))` on the axis `kaxis(grid)`. Time demodulation at ω is
# Σ_j f(t_j) e^{+iω t_j}.

"""Symmetric Hann window of length `n` with zero end points (`WindowFunctions.hanning`)."""
hann(n::Integer) = n == 1 ? [1.0] : [0.5 * (1 - cos(2π * j / (n - 1))) for j in 0:n-1]

"""
    hann_window(grid, x_beg, x_end; convention=:grid)

Spatial Hann window supported on `[x_beg, x_end]`. With `convention=:grid` it spans the
grid points with `x_beg ≤ x ≤ x_end`. With `convention=:legacy` it spans the indices of the code of arXiv:2511.12339,
`floor(x_beg/dx) : floor(x_end/dx)` (1-based), which sit one point to the left.
"""
function hann_window(grid::Grid{T}, x_beg, x_end; convention::Symbol=:grid) where {T}
    w = zeros(T, grid.N)
    r = if convention === :legacy
        floor(Int, x_beg / grid.dx):floor(Int, x_end / grid.dx)
    else
        findfirst(>=(x_beg), grid.x):findlast(<=(x_end), grid.x)
    end
    w[r] .= hann(length(r))
    w
end

"""
    demodulate(samples, ω, Δt; time_window=true)

`Σ_j w_j f(:, j) e^{iω j Δt}` over the columns of `samples` (columns = times `j Δt`,
j = 1…nt), with an optional Hann window `w_j` in time. This picks out the component of `f`
that oscillates as e^{−iωt}.
"""
function demodulate(samples::AbstractMatrix, ω, Δt; time_window::Bool=true)
    nt = size(samples, 2)
    w = time_window ? hann(nt) : ones(nt)
    phases = [w[j] * cis(ω * j * Δt) for j in 1:nt]
    samples * phases
end

"""`fftshift(fft(v .* w))`, the windowed spatial spectrum on `kaxis(grid)`."""
spatial_spectrum(v::AbstractVector, w::AbstractVector) = fftshift(fft(v .* w))

"""
    spacetime_spectrum(grid, samples, wx, Δt)

|S(k, ω)|² of space–time samples (`N × nt`, columns = times `j Δt`) for a spatial window
`wx` and a Hann window in time. It returns `(S, k, ω)`, where `S[iω, ik]` has increasing ω and
k, and e^{i(kx−ωt)} maps to (+k, +ω).
"""
function spacetime_spectrum(grid::Grid, samples::AbstractMatrix, wx::AbstractVector, Δt)
    nt = size(samples, 2)
    A = (samples .* wx) .* transpose(hann(nt))
    S = abs2.(fftshift(bfft(fft(A, 1), 2)))          # fft in x, inverse-sign fft in t
    permutedims(S), kaxis(grid), fftshift(fftfreq(nt, 2π / Δt))
end
