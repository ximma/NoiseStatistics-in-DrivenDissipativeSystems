# Building blocks of the all-optically controlled geometry: the potential landscape, the
# losses, and the structured coherent pump.

gaussian(x, x0, w) = exp(-(x - x0)^2 / w^2)

"""
    GaussianDefect(x0, w, V0)

Real potential `V(x) = V0 exp(-(x-x0)²/w²)`, with `V0` in ps⁻¹ (an energy divided by ħ).
`V0 < 0` is attractive and `V0 > 0` repulsive. `V0 = 0` means no defect.
"""
struct GaussianDefect{T}
    x0::T
    w::T
    V0::T
end

"""
    EdgeAbsorber(x0, w, κ)

Extra *amplitude* damping `κ exp(-(x-x0)²/w²)` (ps⁻¹). The code of arXiv:2511.12339 writes it as an
imaginary potential `-iκ G(x)`. It absorbs waves before they wrap around the periodic box.
"""
struct EdgeAbsorber{T}
    x0::T
    w::T
    κ::T
end

"""
    SechBoost(x0, w, ΔF, k)

Additive pump term `ΔF sech((x-x0)/w) e^{ikx}`. Near a boundary it pushes the fluid onto
the upper branch of the bistability loop.
"""
struct SechBoost{T}
    x0::T
    w::T
    ΔF::T
    k::T
end

"""
    StepPump(; x_step, k_up, k_down, F_up, F_down, boosts=SechBoost[], regional=false)

Structured coherent pump made of two plane waves joined at `x_step`:

    F(x) = F_up e^{i k_up x}      upstream (subcritical side)
           F_down e^{i k_down x}  downstream (supercritical side)
           + Σ boosts

The local pump wavevector sets the flow velocity ħk_p/m. The amplitude sets the operating
point on the bistability loop. `F_down = 0` means a downstream region without density support.

There are two conventions, each reproducing one original code:
- `regional=false` (arXiv:2511.12339): x ≤ x_step is upstream, and boosts apply everywhere.
- `regional=true` (arXiv:2512.17807): x < x_step is upstream, and each boost applies only within the
  region containing its centre `x0`.
"""
Base.@kwdef struct StepPump{T}
    x_step::T
    k_up::T
    k_down::T
    F_up::T
    F_down::T
    boosts::Vector{SechBoost{T}} = SechBoost{T}[]
    regional::Bool = false
end

function pump_amplitude(p::StepPump, x)
    up(y) = p.regional ? y < p.x_step : y <= p.x_step
    if p.regional
        # form of arXiv:2512.17807, ((F_M − F) sech + F) e^{ikx}, with the boosts of the region of x
        A = up(x) ? p.F_up : p.F_down
        k = up(x) ? p.k_up : p.k_down
        for b in p.boosts
            up(b.x0) == up(x) && (A += b.ΔF * sech((x - b.x0) / b.w))
        end
        return A * cis(k * x)
    end
    F = up(x) ? p.F_up * cis(p.k_up * x) : p.F_down * cis(p.k_down * x)
    for b in p.boosts
        F += b.ΔF * sech((x - b.x0) / b.w) * cis(b.k * x)
    end
    F
end

"""
    GaussianProbe(; x0, σ, k, ω, F)

Weak monochromatic probe `F exp(-(x-x0)²/σ²) e^{i(kx - ωt)}`, where ω is measured in the frame
rotating at the pump frequency.
"""
Base.@kwdef struct GaussianProbe{T}
    x0::T
    σ::T
    k::T
    ω::T
    F::T
end

probe_profile(grid::Grid, p::GaussianProbe) = [p.F * gaussian(x, p.x0, p.σ) * cis(p.k * x) for x in grid.x]
