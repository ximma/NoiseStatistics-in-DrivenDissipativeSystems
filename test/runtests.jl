using DrivenDissipativeNoise, Test, FFTW, Statistics, LinearAlgebra, DelimitedFiles, Random


@testset "DrivenDissipativeNoise" begin

    pol = Polariton()

    @testset "grid" begin
        g = Grid(800.0, 2048)
        @test g.dx == 800 / 2048
        @test g.k[1] == 0 && g.k[2] ≈ dk(g)
        @test issorted(kaxis(g)) && kaxis(g)[1025] == 0      # zero bin of fftshift(fft)
    end

    @testset "LDA dispersion" begin
        n, v = 1000.0, 1.0
        δ = pol.g * n                                       # sonic point: gn = δ, gapless
        f = LocalFluid(n, v, δ, pol.m * v / pol.ħ, pol)
        c = sqrt(pol.ħ * pol.g * n / pol.m)
        q = 1e-4
        # branches have a kink at q = 0 (ω = vq ± c|q|): use one-sided slopes
        @test (lda_omega(f, f.k0 + q, +1) - lda_omega(f, f.k0, +1)) / q ≈ v + c rtol = 1e-4
        @test (lda_omega(f, f.k0 + q, -1) - lda_omega(f, f.k0, -1)) / q ≈ v - c rtol = 1e-4
        # gapped case: ω(q=0) = ±√((gn−δ)(3gn−δ))
        f2 = LocalFluid(n, 0.0, 0.5pol.g * n, 0.0, pol)
        gn = pol.g * n
        @test lda_omega(f2, 0.0, +1) ≈ sqrt(0.5gn * 2.5gn)
        @test omega_min(f2) ≈ sqrt(0.5gn * 2.5gn) rtol = 1e-6
        # roots are consistent with the dispersion
        for ω in (0.1, 0.3), b in (1, -1), k in lda_roots(f, ω, b)
            @test lda_omega(f, k, b) ≈ ω atol = 1e-9
        end
    end

    @testset "bistability" begin
        δ = 0.4 / pol.ħ
        nl, nu = turning_points(pol, δ)
        dF(n) = (pump_intensity(pol, δ, n + 1e-3) - pump_intensity(pol, δ, n - 1e-3)) / 2e-3
        @test abs(dF(nl)) < 1e-6 * pump_intensity(pol, δ, nl) && abs(dF(nu)) < 1e-6 * pump_intensity(pol, δ, nu)
        @test sqrt(pump_intensity(pol, δ, δ / pol.g)) ≈ sonic_pump_amplitude(pol, δ)
        @test turning_points(pol, 0.5pol.γ) === nothing
    end

    @testset "windows and demodulation" begin
        @test hann(5) ≈ [0, 0.5, 1, 0.5, 0]
        g = Grid(800.0, 2048)
        w = hann_window(g, 70, 370; convention=:legacy)
        @test findfirst(!iszero, w) == floor(Int, 70 / g.dx) + 1 && count(!iszero, w) == 769 - 2
        # a component e^{-iωt} is picked by demodulation at +ω, not at −ω
        ω, Δt, nt = 0.4, 0.75, 3000
        X = reshape([cis(-ω * j * Δt) for j in 1:nt], 1, nt)
        @test abs(demodulate(X, ω, Δt)[1]) > 1e3 * abs(demodulate(X, -ω, Δt)[1])
    end

    @testset "homogeneous steady state (monostable)" begin
        g = Grid(100.0, 128)
        kp = 4 * dk(g)
        ωp = pol.ω0 + pol.ħ * kp^2 / (2pol.m) + 0.3pol.γ    # δ(k_p) = 0.3γ < √3γ/2: no bistability
        F = 0.5
        s = make_setup(pol, g; ωp, pump=[F * cis(kp * x) for x in g.x], wigner_shift=false)
        ψ = steady_state(s; T=600.0, dt=0.01)
        n = mean(abs2, ψ)
        δ = local_detuning(s, pol.ħ * kp / pol.m)
        @test pump_intensity(pol, δ, n) ≈ F^2 rtol = 1e-8
        @test std(abs2.(ψ)) / n < 1e-10
        @test mean(velocity(s, ψ)) ≈ pol.ħ * kp / pol.m rtol = 1e-10
    end

    @testset "probe convention and pump timing (linear driven oscillator)" begin
        # g = 0, no static pump, a uniform probe at k = 0: ψ(t) → A e^{-iωt}/(ω − D + iκ), D = −δ0
        p0 = Polariton(; g=0.0)
        g = Grid(50.0, 16)
        ωp = p0.ω0 + 0.2
        s = make_setup(p0, g; ωp, pump=zeros(ComplexF64, g.N), wigner_shift=false)
        ω, A = 0.7, 1.0
        pr = GaussianProbe(; x0=0.0, σ=1e9, k=0.0, ω, F=A)
        T, dt = 800.0, 0.01                                   # transient e^{-γT/2} ≈ 4e-13
        _, ψ = evolve(s, zeros(ComplexF64, g.N), (0.0, T); dt, probe=pr)
        exact = A * cis(-ω * T) / (ω - (-detuning0(s)) + im * p0.γ / 2)
        @test abs(ψ[1, end] - exact) / abs(exact) < 1e-4        # without the t_shift fix: ~ω dt = 7e-3
    end

    @testset "batched probes = independent runs" begin
        s = burkhard2026(; N=256, L=100.0)
        ψ0 = zeros(ComplexF64, s.grid.N)
        prs = [burkhard2026_probe(s, 0.3, 0.2; x0=50.0), burkhard2026_probe(s, 0.35, 0.4; x0=50.0)]
        _, a = evolve(s, ψ0, (0.0, 5.0); dt=0.005, probe=prs[2])
        _, b = evolve(s, repeat(ψ0, 1, 2), (0.0, 5.0); dt=0.005, probe=prs)
        @test b[:, 2, end] ≈ a[:, end] rtol = 1e-12
    end

    @testset "online batched demodulation = stored samples + demodulate" begin
        s = burkhard2026(; N=256, L=100.0)
        ψ0 = steady_state(s; T=50.0, dt=0.005)
        pc = ProbeProtocol(; t_relax=2.0, nsamples=40, Δt=0.25)
        prs = [burkhard2026_probe(s, 0.3, 0.2; x0=50.0), burkhard2026_probe(s, 0.4, 0.5; x0=50.0)]
        plus, minus = probe_demodulated(s, ψ0, prs; protocol=pc)
        for (b, p) in enumerate(prs)
            X = probe_samples(s, ψ0, p; protocol=pc)
            @test plus[:, b] ≈ demodulate(X, p.ω, pc.Δt) rtol = 1e-6
            @test minus[:, b] ≈ demodulate(X, -p.ω, pc.Δt) rtol = 1e-6
        end
    end

    @testset "Bogoliubov operator = analytic LDA on a homogeneous flow" begin
        g = Grid(100.0, 128)
        kp = 4 * dk(g)
        for ws in (false, true)
            ωp = pol.ω0 + pol.ħ * kp^2 / (2pol.m) - 0.3pol.γ
            s = make_setup(pol, g; ωp, pump=[0.5 * cis(kp * x) for x in g.x], wigner_shift=ws)
            ψ = steady_state(s; T=800.0, dt=0.01)
            sp = bogoliubov_spectrum(s, ψ)
            v = mean(velocity(s, ψ)); n = mean(abs2, ψ); δ = local_detuning(s, v)
            ωc(k, b) = (q = k - kp; ε = pol.ħ * q^2 / (2pol.m); gn = pol.g * n;
                v * q + b * sqrt(complex((ε + gn - δ) * (ε + 3gn - δ))) - im * pol.γ / 2)
            kN = π / g.dx
            ana = [ωc(k, b) for k in g.k for b in (1, -1) if abs(k - kp) < kN - 2kp]
            @test maximum(minimum(abs.(sp.ω .- a)) for a in ana) < 1e-10
            @test all(abs.(imag.(sp.ω) .+ pol.γ / 2) .< 1e-10)
        end
    end

    @testset "matrix pencil and wavepacket source" begin
        Δt = 0.1; t = (0:599) .* Δt
        y = @. 0.7exp(-0.036t) * cos(0.72t + 0.3) + 0.2exp(-0.05t) * cos(1.3t)
        ω, a = matrix_pencil(y, Δt; nmodes=4)
        @test any(w -> abs(w - (0.72 - 0.036im)) < 1e-8, ω) && any(w -> abs(w - (1.3 - 0.05im)) < 1e-8, ω)
        src = WavePacketSource(; x_stop=400.0, k0=0.5, wk=0.2, F=1.0, c=1.5, t_center=10.0)
        x = [380.0, 400.0, 401.0]
        F = src(x, 10.0)
        @test F[3] == 0 && F[2] ≈ im                     # package convention F = i·WP; zero downstream
        @test abs(src([400.0 - 1.5], 9.0)[1]) ≈ 1         # envelope centre moves at c
    end

    @testset "TWA moments and commutators" begin
        rngt = Random.Xoshiro(3)
        A = randn(rngt, ComplexF64, 6, 900); Bm = A[2:5, :] .+ 0.2randn(rngt, ComplexF64, 4, 900)
        m = Moments(6, 4)
        for r in Iterators.partition(1:900, 128); update!(m, A[:, r], Bm[:, r]); end
        Ia, Ib = abs2.(A), abs2.(Bm)
        C = [mean((Ia[i, :] .- mean(Ia[i, :])) .* (Ib[j, :] .- mean(Ib[j, :]))) for i in 1:6, j in 1:4]
        @test m.M2 ./ m.n ≈ C && m.G1 ≈ A * Bm' ./ 900
        m1, m2 = Moments(6, 4), Moments(6, 4); update!(m1, A[:, 1:300], Bm[:, 1:300]); update!(m2, A[:, 301:end], Bm[:, 301:end])
        @test DrivenDissipativeNoise.merge(m1, m2).M2 ≈ m.M2
        # identical windows: c12(k, k) = c11; gildeoliveira2026 windows equal the paper's (first index, length)
        s = gildeoliveira2026()
        w = SpatialWindow(s.grid, 262.0, 762.0)
        c11, c22, c12 = momentum_commutators(w, w, s.grid.dx)
        @test all(diag(c12) .≈ c11) && w.first_idx == 132 && length(w) == 250
        w1, w2 = SpatialWindow(s.grid, 486.0, 686.0), SpatialWindow(s.grid, 336.0, 536.0)
        @test (w1.first_idx, length(w1), w2.first_idx, length(w2)) == (244, 100, 169, 100)
    end

    @testset "TWA vacuum: ⟨|ψ|²⟩_W = 1/(2dx), n = 0" begin
        p0 = Polariton(; ħ=0.6582, m=1 / 6, γ=0.047 / 0.6582, g=3e-4 / 0.6582)
        g = Grid(128.0, 64)
        s = make_setup(p0, g; ωp=p0.ω0 + 0.5, pump=zeros(ComplexF64, g.N))
        acc = TWAAccumulator(s)
        twa_run!(acc, s, zeros(ComplexF64, g.N); ntraj=4000, batch=1000,
            protocol=TWAProtocol(; T=60.0, dt=0.1, initial=:vacuum), rng=Random.Xoshiro(5))
        m = acc.position
        nexp = (p0.γ / 2 * 0.1 / 2) / (1 - exp(-p0.γ / 2 * 0.1)) / g.dx       # discrete-time vacuum
        sem = sqrt(mean(diag(m.M2)) / m.n / m.n)
        @test abs(mean(m.μa) - nexp) < 4sem / sqrt(g.N)
        @test abs(mean(m.μa) * 2g.dx - 1) < 0.01
    end

    @testset "general geometry builder" begin
        # regional StepPump == the pump formula of the code of arXiv:2512.17807, bit for bit
        s = gildeoliveira2026()
        L, xs, Fu, Fd = 1024.0, 512.0 - 7, s.pump.F_up, s.pump.F_down
        Fm = [x < xs ? ((20 - Fu) * sech(x / 20) + Fu) * cis(0.148x) : ((20 - Fd) * sech((x - L) / 20) + Fd) * cis(0.614x) for x in s.grid.x]
        @test s.F == Fm
        @test maximum(abs, gildeoliveira2026(; support=false).F[520 .< s.grid.x .< 650]) < 1e-6   # only the edge boost near x = L remains
        @test_throws ArgumentError defect_height(:sideways)
        s2 = transcritical_setup(Polariton(), Grid(200.0, 256); ωp=Polariton().ω0 + 0.5, x_d=100.0, k_up=0.2, k_down=0.5,
            F_up=1.0, F_down=0.5, defect=0.3, absorbers=(EdgeAbsorber(200.0, 10.0, 1.0),))
        @test maximum(s2.V) ≈ 0.3 / Polariton().ħ
    end

    @testset "incident probes sit on the upstream branch" begin
        f = LocalFluid(1300.0, 1.1, 0.59, pol.m * 1.1 / pol.ħ, pol)
        s = burkhard2026()
        ωs = range(0.1, 0.8, 8)
        prs = incident_probes(s, f, ωs; x0=300.0, F=1e-3)
        @test length(prs) == 8
        @test all(p -> abs(lda_omega(f, p.k, +1) - p.ω) < 1e-9 && group_velocity(f, p.k, +1) > 0, prs)
    end

    @testset "time-sampled TWA (vacuum)" begin
        p0 = Polariton(; ħ=0.6582, m=1 / 6, γ=0.047 / 0.6582, g=3e-4 / 0.6582)
        g = Grid(128.0, 64)
        s = make_setup(p0, g; ωp=p0.ω0 + 0.5, pump=zeros(ComplexF64, g.N))
        acc = TWAAccumulator(s; position=10:40)
        twa_run!(acc, s, zeros(ComplexF64, g.N); ntraj=200, batch=200,
            protocol=TWAProtocol(; T=60.0, dt=0.1, initial=:vacuum, nsamples=40, Δt_sample=20.0), rng=Random.Xoshiro(9))
        @test acc.position.n == 8000 && size(acc.position.G1) == (31, 31)
        @test abs(mean(acc.position.μa) * 2g.dx - 1) < 0.02
    end

    @testset "mode-resolved scattering tools" begin
        f = LocalFluid(1300.0, 1.1, 0.59, pol.m * 1.1 / pol.ħ, pol)
        # 2×2 lab-frame Bogoliubov problem = LDA branches, symplectic norms ±1
        for k in (0.1, 0.35, 0.7)
            (ωp, Up, Vp), (ωm, Um, Vm) = bogoliubov_uv(f, k)
            @test ωp ≈ lda_omega(f, k, 1) && ωm ≈ lda_omega(f, k, -1)
            @test abs2(Up) - abs2(Vp) ≈ 1 && abs2(Um) - abs2(Vm) ≈ -1
        end
        # complex root solves ω + iκ = Ω_b(k_c) and reduces to the real root for κ → 0
        ω, κ = 0.4, 0.0357
        k = first(filter(kk -> group_velocity(f, kk, 1) > 0, lda_roots(f, ω, 1)))
        @test abs(complex_root(f, ω, 1, k; κ=0.0) - k) < 1e-10
        kc = complex_root(f, ω, 1, k; κ)
        @test imag(kc) > 0 && abs(imag(kc) - κ / group_velocity(f, k, 1)) < 0.1κ / group_velocity(f, k, 1)
        # fit_modes recovers known amplitudes from synthetic u, v fields
        ms = channel_modes(f, ω, :up; κ)
        x = collect(100.0:0.5:300.0); xref = 320.0; θ = 0.7
        ψ0 = sqrt(f.n) .* cis.(f.k0 .* x .+ θ)
        a0 = ComplexF64[0.3 + 0.1im * j for j in eachindex(ms)]
        u = sum(a0[j] * ms[j].U .* exp.(im * ms[j].kc .* (x .- xref)) for j in eachindex(ms))
        v = sum(a0[j] * ms[j].V .* exp.(im * ms[j].kc .* (x .- xref)) for j in eachindex(ms)) .* conj.((ψ0 ./ abs.(ψ0)) .^ 2)
        W = 7.5
        a, res = fit_modes(x, W .* u, W .* conj.(v), ψ0, ms; xref, W)
        @test a ≈ a0 rtol = 1e-8
        @test res < 1e-10
    end

    @testset "driven linear response and norm balance" begin
        g = Grid(200.0, 256)
        kp = 6 * dk(g)
        ωp = pol.ω0 + pol.ħ * kp^2 / (2pol.m) - 0.3pol.γ
        s = make_setup(pol, g; ωp, pump=[0.5 * cis(kp * x) for x in g.x], wigner_shift=true,
            absorbers=(EdgeAbsorber(200.0, 10.0, 2.0),))
        ψ = steady_state(s; T=800.0, dt=0.01)
        lr = LinearResponse(s, ψ)
        pr = GaussianProbe(; x0=60.0, σ=5.0, k=0.3, ω=0.4, F=1e-3)
        u, v = respond(lr, pr)
        M = bogoliubov_matrix(s, ψ)
        b = vcat(probe_profile(g, pr), zeros(ComplexF64, g.N))
        @test norm((pr.ω * I - M) * vcat(u, v) - b) < 1e-10 * norm(b)
        # norm continuity ∂ₓJ = −2κρ away from the probe
        J = norm_current(s, u, v)
        dJ = real.(DrivenDissipativeNoise._dx(s, complex.(J)))
        r = findall(x -> 100 <= x <= 170, g.x)
        @test norm((dJ .+ 2 .* s.κ .* norm_density(u, v))[r]) < 1e-2 * norm((2 .* s.κ .* norm_density(u, v))[r])   # aliasing-limited on a coarse grid
        # exact balance between two points without source
        Jx(x0) = J[argmin(abs.(g.x .- x0))]
        @test abs(Jx(170.0) - Jx(100.0) + dissipation(s, u, v, g.x[argmin(abs.(g.x .- 100.0))], g.x[argmin(abs.(g.x .- 170.0))])) < 0.02 * abs(Jx(100.0))
        # modal (Green) expansion = direct solve
        F = eigen(M)
        c = (F.vectors \ b) ./ (pr.ω .- F.values)
        @test F.vectors * c ≈ vcat(u, v) rtol = 1e-8
    end

    @testset "Bogoliubov fluctuations: noise ports, Lyapunov, Wick g2" begin
        g = Grid(100.0, 48)
        kp = 3 * dk(g)
        ωp = pol.ω0 + pol.ħ * kp^2 / (2pol.m) - 0.3pol.γ
        s = make_setup(pol, g; ωp, pump=[0.5 * cis(kp * x) for x in g.x], wigner_shift=true,
            absorbers=(EdgeAbsorber(100.0, 10.0, 2.0),))
        ψ = steady_state(s; T=600.0, dt=0.02)
        # fluctuation–dissipation: the vacuum is C = 1/(2dx) for any κ(x)
        @test stationary_covariance(s, zero(ψ)) ≈ I / (2g.dx) rtol = 1e-10
        M = bogoliubov_matrix(s, ψ)
        @test maximum(imag, eigvals(M)) < 0
        C = stationary_covariance(s, ψ; M)
        D = Diagonal(noise_diagonal(s))
        @test norm(-im * M * C + C * (-im * M)' + D) < 1e-10 * norm(D)
        # equal-time covariance = ∫ S(ω) dω/2π   (ω = c tan θ)
        c = 5.0
        θs = range(-π / 2, π / 2; length=6002)[2:end-1]
        Sint = sum(spectral_covariance(s, ψ, c * tan(θ); M) .* (c * sec(θ)^2 * step(θs) / 2π) for θ in θs)
        @test maximum(abs, Sint - C) < 1e-3 * maximum(abs, C)
        # excess (non-radiative, thermal) noise only adds fluctuations
        Cx = stationary_covariance(s, ψ; M, noise=NoiseModel(s; radiative_fraction=0.5, n_x=1.0))
        @test real(tr(Cx)) > real(tr(C))
        # window matrix = windowed_fft; normally ordered moments are Hermitian/symmetric
        w = SpatialWindow(g, 20.0, 60.0)
        @test window_matrix(w, g.N, g.dx) * ψ ≈ vec(windowed_fft(reshape(ψ, :, 1), w))
        fl = Fluctuations(s, ψ, C)
        @test fl.N ≈ fl.N'
        @test fl.M ≈ transpose(fl.M)
        _, na, _ = bogoliubov_g2m1(fl)
        @test na ≈ abs2.(ψ) .+ real.(diag(fl.N))
    end

    @testset "self-homodyne noise spectra around a coherent probe" begin
        g = Grid(100.0, 48)
        kp = 3 * dk(g)
        ωp = pol.ω0 + pol.ħ * kp^2 / (2pol.m) - 0.3pol.γ
        s = make_setup(pol, g; ωp, pump=[0.5 * cis(kp * x) for x in g.x], wigner_shift=true)
        ψ = steady_state(s; T=600.0, dt=0.02)
        A = ψ[1] * cis(-kp * g.x[1])
        ks, ωs, fo = 7dk(g), 0.8pol.γ, 0.6
        ki = 2kp - ks
        F = 1e-3 .* cis.(ks .* g.x)
        band(k) = (k - dk(g) / 3, k + dk(g) / 3)
        D = [Detector(:sig, s, (g.x[1], g.x[end]), band(ks)), Detector(:idl, s, (g.x[1], g.x[end]), band(ki))]
        @test D[1].k ≈ [ks] && D[2].k ≈ [ki]
        # coherent light through the linear lossy cavity (ψ₀ = 0, vacuum ports) stays shot-noise limited
        u, v = respond(LinearResponse(s, zero(ψ)), F, ωs)
        on = OutputNoise(s, zero(ψ); noise=NoiseModel(s; radiative_fraction=0.7), output_fraction=fo)
        I0, S0 = self_homodyne(on, D[1:1], coherent_components(on, u, v, ωs))
        @test fano(I0, S0)[1] ≈ 1 rtol = 1e-12
        # independent 2×2 mode-pair calculation (a_k, a_{2kp−k}†), thermal excitonic noise included
        nm = NoiseModel(s; radiative_fraction=0.7, n_x=0.5)
        u, v = respond(LinearResponse(s, ψ), F, ωs)
        on = OutputNoise(s, ψ; noise=nm, output_fraction=fo)
        Ig, Sg = self_homodyne(on, D, coherent_components(on, u, v, ωs))
        gg, κ = pol.g, pol.γ / 2
        L(q) = pol.ħ * q^2 / (2pol.m) - detuning0(s) + 2gg * abs2(A) - gg / g.dx - im * κ
        m(a, b) = [L(a) gg*A^2; -gg*conj(A)^2 -conj(L(b))]
        d, K2 = nm.κr[1] + nm.κnr[1] * (1 + 2nm.nx[1]), 2fo * nm.κr[1]
        E2 = Matrix{ComplexF64}(I, 2, 2)
        Sout(ω, a, b) = (G = inv(ω * E2 - m(a, b)); K2 * d * G * G' + (E2 - im * K2 * G + im * K2 * G') / 2)
        UV = (ωs * E2 - m(ks, ki)) \ [sum(cis.(-ks .* g.x) .* F) * g.dx / sqrt(g.L), 0]
        βs, βi = sqrt(K2) * UV[1], sqrt(K2) * conj(UV[2])
        W1, W2 = [βs 0; 0 conj(βi)], [0 βi; conj(βs) 0]          # ν = +ωs and ν = −ωs
        @test Ig ≈ abs2.([βs, βi]) rtol = 1e-10
        @test Sg ≈ W1' * Sout(ωs, ks, ki) * W1 + W2' * Sout(-ωs, ki, ks) * W2 rtol = 1e-10
        @test maximum(abs, imag(Sg)) < 1e-12 * maximum(abs, Sg)      # Ω → 0: real cross-spectra
        # detection efficiency η pulls the Fano factor towards 1: F_η = ηF + 1 − η
        Iη, Sη = with_efficiency(Ig, Sg, 0.5)
        @test fano(Iη, Sη) ≈ 0.5 .* fano(Ig, Sg) .+ 0.5
        # spontaneous terms (quadratic in the fluctuations), Schur route vs the 2×2 pair on the
        # same frequency grid
        ωg = range(-30.0, 30.0, length=24001)
        sn = spontaneous_noise(on, DetectionBasis(s, D), ωg)
        @test sn.tail < 1e-5
        Isp, Ssp = spontaneous_terms(sn, D)
        wt = [(i == 1 || i == length(ωg) ? 0.5 : 1.0) * step(ωg) / 2π for i in eachindex(ωg)]
        Ss, Si = [Sout(ω, ks, ki) for ω in ωg], [Sout(ω, ki, ks) for ω in ωg]
        nss, nii = [real(S[1, 1]) - 0.5 for S in Ss], [real(S[1, 1]) - 0.5 for S in Si]
        Pref = [sum(wt .* abs2.(nss)) sum(wt .* (abs2.(getindex.(Ss, 1, 2)) .+ abs2.(getindex.(Si, 1, 2)))) / 2;
                0 sum(wt .* abs2.(nii))]
        Pref[2, 1] = Pref[1, 2]
        @test Isp ≈ [sum(wt .* nss), sum(wt .* nii)] rtol = 1e-9
        @test Ssp ≈ Pref + Diagonal(Isp) rtol = 1e-9
        @test fano(Isp, Ssp)[1] > 1                                  # incoherent light is bunched
        @test sn.flux2 ≈ sn.flux rtol = 1e-8
        @test sn.power2 ≈ sn.power rtol = 1e-8
        # ... and the frequency-integrated flux equals the Lyapunov (equal-time) one. Without
        # thermal input (n_x = 0) n(ω) falls as ω⁻⁴; white thermal noise gives Lorentzian ω⁻² tails.
        nm0 = NoiseModel(s; radiative_fraction=0.7)
        onv = OutputNoise(s, ψ; noise=nm0, output_fraction=fo)
        snv = spontaneous_noise(onv, DetectionBasis(s, D), ωg)
        fl = Fluctuations(s, ψ, stationary_covariance(s, ψ; noise=nm0))
        @test spontaneous_terms(snv, D)[1] ≈ [spontaneous_flux(onv, fl, d) for d in D] rtol = 1e-7
        # output_scale: n, m ∝ κ_d
        I2, S2 = spontaneous_terms(sn, D; output_scale=0.5)
        @test I2 ≈ Isp / 2 && S2 - Diagonal(I2) ≈ (Ssp - Diagonal(Isp)) / 4
        # vacuum: no spontaneous light without the fluid, even with lossy ports
        on0 = OutputNoise(s, zero(ψ); noise=NoiseModel(s; radiative_fraction=0.7), output_fraction=fo)
        I0, S0 = spontaneous_terms(spontaneous_noise(on0, DetectionBasis(s, D), ωg), D)
        @test maximum(abs, I0) < 1e-12 && maximum(abs, S0) < 1e-12
        # frequency-resolved two-mode state (a_ks at +ωs, a_ki at −ωs) vs the 2×2 pair
        fs, fi = aperture_mode(s, (g.x[1], g.x[end]), ks), aperture_mode(s, (g.x[1], g.x[end]), ki)
        @test fs ≈ D[1].E[:, 1] && fi ≈ D[2].E[:, 1]
        nq, mq = mode_state(on, [(fs, 1), (fi, -1)], ωs)
        S1, S2m = Sout(ωs, ks, ki), Sout(-ωs, ki, ks)
        @test nq ≈ [S1[1, 1]-0.5 0; 0 S2m[1, 1]-0.5] atol = 1e-10
        @test mq ≈ [0 S1[1, 2]; S1[1, 2] 0] atol = 1e-10
        σq = quadrature_covariance(nq, mq)
        @test all(symplectic_eigenvalues(σq) .>= 0.5 - 1e-12)          # a physical state
        @test (log_negativity(σq, [1]) > 0) == (pair_ratio(nq, mq, 1, 2) > 1)
        # joint balanced homodyne (LOs at ω_p ± ωs, phases θ₁, θ₂) through self_homodyne
        β1, β2 = 30 * cis(0.4), 20 * cis(-1.1)
        Ih, Sh = self_homodyne(on, D, [(ωs, β1 .* fs), (-ωs, β2 .* fi)])
        quad(θ) = [cos(θ), sin(θ)]
        @test Ih ≈ abs2.([β1, β2])
        @test Sh[1, 1] / Ih[1] ≈ 2σq[1, 1] rtol = 1e-9                    # phase-insensitive: 2n + 1
        @test real(Sh[1, 2]) / sqrt(Ih[1] * Ih[2]) ≈ 2 * quad(angle(β1))' * σq[1:2, 3:4] * quad(angle(β2)) rtol = 1e-9
        # non-orthogonal profiles at the same frequency are rejected, and orthonormalised
        fb = aperture_mode(s, (g.x[1], g.x[24]), ks + 0.3im)
        @test_throws ArgumentError mode_state(on, [(fs, 1), (fb, 1)], ωs)
        mo = orthonormalize_modes([(fs, 1), (fb, 1), (fi, -1)], g.dx)
        @test reduce(hcat, first.(mo[1:2]))' * reduce(hcat, first.(mo[1:2])) .* g.dx ≈ I
        @test mo[3][1] ≈ fi
        @test minimum(symplectic_eigenvalues(quadrature_covariance(mode_state(on, mo, ωs)...))) >= 0.5 - 1e-12
        # two-mode squeezed vacuum: E_N = 2r/ln 2
        r = 0.7
        σt = quadrature_covariance(sinh(r)^2 * Matrix(I, 2, 2), sinh(r) * cosh(r) * [0 1; 1 0])
        @test log_negativity(σt, [1]) ≈ 2r / log(2)
        @test symplectic_eigenvalues(σt) ≈ [0.5, 0.5]
        # Schur route = Hessenberg route; the pixel basis gives any mode through transform_modes
        R = SchurResolvent(bogoliubov_matrix(s, ψ))
        n0, m0 = mode_state(on, mo, ωs)
        nS, mS = mode_state(on, mo, ωs; schur=R)
        @test nS ≈ n0 atol = 1e-10
        @test mS ≈ m0 atol = 1e-10
        Np = g.N
        np_, mp_ = pixel_state(on, 1:Np, ωs; schur=R)
        cm = zeros(ComplexF64, 2Np, 3)
        cm[1:Np, 1] .= sqrt(g.dx) .* mo[1][1]; cm[1:Np, 2] .= sqrt(g.dx) .* mo[2][1]; cm[Np+1:2Np, 3] .= sqrt(g.dx) .* mo[3][1]
        n3, m3 = transform_modes(np_, mp_, cm)
        @test n3 ≈ n0 atol = 1e-10
        @test m3 ≈ m0 atol = 1e-10
        # principal modes: diagonal occupations, entanglement unchanged by dropping near-vacuum modes
        pg = [collect(1:Np), collect(Np+1:2Np)]
        σp = quadrature_covariance(np_, mp_)
        cp, gp = principal_modes(np_, pg; rtol=1e-12)
        n4, m4 = transform_modes(np_, mp_, cp)
        @test maximum(abs, n4[gp[1], gp[1]] - Diagonal(n4[gp[1], gp[1]])) < 1e-10
        @test log_negativity(quadrature_covariance(n4, m4), gp[1]) ≈ log_negativity(σp, pg[1]) rtol = 1e-8
        # optimiser: a two-mode squeezed vacuum hidden in random local bases among thermal modes
        rng = Random.MersenneTwister(1)
        n6 = zeros(ComplexF64, 6, 6); m6 = zeros(ComplexF64, 6, 6)
        n6[1, 1] = n6[4, 4] = sinh(r)^2; m6[1, 4] = m6[4, 1] = sinh(r) * cosh(r)
        n6[2, 2], n6[3, 3], n6[5, 5], n6[6, 6] = 0.3, 0.1, 0.2, 0.05
        Uq, Vq = Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q), Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q)
        n7, m7 = transform_modes(n6, m6, [Uq zeros(3, 3); zeros(3, 3) Vq])
        ENo, co, so = optimise_modes(n7, m7, [1:3, 4:6], [1]; rng)
        @test ENo ≈ 2r / log(2) rtol = 1e-6
        # closed-form best pair: recovers the hidden pair, coth²r, and matches transform_modes
        Rb, ab, bb = max_pair_ratio(n7, m7, 1:3, 4:6)
        @test Rb ≈ coth(r)^2 rtol = 1e-10
        n8, m8 = transform_modes(n7, m7, [ab zeros(3); zeros(3) bb])
        @test pair_ratio(n8, m8, 1, 2) ≈ Rb rtol = 1e-10
        @test all(pair_ratio(n7, m7, i, j) <= Rb * (1 + 1e-12) for i in 1:3, j in 4:6)
        # homogeneous fluid: plane waves (k, 2k_p − k) are independent pairs, so the optimal
        # single pair is the best of them
        ENp, _, _ = optimise_modes(n4, m4, gp, [1]; rng, restarts=2)
        ENk = map(0:Np-1) do j
            k = j * dk(g); fk = aperture_mode(s, (g.x[1], g.x[end]), k); fk2 = aperture_mode(s, (g.x[1], g.x[end]), 2kp - k)
            log_negativity(quadrature_covariance(mode_state(on, [(fk, 1), (fk2, -1)], ωs)...), [1])
        end
        @test ENp ≈ maximum(ENk) rtol = 1e-6
    end

    @testset "lattice-phonon bath (Frérot et al. 2023)" begin
        E2 = Matrix{ComplexF64}(I, 2, 2)
        b4 = PhononBath(; T=4.0)
        @test phonon_shape(b4, 1.03 / 0.6582) ≈ 1 && phonon_shape(b4, 0.5) < 1 && phonon_occupation(PhononBath(; T=0.0), 0.3) == 0
        # 2×2 pair (a_k, a_{k′}†) with the bath: Σ and D_ph keep the output commutator, i.e. the
        # symmetric route (S_out − ½) equals direct normal ordering of the vacuum and phonon inputs
        κ, fo, Ap = 0.035, 0.6, 2.0 * cis(0.7); K2 = 2fo * κ; e = Ap^2 / abs2(Ap); w = [Ap, -conj(Ap)] ./ abs(Ap)
        mp = [0.4-im*κ 0.36*e; -0.36*conj(e) -conj(0.55-im*κ)]
        Σ2(ω, R) = (c = -0.5im * sign(ω) * R; [c c*e; -c*conj(e) -c])
        D2(R, nT) = R * (nT + 0.5) * [1 -e; -conj(e) 1]
        for (R, T) in ((0.02, 0.0), (0.3, 10.0)), ω in (-0.3, 0.8)
            nT = phonon_occupation(PhononBath(; T), ω); G = inv(ω * E2 - mp - Σ2(ω, R))
            Sout = K2 * G * (κ * E2 + D2(R, nT)) * G' + (E2 - im * K2 * G + im * K2 * G') / 2
            @test real(Sout[1, 1]) - 0.5 ≈ K2 * (2κ * abs2(G[1, 2]) + R * (ω < 0 ? nT + 1 : nT) * abs2((G*w)[1])) rtol = 1e-10
        end
        # fluid at rest, phonon-driven light: ghost/normal = |v/u|² e^{ħω/k_BT} (Frérot Eq. 29)
        κ = 0.001 / 0.6582; K2 = 2κ; gn = 0.19 / 0.6582; a = 1.0 / 0.6582
        m0 = [a-im*κ gn; -gn -(a+im*κ)]; ωq = sqrt(a^2 - gn^2); vu2 = ((a - ωq) / gn)^2
        function nphonon(ω, b, R)
            G = inv(ω * E2 - m0 - (c = -0.5im * sign(ω) * R; [c c; -c -c]))
            Sout = K2 * G * (κ * E2 + R * (phonon_occupation(b, ω) + 0.5) * [1 -1; -1 1]) * G' + (E2 - im * K2 * G + im * K2 * G') / 2
            real(Sout[1, 1]) - 0.5 - K2 * 2κ * abs2(G[1, 2])
        end
        @test nphonon(-ωq, b4, 1e-4) / nphonon(ωq, b4, 1e-4) ≈ vu2 * exp(0.6582 * ωq / (DrivenDissipativeNoise.KB_MEV * 4.0)) rtol = 1e-3
        b0 = PhononBath(; T=0.0)                                   # T = 0: ghost line only
        @test abs(nphonon(ωq, b0, 1e-4)) < 1e-3 * nphonon(-ωq, b0, 1e-4)
        # grid implementation (PhononNoise) = the 2×2 pair model, flowing homogeneous fluid
        g = Grid(100.0, 48); kp = 3dk(g); ωp = pol.ω0 + pol.ħ * kp^2 / (2pol.m) - 0.3pol.γ
        s = make_setup(pol, g; ωp, pump=[0.5 * cis(kp * x) for x in g.x], wigner_shift=true)
        A = 3.0 * cis(0.4); ψ = A .* cis.(kp .* g.x); gg, κ = pol.g, pol.γ / 2
        b = PhononBath(; T=6.0, strength=40.0); pn = PhononNoise(OutputNoise(s, ψ; output_fraction=fo), b)
        ks = 7dk(g); ki = 2kp - ks; ωs = 0.8pol.γ
        (n, mo), = phonon_mode_state(pn, [(cis.(ks .* g.x) ./ sqrt(g.L), 1), (cis.(ki .* g.x) ./ sqrt(g.L), -1)], ωs)
        Lq(q) = pol.ħ * q^2 / (2pol.m) - detuning0(s) + 2gg * abs2(A) - gg / g.dx - im * κ
        e = A^2 / abs2(A); R = phonon_rates(b, s, ψ, ωs)[1]; K2 = 2fo * κ
        function Spair(ω, q1, q2)
            G = inv(ω * E2 - [Lq(q1) gg*A^2; -gg*conj(A)^2 -conj(Lq(q2))] - (c = -0.5im*sign(ω)*R; [c c*e; -c*conj(e) -c]))
            K2 * G * (κ * E2 + R * (phonon_occupation(b, ω) + 0.5) * [1 -e; -conj(e) 1]) * G' + (E2 - im * K2 * G + im * K2 * G') / 2
        end
        S1, S2 = Spair(ωs, ks, ki), Spair(-ωs, ki, ks)
        @test real(n[1, 1]) ≈ real(S1[1, 1]) - 0.5 rtol = 1e-8
        @test real(n[2, 2]) ≈ real(S2[1, 1]) - 0.5 rtol = 1e-8
        @test abs(mo[1, 2]) ≈ abs(S1[1, 2]) rtol = 1e-8
        # strength → 0 recovers mode_state
        (n0, m0_), = phonon_mode_state(PhononNoise(pn.on, PhononBath(; T=6.0, strength=0.0)), [(cis.(ks .* g.x) ./ sqrt(g.L), 1)], ωs)
        @test n0 ≈ mode_state(pn.on, [(cis.(ks .* g.x) ./ sqrt(g.L), 1)], ωs)[1] rtol = 1e-8
    end

    @testset "burkhard2026 configuration" begin
        s = burkhard2026()
        @test s.pump.k_up ≈ 0.27 + 2π / 800 && s.pump.k_down ≈ 0.539 + 2π / 800
        @test minimum(s.V) ≈ -0.85 / pol.ħ rtol = 1e-12
        @test maximum(burkhard2026(; defect=:repulsive).V) ≈ 0.85 / pol.ħ rtol = 1e-12
        @test all(iszero, burkhard2026(; defect=:none).V)
        @test burkhard2026(; support=false).pump.F_down == 0
    end

    @testset "all-optical design: exact pump, frozen reservoir" begin
        pol = lkb_gaas_polariton(); gr = Grid(200.0, 512); ħ = pol.ħ; β = 1.84
        x = gr.x
        v = [waterfall(xi; xH=100.0, vu=0.2, vd=1.0, wH=2.0) for xi in x] .+ (0.2 - 1.0) .* (tanh.((x .- 180) ./ 3) .+ 1) ./ 2
        fix = findall(xi -> 160 <= xi <= 195, x)
        s, ψt = designed_setup(pol, gr; δ0=0.2 / ħ, v, c=fill(0.4, gr.N), β, absorbers=(EdgeAbsorber(200.0, 10.0, 4.5 / ħ),), fix)
        @test maximum(abs, s.Vres .- β .* pol.g .* abs2.(ψt)) < 1e-12           # frozen reservoir g_r n_r = β g n
        ψ = steady_state(s; T=300.0, dt=0.01, u0=ψt)
        @test maximum(abs, ψ .- ψt) / maximum(abs, ψt) < 1e-3                   # the target is the stationary state (O(dt²))
        p = profiles(s, ψ); r = findall(xi -> 20 <= xi <= 90 || 110 <= xi <= 150, x)
        @test maximum(abs, p.cB[r] .- 0.4) < 1e-3                               # Bogoliubov speed with the reservoir
        up = local_fluid(s, ψ, 20, 90); dn = local_fluid(s, ψ, 110, 150)
        @test isapprox(up.v / sqrt(ħ * (2pol.g * up.n - up.δ) / pol.m), 0.5; atol=2e-3)
        @test isapprox(dn.v / sqrt(ħ * (2pol.g * dn.n - dn.δ) / pol.m), 2.5; atol=5e-3)
        # the design target is the exact self-consistent state (V_res = βg|ψt|²): Newton recovers it
        # from the time-stepped state (O(dt²) away), and it is stable with an adiabatic reservoir
        sN, ψN = reservoir_newton(s, ψ; β)
        @test maximum(abs, ψN .- ψt) / maximum(abs, ψt) < 1e-9
        @test maximum(abs, sN.Vres .- s.Vres) / maximum(abs, s.Vres) < 1e-9
        @test maximum(imag, eigvals(reservoir_jacobian(sN, ψN; β))) < 0
    end

    @testset "lower-polariton dispersion" begin
        p0 = lkb_gaas_polariton(); ħ = p0.ħ
        lpol(Δ; Ω=5.1) = Polariton(; ħ, m=p0.m, γ=p0.γ, g=p0.g, ω0=p0.ω0, ΩR=Ω / ħ, Δxc=Δ / ħ)
        lp = lpol(0.0)
        @test is_parabolic(p0) && !is_parabolic(lp)
        @test kinetic(lp, 0.0) == 0 && isapprox(kinetic_d2(lp, 0.0), ħ / p0.m; rtol=1e-12)    # m* kept at k = 0
        @test isapprox(kinetic(lpol(0.0; Ω=1e6), 1.5), kinetic(p0, 1.5); rtol=1e-4)           # large Rabi: parabola
        @test photonic_fraction(lp, 0.0) ≈ 0.5 && photonic_fraction(lpol(-2.0), 0.0) > 0.5
        for k in (0.7, 1.3, 0.9 + 0.05im), h in (1e-5,)                                      # analytic derivatives
            @test isapprox(kinetic_d1(lp, k), (kinetic(lp, k + h) - kinetic(lp, k - h)) / 2h; rtol=1e-7)
            @test isapprox(kinetic_d2(lp, k), (kinetic_d1(lp, k + h) - kinetic_d1(lp, k - h)) / 2h; rtol=1e-6)
        end
        ki = inflection_k(lp); @test abs(kinetic_d2(lp, ki)) < 1e-10 && max_velocity(lp) ≈ kinetic_d1(lp, ki)
        k1 = wavevector_for_velocity(lp, 1.0); @test kinetic_d1(lp, k1) ≈ 1.0 && 0 < k1 < ki
        @test wavevector_for_velocity(lp, -0.5) ≈ -wavevector_for_velocity(lp, 0.5)
        @test_throws ArgumentError wavevector_for_velocity(lp, 1.01max_velocity(lp))

        # Bogoliubov matrix = generalised LDA on a homogeneous lower-polariton flow at large k
        g = Grid(100.0, 128); kp = 20dk(g)                                                  # 1.26 µm⁻¹
        s0 = make_setup(lp, g; ωp=lp.ω0 + kinetic(lp, kp) + 0.2 / ħ, pump=zeros(ComplexF64, g.N))
        ψ = fill(sqrt(0.25 / ħ / lp.g), g.N) .* cis.(kp .* g.x)                            # gn = 0.25 meV/ħ
        s = remake(s0; F=exact_pump(s0, ψ))
        @test stationarity(s, ψ) < 1e-6                                          # O(dt²) splitting residual
        sp = bogoliubov_spectrum(s, ψ); f = local_fluid(s, ψ, 0, 100)
        @test isapprox(f.k0, kp; rtol=1e-12) && isapprox(f.v, kinetic_d1(lp, kp); rtol=1e-12)
        ωc(k, b) = (e1 = kinetic(lp, k); e2 = kinetic(lp, 2kp - k); ε = (e1 + e2) / 2 - kinetic(lp, kp); gn = lp.g * f.n;
            (e1 - e2) / 2 + b * sqrt(complex((ε + gn - f.δ) * (ε + 3gn - f.δ))) - im * lp.γ / 2)
        ana = [ωc(k, b) for k in g.k for b in (1, -1) if abs(k - kp) < π / g.dx - 2kp]
        @test maximum(minimum(abs.(sp.ω .- a)) for a in ana) < 1e-9
        @test all(abs(lda_omega(f, k, 1) - real(ωc(k, 1))) < 1e-12 for k in (0.3, 1.0, 2.0) if imag(ωc(k, 1)) ≈ -lp.γ / 2)

        # design, exact pump and time stepping with the lower-polariton dispersion
        gr = Grid(200.0, 512); x = gr.x; β = 1.84
        v = [waterfall(xi; xH=100.0, vu=0.2, vd=1.0, wH=2.0) for xi in x] .+ (0.2 - 1.0) .* (tanh.((x .- 180) ./ 3) .+ 1) ./ 2
        fix = findall(xi -> 160 <= xi <= 195, x)
        sL, ψt = designed_setup(lp, gr; δ0=0.2 / ħ, v, c=fill(0.4, gr.N), β, absorbers=(EdgeAbsorber(200.0, 10.0, 4.5 / ħ),), fix)
        ψL = steady_state(sL; T=300.0, dt=0.01, u0=ψt)
        @test maximum(abs, ψL .- ψt) / maximum(abs, ψt) < 1e-3                  # GGP dispersion = exact_pump kinetic
        pL = profiles(sL, ψL); r = findall(xi -> 20 <= xi <= 90 || 110 <= xi <= 150, x)
        @test maximum(abs, pL.cB[r] .- 0.4) < 1e-3 && maximum(abs, pL.v[r] .- v[r]) < 1e-3
        dn = local_fluid(sL, ψL, 110, 150)
        @test isapprox(dn.v / sound_speed(dn), 2.5; atol=5e-3)
        @test dn.k0 > p0.m * 1.0 / ħ                                             # heavier than the parabola at k > 0
    end
end
