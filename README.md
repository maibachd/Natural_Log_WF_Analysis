# Natural Log Project – deformed ringdowns and echoes

This folder contains a small pipeline that computes gravitational-wave **echoes seeded by a
natural-log deformed ringdown**, following the benchmark of `The_Natural_Log_Project.pdf`
(equation numbers "Eq. (n)" refer to that draft).

The echo machinery re-implements the echo computation of

* **[D25]** N. Deppe, L. Heisenberg, H. Inchauspé, L. E. Kidder, D. Maibach, S. Ma, J. Moxon,
  K. C. Nelli, W. Throwe, N. L. Vu, *Echoes from beyond: detecting gravitational-wave quantum
  imprints with LISA*, PRD 111, 124035 (2025), [arXiv:2411.05645](https://arxiv.org/abs/2411.05645),
  Appendix A (ref. [20] of the PDF),

which follows

* **[M22]** S. Ma, Q. Wang, N. Deppe, F. Hébert, L. E. Kidder, J. Moxon, W. Throwe, N. L. Vu,
  M. A. Scheel, Y. Chen, *Gravitational-wave echoes from numerical-relativity waveforms via
  spacetime construction near merging compact objects*, PRD 105, 104007 (2022).

The wall reflectivity and the echo delay follow

* **[WOA19]** Q. Wang, N. Oshita, N. Afshordi, *Echoes from quantum black holes*, PRD 101, 024031
  (2020), [arXiv:1905.00446](https://arxiv.org/abs/1905.00446) (ref. [17] of the PDF),
* **[OA23]** N. Oshita, N. Afshordi, *From entropy to echoes: counting the quasi-normal modes and
  the quantum limit of silence*, PLB 841, 137901 (2023),
  [arXiv:2302.08964](https://arxiv.org/abs/2302.08964) (refs. [15] and [24] of the PDF).

Parts taken from the original implementation of that computation (not included here) are marked
`[orig]` in the code. Everything they do is written out in the docstrings and in the section
[Origin of the echo machinery](#origin-of-the-echo-machinery) below, so this repository is
self-contained.

## Idea

1. From the CCE data of an SXS event, compute the radiation falling into the remnant,
   `Y_22^in = FFT[Psi0(ω) C_in(ω)]` ([D25] Eqs. A2, A9). This field seeds the echoes.
2. Determine the ringdown start `t0` (`v_Σ` of [D25]) as the minimum of the mismatch between
   `Y_in` and a QNM model, scanned over start times. `t0` is where the echo seed starts.
3. Deform the Kerr QNM spectrum of the true remnant, `ω^Kerr_22n(M_f, χ_f)` (SXS metadata), with
   the natural-log benchmark (Eqs. 13–14):

   ```
   ω_n → ω_n (1 + δω_n),   τ_n → τ_n (1 + δτ_n),   δ{ω,τ}_n = c^{(ω,τ)}_n λ_NH Λ_log^{-p},
   Λ_log = ln(ζ S_BH(M_f, χ_f))                                     (Eq. 5)
   ```

   with ω = Re ω the oscillation frequency and τ = −1/Im ω the damping time.
4. **Build the modified ringdown** of `h` and `Y_in` (next section): the NR waveform with its QNM
   content replaced by the deformed one. `λ_NH = 0` is GR.
5. Compute echoes from the modified `Y_in` with the transfer function of [D25], where the
   reflectivity is the Boltzmann reflectivity of the quantum horizon ([WOA19] Eqs. 1, 13, 15) with
   the natural-log delay (Eqs. 9–10):

   ```
   K(ω)    = -R_wall / ((1 - R_wall R_BH) 4 D_out)                              ([D25] Eq. A11)
   R_wall  = exp(-|ω̃|/(2 T_H)) exp(+i(ω̃ t_echo + φ_echo)),   ω̃ = ω - m Ω_H
   t_echo  = α_echo/(2π T_H) (Λ_log + Δ_ℓ),   Δ_ℓ = ln[κ_echo/(ζ ℓ(ℓ+1))]   (Eqs. 9, 10 = Eq. 15)
   h_echo  = -4 FFT[K Y_seed / ω²]
   ```

   `T_H` and `Ω_H` are those of the Kerr remnant (Eqs. 3–4). The total strain is
   `h_total = h_hybrid + h_echo`.

## Modified ringdown

The PDF fixes only the deformed QNM *frequencies*. It does not say how the deformed ringdown is
attached to the waveform, nor what happens to the mode amplitudes. Two constructions are
implemented; `--method residual` is the default.

### Default: `residual` (NR data + deformation)

```
h_mod(t) = h_NR(t) + w(t) Σ_n [ A_n (e^{-iω'_n τ} - e^{-iω_n τ}) + B_n (e^{+iω'_n* τ} - e^{+iω_n* τ}) ],   τ = t - t_a
```

* `ω_n` is the Kerr spectrum, `ω'_n` the deformed one, and `B_n` the mirror modes `−ω_n*`.
* `A_n`, `B_n` are the QNM amplitudes at `t_a`. They come from a linear least-squares fit with
  **only the QNMs whose amplitudes are resolved in the data**:
  * by default the fundamental mode and the first overtone (`--n-overtones 1`);
  * the fit runs from `t_fit = 10 M` after the peak (`--t-fit`) to `100 M` (`--upper-t`) and
    includes a constant (memory/offset) term, which is not deformed.
* `w(t)` is a Planck taper that switches the deformation on between `t_a = 5 M` after the peak
  (`--t-attach`) and `t_a + 5 M` (`--switch-width`). Since the bracket vanishes at `t = t_a`,
  there is no jump.
* The same construction is applied to `Y_in`, with its own fit.
* `λ_NH = 0` reproduces the NR data exactly, up to where the continuation (below) starts. The
  echoes for `λ_NH = 0` are therefore exactly those of the original method.
* The amplitude stability is checked automatically. Each fit is repeated from `t_fit ∓ 5 M`, and
  the relative spread of `sqrt(|A_n|² + |B_n|²)`, referenced to the peak, is printed and stored.
  The script warns above 10% for n = 0 and above 50% for overtones. A physical QNM has an
  amplitude that does not depend on the fit start; an overfitted one does.
* Beyond the end of the trustworthy data, both fields are continued by their GR QNM fit:
  * the switch happens at the start of the Tukey window of the Ψ0 FFT, where `Y_in` begins to be
    tapered, with a 50 M Planck blend (`--cont-width`);
  * the deformation term continues analytically.

  This extension is needed because echo delays are hundreds of M while the data end
  430–890 M after the peak.

**Why this construction.**

* It follows the strategy of the parametrized EOB models used in LIGO–Virgo tests of the ringdown.
  These keep the GR waveform and change the QNM frequencies only in the ringdown part, with the
  overtones held at GR or absorbed into a smooth envelope:
  * pEOBNR: R. Brito, A. Buonanno, V. Raymond, [arXiv:1805.00293](https://arxiv.org/abs/1805.00293);
  * pSEOBNRv4HM: A. Ghosh, R. Brito, A. Buonanno, [arXiv:2104.01906](https://arxiv.org/abs/2104.01906),
    whose merger-ringdown is `ν Ã(t) e^{iφ̃(t)} e^{−iσ_220 (t−t_match)}` with only σ_220 deformed.
* Restricting the deformed content to QNMs that are stable in the data follows the overtone-content
  literature:
  * V. Baibhav et al., [arXiv:2302.03050](https://arxiv.org/abs/2302.03050): higher overtones
    fitted near the peak mostly "fit away" initial-data effects, tails and nonlinearities;
  * E. Coleman, E. Finch, [arXiv:2512.08098](https://arxiv.org/abs/2512.08098): amplitudes of high
    overtones are strongly correlated and hard to measure.
* The deformation prescription itself (Eqs. 13–14, `c = 1` for every mode in the model) is not
  changed. With `t_a = 5 M`, deforming n = 0 alone, n ≤ 1 or n ≤ 2 gives nearly the same result
  (tested on 1936 and 1424):
  * the first-echo change differs by at most 0.4 percentage points (1424: 1.6%, 1.9%, 2.0%);
  * the early deviation of h differs by at most 0.015 of the peak.

  Leaving out the higher overtones therefore hardly changes the result; it removes amplitudes
  that are not physical.

**Choice of `t_a` (main systematic).** When the deformation is switched on is a modelling choice that
the PDF does not fix. It is the main systematic of the modified ringdown. Change of the first echo,
`|e(λ=+1) − e(GR)| / |e(GR)|`, for event 1936 with only the fundamental mode deformed:

| `t_a` | −5.5 M (= t0) | 0 (peak) | +5 M (default) | +10 M |
|---|---|---|---|---|
| echo change | 5.8% | 3.5% | 2.2% | 1.4% |

`t_a = 5 M` is the earliest time from which the fundamental-mode amplitude of h is stable for the
low-spin events. Its spread under fits from 5, 10 and 15 M is ≤ 4%, except 0206 with 12%. From
`t_a = 5 M` on, the result also hardly depends on which overtones are included. Before that, the QNM description itself is not reliable. For example, for 1424 with
`t_a = t0` and n ≤ 2 deformed, the deformation reaches 27× the peak.


## Wall reflectivity and echo parameters

* **Boltzmann reflectivity** ([WOA19] Eqs. 3, 13): `R_wall = e^{−|ω̃|/2T_H} exp[−iω̃ ln(γ|ω̃|)/(πT_H)]`,
  with γ the dissipation constant of the near-horizon modes (Planck units).
  * The flux reflectivity `|R_wall|² = e^{−|ω̃|/T_H}` does not depend on γ.
  * γ only sets the position of the wall, `x₀ = ln(γ|ω̃|)/(2πT_H)` ([WOA19] Eq. 14), i.e. the
    delay `Δt_echo = −ln(γ|ω̃|)/(πT_H)` ([WOA19] Eq. 15).
* **Delay and κ_echo.** [OA23] Eq. 20 gives `t_echo = (2πT_H)⁻¹ ln[S_BH/(π ℓ(ℓ+1) γ²)]` (γ in
  Planck times). This is Eq. 10 with `α_echo = 1` and **`κ_echo = 1/(πγ²)`**: γ enters the pipeline
  only through κ_echo.
* **Default:** `α_echo = 1`, `κ_echo = 55166/(4π) = 4390` (`nl_utils.KAPPA_ECHO_OA23`), the
  dissipation time `γ = (4/55166)^{1/2} t_Pl = 8.5·10⁻³ t_Pl` for which the entropy of the cavity
  modes equals S_BH ([OA23] Eq. 19). `φ_echo = 0`.
* **Differences to Eq. 16 of the PDF:**
  * no amplitude prefactor `γ_echo` (the PDF puts γ there; in [WOA19, OA23] it only enters the delay);
  * `|ω̃|` in the Boltzmann factor ([WOA19] Eqs. 1, 13);
  * `exp(+iω̃ t_echo)`: a delay in this code's Fourier convention ([WOA19] Eq. 3 has the same sign,
    since ln(γ|ω̃|) < 0).
* **Approximation:** the delay of [WOA19] Eq. 15 depends logarithmically on ω̃; the pipeline uses the
  constant t_echo of Eq. 10. For M = 10⁶ M☉ and |ω̃| = 0.05–0.5/M, [WOA19] Eq. 15 gives
  803–822 M (1936) and 921–942 M (1155) for γ = 1, against t_echo = 796 M and 912 M from Eq. 10
  with α_echo = 1, κ_echo = 1/π, i.e. agreement to 1–3%.


## Files

| file | purpose |
|---|---|
| `nl_utils.py` | All shared functions: data loading, FFT and coefficient helpers, Kerr QNMs, QNM fit and stability test, t0 scan, natural-log deformation, both modified-ringdown constructions, Kerr thermodynamics (T_H, Ω_H, S_BH, Λ_log), wall and barrier reflectivities, transfer function, echo strain, I/O helpers. Each function is marked as copied (`[orig]`), adapted (`[orig, adapted]`, with what changed and why), or new (unmarked). |
| `compute_modified_ringdown.py` | **Step 1.** Steps 1–4 of the idea for each event. Writes `output/ringdown/ringdown_<event>.h5`. |
| `compute_echoes.py` | **Step 2.** Step 5 for every deformed ringdown and every echo-parameter set. Also checks the `λ_NH = 0` echo against the echo of the original `Y_in`. Writes `output/echoes/echoes_<event>.h5`. |
| `plot_test_run.py` | **Step 3.** Two test figures for one event: (1) original vs. deformed ringdowns, (2) full time series with echoes for several echo parameters, plus the first echo for the different ringdown deformations. Writes `output/plots/`. |
| `fetch_remnant_metadata.py` | Regenerates `remnant_metadata.json` from the SXS catalogue. Only needed for new events; requires `sxs` and internet. |
| `remnant_metadata.json` | Remnant mass and spin, plus the initial binary parameters, of the 10 events (SXS catalogue v3.0.0). |
| `coefficients/CCoeff.h5`, `coefficients/DCoeff.h5` | Tabulated barrier coefficients `C_in/out`, `D_in/out` of the ℓ = 2 Schwarzschild potential (χ = 0), used in [D25]. See [Origin of the echo machinery](#origin-of-the-echo-machinery). Used with `--schwarzschild-coefficients`. |
| `coefficients/chi_<χ>/{C,D}Coeff.h5`, `Eigenvalue.h5` | The same tables for a Kerr remnant of spin χ (signed, 4 decimals, e.g. `chi_+0.0217`), plus the spheroidal eigenvalue. Computed automatically on first use and then reused. See [Barrier coefficients for spinning remnants](#barrier-coefficients-for-spinning-remnants). |
| `coefficients/Coeff_QG_Mem.nb` | Mathematica notebook that documents how the tables are computed with the BH Perturbation Toolkit. |
| `coefficients/compute_coefficients.wl` | Non-interactive version of the notebook's Teukolsky chapter with the spin as a parameter; called by `nl_utils.load_bh_coefficients`. |
| `<event>/{Strain,Psi0,Psi4}.h5` | CCE data per event (modes `Y_l{l}_m{m}.dat`: time, Re, Im). Folders are named `205`, `1936`, …; the code accepts `205` and `0205`. |

## Running

```bash
pip install -r requirements.txt          # qnm is the only package not in a standard scientific stack
# Kerr barrier tables (first run per remnant only): Mathematica with the Teukolsky package of the
# BH Perturbation Toolkit, see "Barrier coefficients for spinning remnants"

python compute_modified_ringdown.py --events 1936            # step 1 (default: all 10 events, ~5 s each)
python compute_echoes.py --events 1936                       # step 2
python plot_test_run.py --event 1936                         # step 3
```

Important options (see `--help`):

* `compute_modified_ringdown.py`
  * `--method residual|replace` – construction of the modified ringdown (default `residual`).
  * `--lambda-nh -1 0 1` – λ_NH values (0 = GR).
  * `--p 0.5`.
  * `--c-omega 1`, `--c-tau 1` – one value, or one per overtone of the model.
  * `--m-total 1e6` – detector-frame total mass in M☉. It sets S_BH, and through it Λ_log and t_echo.
  * `--zeta 1`.
  * `--n-overtones` – highest deformed overtone (default 1 for residual, 3 for replace).
  * `residual` only:
    * `--t-attach 5` (t_a);
    * `--t-fit 10`;
    * `--stability-step 5`, `--stability-tol-fundamental 0.1`, `--stability-tol-overtones 0.5`;
    * `--cont-width 50`.
  * `--switch-width 5` – switch-on of the deformation (residual) or data-to-model blend (replace).
  * `--n-scan 3`, `--t0-min/-max/-num` – the t0 scan (default `linspace(-20, 10, 121)` as in the
    original implementation, which used n = 5).
  * `--upper-t 100` – end of the fit windows.
  * `--t-end 3000` – end of the time grid; it must contain the echoes.
  * `--schwarzschild-coefficients` – use the χ = 0 tables of [D25] for every event instead of the
    Kerr tables for the remnant spin. `compute_echoes.py` always uses the tables of the ringdown file.
* `compute_echoes.py`
  * `--echo alpha_echo=1,kappa_echo=4390,phi_echo=0,zeta=1` – repeatable, one per echo-parameter set.
  * Default: one set, α_echo = 1, κ_echo = 55166/(4π) = 4390, φ_echo = 0 (see
    [Wall reflectivity and echo parameters](#wall-reflectivity-and-echo-parameters)).
* `plot_test_run.py`
  * `--event`.
  * `--lambda-echo 1` – which deformed ringdown is used in the echo-parameter comparison.

## Output files

`output/ringdown/ringdown_<event>.h5`

* attributes:
  * remnant: `remnant_mass`, `remnant_chi`, `sxs_id`
  * `method`, `t0` (scanned ringdown start = start of the echo seed)
  * `t_attach` (`t_a`, or `t0` for replace), `t_ref` (time the QNM coefficients refer to), `switch_width`
  * `n_overtones`, `n_scan`, `upper_t`, `Lambda_log`, `p`, `zeta`, `m_total_solar`, `t_peak`
  * `coeff_tables` (folder of the barrier tables, `coefficients` = χ = 0 tables of [D25]),
    `coeff_chi` (spin of the tables); the same two attributes are in the echo file
  * fit mismatches
  * residual only: `t_fit`, `t_cont`, `cont_width`
  * replace only: `jump_h`, `jump_Y_in` (fit residual at `t0` / peak)
* `grid/` – `t_start`, `dt`, `n_data`, `n_ext` of the uniform (extended) time grid.
* `data/` – `time`, `h`, `Y_in` on the uniform grid of the NR data.
* `scan/` – `t0_grid`, `mismatch`.
* `qnm/` – the spectrum, fits and deformations:
  * `omegas_kerr`
  * `coeffs_h`, `coeffs_Y_in` – basis `[1, e^{-iω_n(t-t_ref)}, e^{+iω_n^*(t-t_ref)}, …]`
  * `stability_h`, `stability_Y_in` – relative amplitude spread per mode (residual only)
  * `lambda_nh`, `c_omega`, `c_tau`, `delta_omega`, `delta_tau`, `omegas_mod`
  * `deviation_h_Y_in[λ, field]` – `max|deformed − GR|` on `[t_attach, upper_t)`, relative to the
    ringdown peak `max|field|` on `[t0, upper_t)`.
* `hybrid/` – the modified waveforms on `t ≥ −1000 M`: `time`, `h[λ, t]`, `Y_in[λ, t]`.

`output/echoes/echoes_<event>.h5`

* `time`
* `lambda_nh`
* `t_echo[set]`
* `max_round_trip[set]` – max |R_wall R_BH| (≥ 1: the echo sum diverges)
* `precursor[set]` – echo before the ringdown relative to its maximum (FFT wrap-around)
* `echo_params/*`
* `h_echo[λ, set, t]`

The full strain is `hybrid/h[λ] + h_echo[λ, set]`, on the same time axis.
`nl_utils.load_ringdown(event)` and `nl_utils.load_echoes(event)` read these files.
`load_ringdown` also rebuilds the modified `Y_in` on the full extended grid, with the same code
(`build_hybrid`).

`output/plots/`

* `ringdown_1936.png`, `echoes_1936.png` – the test run (`plot_test_run.py`).
* `jump_1936_1424.png` – diagnostic of the legacy replace method (see above). It was made with a
  one-off script that is not part of the repository.

## Conventions

* Units: times and frequencies are in units of the total mass `M` of the simulation, G = c = 1.
  All frequencies are angular. `t = 0` is the peak of |h_22|. The barrier tables are in units of
  the remnant mass and are converted (see [Units](#units)); the Kerr QNMs, T_H and Ω_H are
  converted with M_f as well.
* Fourier transforms (convention of the original implementation): time → frequency is
  `np.fft.ifft`, frequency → time is `np.fft.fft`, both with `norm="ortho"`. A signal `e^{-iωt}`
  with ω > 0 sits at positive frequency, and **a time delay Δ multiplies the spectrum by `e^{+iωΔ}`**.
* QNMs: time dependence `e^{-iωt}` with Im ω < 0 and damping time τ = −1/Im ω. A deformation acts as
  `Re ω → Re ω (1+δω)` and `Im ω → Im ω /(1+δτ)`. The mirror modes `−ω*` are deformed in the same way.

## Origin of the echo machinery

These parts follow the original implementation of [D25] App. A. Its content, written out here:

* **Data preparation** (`fft_for_waveforms_CCE`):
  * keep `−9000 M < t < t_end`;
  * resample with cubic splines to `dt = max(min spacing, 0.03 M)`;
  * apply a Tukey window with α = 0.08;
  * spectrum = `np.fft.ifft(·, norm="ortho")`;
  * `freq[0]` is then set to 1e-4 to avoid divisions by zero.
* **Ingoing field** ([D25] Eqs. A1–A2, A9): `Y_in(t) = FFT[low_pass(Ψ0(ω) C_in(ω), |ω| ≤ 1)]`.
* **Barrier coefficients** (`coefficients/*.h5`). They were computed with the Black Hole
  Perturbation Toolkit (bhptoolkit.org, Mathematica) for the ℓ = 2 Schwarzschild potential, as
  functions of Mω.
  * Each file holds `['frequency', 'inc', 'ref']`, the last two as (Re, Im) pairs.
  * The original code assigns `_in = 'ref'` and `_out = 'inc'`; this code does the same.
  * The table contains four duplicated frequencies around |Mω| ≈ 0.2, which are removed before the
    real and imaginary parts are spline-interpolated onto the FFT grid, separately for negative
    and positive frequencies (`interpolate_coeff`).
* **Teukolsky–Starobinsky constants** ([D25] Eq. A3, ℓ = 2): `C = 24 + 12iω` and
  `D = 64iω(128ω² + 8)(1 − 2iω)` (`c_coeff`, `d_coeff`).
* **Barrier reflectivity**: `R_BH = −(D_in/D_out) D(ω)/(4 C(ω))`.
* **Ringdown start** `v_Σ` ([D25] around Eq. A5): the start time in `linspace(−20, 10, 121)` that
  minimises the mismatch `1 − ⟨a,b⟩/sqrt(⟨a,a⟩⟨b,b⟩)`, where `⟨a,b⟩ = Re ∫ a b* dt` (Simpson) over
  `t < 90 M`, between `Y_in` and its QNM fit.
  * The original fitted n ≤ 5 overtones plus their mirror modes and a constant, with
    `(χ_f, M_f)` free in a Nelder–Mead minimisation (bounds 0 ≤ χ ≤ 1, 0 ≤ M ≤ 1.1).
  * The remnant was read from the SXS catalogue as `(|χ_f|, M_f)`.
* **Echo seed** ([D25] Eqs. A6–A7): `Y_in × F(v)`, where `F` is the Planck taper
  * 0 for `v < v_Σ − Δv`;
  * `1/(e^z + 1)` with `z = Δv/(v − v_Σ) + Δv/(v − v_Σ + Δv)` in between;
  * 1 for `v > v_Σ`;
  * with Δv = 8 M.

  The seed value at `t = 100 M` is held for all later times; the original describes this as a fix
  of a late-time numerical artefact.
* **Transfer function** ([D25] Eq. A11 with ℓ = m = 2 and γ = 4, the ECO value):
  `K = −R/((1 − R R_BH) 4 D_out)`.
  * The `γ = 4` of [D25] is unrelated to the dissipation constant γ of [WOA19, OA23].
  * The original reflectivity was the Boltzmann model of [WOA19] Eq. 13 for χ = 0,
    `R = exp(−iω ln(γ_β|ω|)/(π T)) exp(−|ω|/(2T))`. `R_wall` is the same model with the
    co-rotating frequency ω̃ and the constant natural-log delay t_echo (γ_β → κ_echo).
* **Strain** ([D25] Eq. A10): `h_echo = −4 FFT[high_pass(low_pass(K Y_seed/ω², |ω| ≤ 10), |ω| ≥ 7·10⁻⁵)]`.
  The filters and the factor −4 are those of the original implementation.


## Barrier coefficients for spinning remnants

For each event the barrier is that of a Kerr BH with the remnant spin χ_f (signed z-component
from `remnant_metadata.json`, the axis of the m = 2 mode). `--schwarzschild-coefficients` restores
the χ = 0 barrier of [D25] for every event.

Additional reference:

    [X21] S. Xin et al., "Gravitational-wave echoes from spinning exotic compact objects:
          numerical waveforms from the Teukolsky equation", PRD 104, 104005 (2021),
          arXiv:2105.12313 (ref. [88] of [D25]).

[M22] and [D25] treat only non-spinning remnants ([D25] Sec. III: the reconstruction of [M22] "is
adapted to non-spinning remnants"). [X21] gives the Kerr form of the same quantities.

### Tables

* **What is computed.** The "Up" amplitudes of the Teukolsky radial function of the
  BH Perturbation Toolkit (Teukolsky package, Mathematica), ℓ = m = 2, in units of the remnant
  mass, plus the spheroidal eigenvalue:
  * `DCoeff.h5`: `TeukolskyRadial[-2, 2, 2, χ, ω]["Up"]["Amplitudes"]`;
  * `CCoeff.h5`: `TeukolskyRadial[+2, 2, 2, χ, ω]["Up"]["Amplitudes"]`;
  * `inc` = `"Incidence"`, `ref` = `"Reflection"`; same file layout as the χ = 0 tables;
  * `Eigenvalue.h5`: `SpinWeightedSpheroidalEigenvalue[-2, 2, 2, χ ω]`.

  This is the chapter "Same Coefficients for the Teukolsky Equation" of
  `coefficients/Coeff_QG_Mem.nb`. `coefficients/compute_coefficients.wl` is that chapter as a
  script with the spin as a parameter (the notebook itself has the spin hard-coded and is
  interactive). Differences to the notebook:
  * The notebook gets ω < 0 from ω > 0 by complex conjugation. That uses the χ = 0 symmetry; for
    χ ≠ 0 and fixed m = 2 it does not hold, so the script computes negative frequencies directly.
  * Both amplitudes come from one `TeukolskyRadial` call per frequency, on parallel kernels.
  * It also writes `Eigenvalue.h5`.
* **When.** `nl_utils.load_bh_coefficients(χ)` looks for `coefficients/chi_<χ>/` (χ rounded to
  4 decimals) and runs the script if the tables are missing (~12 min on 8 kernels; only the
  eigenvalues, a few seconds, if just `Eigenvalue.h5` is missing). The result is kept, so this
  happens once per remnant. Requirements: Mathematica with the BH Perturbation Toolkit;
  `wolframscript` is looked up in `$NL_WOLFRAMSCRIPT`, the `PATH` and `/Applications/Mathematica.app`.
  Mathematica sometimes reports an error while shutting down, after the tables are written; the
  loader then checks the files and keeps them.
* **Frequency grid.** The grid of the χ = 0 tables (4798 values, |Mω| ≤ 3.96) is reused unchanged,
  because `interpolate_coeff` relies on its layout (the four duplicated frequencies). The loader
  checks that the grids agree. At χ = 0 the script reproduces the [D25] tables (relative
  difference ≤ 4·10⁻¹⁰).

### Barrier reflectivity

Only `R_BH` contains Teukolsky–Starobinsky (TS) constants; `Y_in = Ψ0 C_in` and the transfer
function `K = −R_wall/((1 − R_wall R_BH) 4 D_out)` keep their form. In particular the factor
`(−1)^{ℓ+m+1}/4` of the wall boundary condition is the same for Kerr ([X21] Eq. 54, [M22] Eq. 24).

    R_BH = −(D_in/D_out) 𝒟/(4 C)                         ([M22] Eq. 28, [X21] Eq. 60)
    𝒟    = 64 (2Mr₊)⁴ i p (p² + 4ε²)(−ip + 4ε)            ([X21] Eq. 52)
    p    = ω − mΩ₊,  Ω₊ = a/(2Mr₊),  ε = √(M² − a²)/(4Mr₊)  ([X21] Eqs. 6, 49)
    |C|² = (Q² + 4aωm − 4a²ω²)[(Q − 2)² + 36aωm − 36a²ω²]
           + (2Q − 1)(96a²ω² − 48aωm) + 144ω²(M² − a²),
    Im C = 12Mω,  Re C = +√(|C|² − (Im C)²)                ([X21] Eq. 53)

with M = 1 (remnant mass), a = χ, and `Q = eigenvalue + 2` (the BHPT eigenvalue already contains
a²ω² − 2amω). For χ = 0 these are exactly `c_coeff` = 24 + 12iω and `d_coeff` of [D25] Eq. A3
(`starobinsky_constant`, `horizon_ts_constant`, `barrier_reflectivity_table` in `nl_utils`).

`D_in` has a simple pole at p = 0 (|D_in| changes by a factor 20 between neighbouring grid
points there), and 𝒟 has a zero there. `R_BH` is therefore built on the table grid, where the
product is regular, and only then interpolated. Interpolating `D_in` itself (as in the χ = 0
code) would put a spike into `R_BH` (|R_BH| up to 350 for χ = 0.06).

### Units

The tables are in units of the remnant mass M_f (`TeukolskyRadial` sets the BH mass to 1; the pole
of `D_in` sits at Mω = mχ/(2r₊) with M = 1), while the code works in units of the total mass M.
For the Kerr tables the code therefore (`table_on_grid`)
* evaluates the tables at M_f ω, and
* converts the dimensionful amplitudes to units of M, with `M_f^d` for dimension length^d. From
  the asymptotic forms of the "Up" solutions ([M22] Eq. 5): C_in, D_in ~ length⁻¹,
  D_out ~ length³; R_BH is dimensionless.

The [D25] tables (`--schwarzschild-coefficients`) are used as in the original implementation, at
Mω and without conversion, so that the earlier results stay reproducible.

Checks:
* **Units.** Combining the TS relations, `−C_in/(4 D_out) = R_BH 4ω⁴/C*` must hold in units of M,
  with the physical ω in ω⁴ — as the CCE data confirm (see below: the factor 4 is the same for
  1936 and 1155 only with the physical ω). For 1155 it holds to 1.5·10⁻³ (interpolation) with the
  conversion above; without any conversion (the earlier code) it is off by 18%, with other
  exponents by 5–14%. On the FFT grid of 1155, |R_BH| > 1 exactly for 0 < ω < mΩ_H = 0.4176/M,
  i.e. the superradiant edge of the barrier now coincides with ω̃ = 0 of `R_wall`.
* **TS identity.** The two TS relations imply `C_in |C|² = 4ω⁴ 𝒟 D_in` ([M22] Eq. 11). With the
  Teukolsky tables it holds to 10⁻¹² for χ = −0.09, 0.31, 0.69, Mω between −3.5 and 3.5, and
  p = ±0.003. With Q = eigenvalue + 2 + a²ω² − 2amω (i.e. the formula read literally with the BHPT
  eigenvalue) it is off by up to a factor 3.6.
* **χ → 0.** Fed the χ = 0 tables, the Kerr code path gives the same `R_BH` as the original code
  to 4·10⁻⁸.
* **Superradiance.** For χ = 0.69 (1155), on the table grid: |R_BH| > 1 exactly for 0 < M_f ω < M_f mΩ₊ = 0.397 (at most
  1.0011) and ≤ 1 elsewhere; nothing in the construction enforces this.
* **TS relation at infinity, CCE data** (as [M22] Fig. 6, on the window −400 M … 250 M):
  `Ψ4 = 4 × 4ω⁴ Ψ0 / C*` with relative residual 6% (1936) and 3% (1155) for the Schwarzschild form
  of C. The factor 4 is a normalisation of the CCE Ψ0 and, with the sign of `Ψ4 = −ḧ`, explains
  the factor −4 of the strain conversion. For 1155 the Kerr C fits the data worse (10%): the NR
  modes are spherical-harmonic modes, while the Kerr C belongs to spheroidal modes (see below).
  R_BH uses the Kerr C, since it relates Teukolsky amplitudes, for which the identity above holds.
* **End to end** (λ_NH = −1, 0, 1; with the units above; first echo set α_echo = 0.4, κ_echo = 1;
  the current default α_echo = 1, κ_echo = 4390 gives the same numbers to the digits shown:
  1936 0.0118 and 2.4%, 1155 0.35 and 9%):
  * 1936 (χ = 0.02): the first echo changes by 2% with respect to the χ = 0 barrier (0.0118 vs.
    0.0116), the effect of λ_NH = +1 on it is 2.4% (2.2% with the χ = 0 barrier). The GR check
    gives 0.
  * 1155 (χ = 0.69): the GR check gives 0; the first echo is 0.34, against 0.12 with the χ = 0
    barrier and a ringdown peak of 0.40; the effect of λ_NH = +1 on it is 9% (7%). This follows from the model: combining the two TS
    relations, the first echo is `R_wall R_BH` times the original signal ([M22] Eq. 40), and near
    Mω ≈ 0.42 the Kerr barrier reflects almost completely, `R_wall` is maximal (ω = mΩ_H), and the
    merger of this binary has most of its power.
  * `--schwarzschild-coefficients` reproduces the earlier output files of 1936 (to 10⁻¹⁴, i.e.
    round-off of the installed scipy version).

### Remaining approximations for spinning remnants

* **Spherical vs. spheroidal modes.** The NR Ψ0 (2,2) mode is a spherical-harmonic mode, the
  Teukolsky amplitudes belong to the spheroidal (2,2) mode; the mixing (order aω) is neglected, as
  in [X21]. Like [M22] Eq. 12, the TS relations are used for one (ℓ, m, ω); this needs the
  reflection symmetry of non-precessing binaries.
* **Nearly lossless cavity.** For co-rotating frequencies near mΩ_H a spinning barrier reflects
  almost completely (χ = 0.69: |R_BH| = 0.9998 at Mω = 0.40, against 0.56 for χ = 0), and this is
  where `R_wall` is largest: the edge of the superradiant band (|R_BH| = 1) coincides with the
  maximum of `R_wall` (ω̃ = 0, |R_wall| = 1). The round trip `|R_wall R_BH|` therefore comes
  close to 1 there without exceeding it (0.99999 on a fine grid, 0.997–1.000 on the FFT grids of
  1155), as in [WOA19] Sec. V, Fig. 8: the Boltzmann factor suppresses the ergoregion instability,
  but only barely at ω̃ ≈ 0. The pole of the transfer function lies close to the real axis.
* **Slowly decaying echoes.** For 1155 (default echo parameters, λ_NH = +1) the echo peaks are
  0.35, 0.17, 0.11, 0.080, 0.063, 0.052, 0.044, 0.038, 0.033, i.e. they decay as 1/k, as described
  by [WOA19] (Sec. IV B: 1/t for the first ~20 echoes, then exponential). Between the echoes a
  narrow-band ringing at ω̃ ≈ 0 persists. For the χ = 0.02 remnant 1936 the peaks fall from
  0.012 to 1.1·10⁻³ and 2.9·10⁻⁴.
* **FFT wrap-around.** The echo train that has not decayed by the end of the grid wraps around in
  the FFT. `compute_echoes.py` reports the echo before the ringdown relative to its maximum and
  warns above 3·10⁻³; it also warns if `|R_wall R_BH|` ≥ 1 on the FFT grid.
  * 1155: 6.4·10⁻² (`--t-end 3000`), 5.3·10⁻² (6000), 4.2·10⁻² (9000). A longer grid hardly
    helps, because of the 1/k decay. The first echo changes by 1% between 3000 and 9000.
  * 1936: 1.0·10⁻³ (3000), 4.7·10⁻⁴ (6000).

## Checks

The numbers in this section, and the files in `output/`, were produced with the χ = 0 tables
(before the Kerr tables were introduced; today `--schwarzschild-coefficients`) and with the former
echo parameter sets α_echo = 0.4 and 0.7 (κ_echo = 1, no amplitude factor; plus a set with the
former amplitude factor γ_echo = 0.5, which no longer exists). The echo files in `output/` therefore
cannot be read by `nl_utils.load_echoes` any more; rerun `compute_echoes.py`.

* **Coefficient assignment.** With `_in = 'ref'`, `_out = 'inc'`, `|R_BH|²` agrees with an
  independent Regge–Wheeler integration of the Schwarzschild ℓ = 2 barrier to four digits
  (0.310 at Mω = 0.4, 0.0155 at Mω = 0.5). The opposite assignment gives |R_BH| > 1.
* **GR limit.** For `λ_NH = 0` the first echo equals the echo of the original `Y_in`:
  * exactly (difference 0) for 9 events;
  * 7·10⁻⁴ for 0206, whose continuation starts at 86 M, before the seed is held.
* **Echo timing.** The first echo arrives at `t_echo` (Eq. 10): 320 M (α = 0.4) and 561 M (α = 0.7)
  for 1936 (κ_echo = 1); with the current default (α = 1, κ_echo = 4390) 834 M.
* **Wall reflectivity.** With α_echo = 0.4, κ_echo = 1 the current `R_wall` reproduces the former
  set (α = 0.4, γ_echo = 1) to 5·10⁻¹⁶ (1936).
* **Event 1936 (test run).** `Λ_log = 204.9` for M = 10⁶ M☉, so `Λ_log^{-1/2} = 0.070`: λ_NH = ±1
  means a ±7% shift in ω and τ.
  * Amplitude fits from 10 M: mismatch 1.7·10⁻⁶ (h), 4.4·10⁻³ (Y_in).
  * Fundamental-mode amplitude spread: 2% (h), 15% (Y_in).
  * max|h_echo| = 1.2·10⁻².
* **All events**, residual method, λ_NH = +1 (echo set α = 0.4, κ_echo = 1):

  | event | χ_f | t0 [M] | max\|Δh\|/peak | first-echo change | spread n=0 (h / Y_in) |
  |---|---|---|---|---|---|
  | 0205 | 0.06 | −2.75 | 0.12 | 3.4% | 0.01 / 0.05 |
  | 0206 | 0.06 | 10.0 | 0.31 | 16% | 0.12 / 0.41 |
  | 0334 | 0.31 | 0.25 | 0.25 | 3.1% | 0.07 / 0.51 |
  | 1155 | 0.69 | 2.75 | 0.41 | 7.0% | 0.12 / 1.20 |
  | 1424 | −0.09 | −8.5 | 0.14 | 1.9% | 0.00 / 0.13 |
  | 1448 | 0.01 | −1.5 | 0.18 | 4.1% | 0.04 / 0.11 |
  | 1449 | 0.01 | −7.0 | 0.11 | 2.0% | 0.00 / 0.01 |
  | 1455 | 0.02 | −2.5 | 0.16 | 3.5% | 0.03 / 0.09 |
  | 1936 | 0.02 | −5.5 | 0.15 | 2.2% | 0.02 / 0.15 |
  | 2108 | 0.29 | −7.75 | 0.14 | 1.2% | 0.01 / 0.24 |

## Known limitations

* **The switch-on time `t_a`** is the main systematic of the modified ringdown (table above).
* **`Y_in` is not a clean QNM ringdown.** For t ≳ 60 M it is dominated by a slowly growing
  non-QNM component, probably an artefact of the brick-wall low pass and the finite FFT window.
  Its QNM amplitudes are therefore less stable than those of h: spread 1–24% for the low- and
  moderate-spin events, more for 0206, 0334 and 1155, as flagged by the warnings. The echo seed
  is held constant after 100 M, so the component enters the echoes only weakly.
* **Event 0206** behaves differently from the other low-spin events:
  * the t0 scan has its minimum at the edge of the interval (10 M);
  * the amplitudes are less stable (h 12%, Y_in 41%);
  * the data are short enough that the continuation starts inside the fit window.
* **Spinning remnants.** The barrier is that of the Kerr remnant, but the spherical/spheroidal
  mode mixing is neglected, and near mΩ_H the cavity is nearly lossless (slowly decaying echoes,
  FFT wrap-around); see
  [Remaining approximations for spinning remnants](#remaining-approximations-for-spinning-remnants).
* The NR strain flattens at |h − h_const| ~ 10⁻⁶–10⁻⁴ after ~90 M (numerical floor / non-QNM
  content). The residual method keeps it, because it keeps the data.
* The output files are large (~30 MB per event). `--t-save-min` controls how much of the modified
  waveforms is stored.
