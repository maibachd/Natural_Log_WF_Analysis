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
   reflectivity is the natural-log quantum-horizon wall (Eq. 16):

   ```
   K(ω)    = -R_wall / ((1 - R_wall R_BH) 4 D_out)                              ([D25] Eq. A11)
   R_wall  = γ_echo exp(-|ω̃|/(2 T_H)) exp(+i(ω̃ t_echo + φ_echo)),   ω̃ = ω - m Ω_H
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


## Files

| file | purpose |
|---|---|
| `nl_utils.py` | All shared functions: data loading, FFT and coefficient helpers, Kerr QNMs, QNM fit and stability test, t0 scan, natural-log deformation, both modified-ringdown constructions, Kerr thermodynamics (T_H, Ω_H, S_BH, Λ_log), wall and barrier reflectivities, transfer function, echo strain, I/O helpers. Each function is marked as copied (`[orig]`), adapted (`[orig, adapted]`, with what changed and why), or new (unmarked). |
| `compute_modified_ringdown.py` | **Step 1.** Steps 1–4 of the idea for each event. Writes `output/ringdown/ringdown_<event>.h5`. |
| `compute_echoes.py` | **Step 2.** Step 5 for every deformed ringdown and every echo-parameter set. Also checks the `λ_NH = 0` echo against the echo of the original `Y_in`. Writes `output/echoes/echoes_<event>.h5`. |
| `plot_test_run.py` | **Step 3.** Two test figures for one event: (1) original vs. deformed ringdowns, (2) full time series with echoes for several echo parameters, plus the first echo for the different ringdown deformations. Writes `output/plots/`. |
| `fetch_remnant_metadata.py` | Regenerates `remnant_metadata.json` from the SXS catalogue. Only needed for new events; requires `sxs` and internet. |
| `remnant_metadata.json` | Remnant mass and spin, plus the initial binary parameters, of the 10 events (SXS catalogue v3.0.0). |
| `coefficients/CCoeff.h5`, `coefficients/DCoeff.h5` | Tabulated barrier coefficients `C_in/out`, `D_in/out` of the ℓ = 2 Schwarzschild potential, used in [D25]. See [Origin of the echo machinery](#origin-of-the-echo-machinery). |
| `<event>/{Strain,Psi0,Psi4}.h5` | CCE data per event (modes `Y_l{l}_m{m}.dat`: time, Re, Im). Folders are named `205`, `1936`, …; the code accepts `205` and `0205`. |

## Running

```bash
pip install -r requirements.txt          # qnm is the only package not in a standard scientific stack

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
* `compute_echoes.py`
  * `--echo alpha_echo=0.4,gamma_echo=1,phi_echo=0,kappa_echo=1,zeta=1` – repeatable, one per echo-parameter set.
  * Defaults: (α, γ) = (0.4, 1), (0.7, 1), (0.4, 0.5).
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
  All frequencies are angular. `t = 0` is the peak of |h_22|.
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
  * The `γ = 4` of [D25] is unrelated to `γ_echo` of the PDF, which sits inside `R_wall`.
  * The original reflectivity was the Boltzmann model
    `R = exp(−iω ln(γ_β|ω|)/(π T)) exp(−|ω|/(2T))`.
* **Strain** ([D25] Eq. A10): `h_echo = −4 FFT[high_pass(low_pass(K Y_seed/ω², |ω| ≤ 10), |ω| ≥ 7·10⁻⁵)]`.
  The filters and the factor −4 are those of the original implementation.


## Checks

* **Coefficient assignment.** With `_in = 'ref'`, `_out = 'inc'`, `|R_BH|²` agrees with an
  independent Regge–Wheeler integration of the Schwarzschild ℓ = 2 barrier to four digits
  (0.310 at Mω = 0.4, 0.0155 at Mω = 0.5). The opposite assignment gives |R_BH| > 1.
* **GR limit.** For `λ_NH = 0` the first echo equals the echo of the original `Y_in`:
  * exactly (difference 0) for 9 events;
  * 7·10⁻⁴ for 0206, whose continuation starts at 86 M, before the seed is held.
* **Echo timing.** The first echo arrives at `t_echo` (Eq. 10): 320 M (α = 0.4) and 561 M (α = 0.7)
  for 1936.
* **Event 1936 (test run).** `Λ_log = 204.9` for M = 10⁶ M☉, so `Λ_log^{-1/2} = 0.070`: λ_NH = ±1
  means a ±7% shift in ω and τ.
  * Amplitude fits from 10 M: mismatch 1.7·10⁻⁶ (h), 4.4·10⁻³ (Y_in).
  * Fundamental-mode amplitude spread: 2% (h), 15% (Y_in).
  * max|h_echo| = 1.2·10⁻².
* **All events**, residual method, λ_NH = +1 (echo set α = 0.4, γ = 1):

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
* **Spinning remnants (1155, 0334, 2108)** are only approximately described, because of the
  Schwarzschild barrier (decision 6).
* The NR strain flattens at |h − h_const| ~ 10⁻⁶–10⁻⁴ after ~90 M (numerical floor / non-QNM
  content). The residual method keeps it, because it keeps the data.
* The output files are large (~30 MB per event). `--t-save-min` controls how much of the modified
  waveforms is stored.
