"""
Utilities for the Natural Log Project echo pipeline.

This module collects everything that is shared between

    compute_modified_ringdown.py   (natural-log deformed ringdown)
    compute_echoes.py              (echoes seeded by the deformed ringdown)
    plot_test_run.py               (test plots)

Equation numbers "Eq. (n)" refer to The_Natural_Log_Project.pdf (draft v0.1).

Provenance
----------
The echo machinery (FFT of the CCE data, ingoing field at the horizon, barrier
coefficients, Planck taper, ringdown-start scan, transfer function, conversion
to strain) re-implements the echo computation of

    [D25] N. Deppe et al., "Echoes from beyond: detecting gravitational-wave
          quantum imprints with LISA", PRD 111, 124035 (2025),
          arXiv:2411.05645, Appendix A (= ref. [20] of the PDF),

which in turn follows

    [M22] S. Ma et al., "Gravitational-wave echoes from numerical-relativity
          waveforms via spacetime construction near merging compact objects",
          PRD 105, 104007 (2022).

Functions marked [orig] are taken unchanged from the original implementation of
that computation (a notebook plus helper module that is NOT part of this
repository); everything they do is written out here. Functions marked
[orig, adapted] say what was changed and why. Unmarked functions are new.

The construction of the modified ringdown (section "Modified ringdown" below) is
new; its references are given there.

Units and conventions
---------------------
All times/frequencies are in units of the total (initial) mass M of the NR
simulation, G = c = 1; t = 0 is the peak of |h_22|. Angular frequencies are used
throughout (omega = 2 pi f). Fourier convention (that of the original
implementation): time -> frequency is np.fft.ifft, frequency -> time is
np.fft.fft, both with norm="ortho". In this convention a signal ~ exp(-i omega t)
with omega > 0 appears at positive frequency and a time delay Delta multiplies
the spectrum by exp(+i omega Delta).
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import qnm
from astropy import constants as const
from astropy import units as u
from scipy import signal
from scipy.integrate import simpson
from scipy.interpolate import BSpline, splrep

##############################################################################
# Paths and event bookkeeping
##############################################################################

PROJECT_DIR = Path(__file__).resolve().parent
COEFF_DIR = PROJECT_DIR / "coefficients"          # CCoeff.h5, DCoeff.h5 (see BHCoefficients)
REMNANT_FILE = PROJECT_DIR / "remnant_metadata.json"
OUTPUT_DIR = PROJECT_DIR / "output"
RINGDOWN_DIR = OUTPUT_DIR / "ringdown"
ECHO_DIR = OUTPUT_DIR / "echoes"
PLOT_DIR = OUTPUT_DIR / "plots"

DEFAULT_EVENTS = ["0205", "0206", "0334", "1155", "1424", "1448", "1449", "1455", "1936", "2108"]

# The C/D coefficient tables (and the Teukolsky-Starobinsky constants c_coeff,
# d_coeff) describe a Schwarzschild barrier. Above this remnant spin we warn.
SCHWARZSCHILD_SPIN_WARNING = 0.1


def normalize_event_id(event):
    """'205', 205, '0205' -> '0205' (4-digit SXS BBH number)."""
    return str(event).strip().zfill(4)


def event_dir(event):
    """Folder of an event. Folders exist both as '0205' and '205'."""
    event = normalize_event_id(event)
    for name in (event, event.lstrip("0")):
        path = PROJECT_DIR / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"no data folder for event {event} in {PROJECT_DIR}")


##############################################################################
# Loading data
##############################################################################

def load_event_mode(event, quantity, ell=2, m=2):
    """
    Load one (ell, m) mode of 'Strain', 'Psi0' or 'Psi4' of an event.

    The files contain one dataset per mode ('Y_l2_m2.dat') with columns
    (time, real part, imaginary part).
    Some files contain (near-)duplicate time stamps (spacing ~1e-10), which
    break the spline interpolation; those samples are dropped.
    """
    with h5py.File(event_dir(event) / f"{quantity}.h5", "r") as f:
        data = f[f"Y_l{ell}_m{m}.dat"][()]
    time = data[:, 0]
    values = data[:, 1] + 1j * data[:, 2]
    keep = np.concatenate(([True], np.diff(time) > 1e-6))
    return time[keep], values[keep]


def load_event(event, ell=2, m=2):
    """
    Load strain h and Psi0 of an event, with time measured from the peak of |h|
    (as in the original implementation: t = t_NR - t_peak).
    """
    t, h = load_event_mode(event, "Strain", ell, m)
    t_psi0, psi0 = load_event_mode(event, "Psi0", ell, m)
    if len(t_psi0) != len(t) or not np.allclose(t_psi0, t):
        raise ValueError(f"time axes of Strain and Psi0 differ for event {event}")
    peak = t[np.argmax(np.abs(h))]
    return dict(time=t - peak, h=h, psi0=psi0, t_peak=peak)


def load_remnant(event):
    """Remnant mass M_f (units of M) and spin of an event (SXS metadata, see fetch_remnant_metadata.py)."""
    event = normalize_event_id(event)
    data = json.loads(REMNANT_FILE.read_text())
    if event not in data:
        raise KeyError(f"event {event} not in {REMNANT_FILE}; run fetch_remnant_metadata.py {event}")
    entry = data[event]
    chi_vec = np.array(entry["remnant_dimensionless_spin"])
    # The modes are decomposed w.r.t. the z axis (orbital angular momentum). For
    # the non-precessing systems used here chi_f is along z; we keep its sign.
    if np.linalg.norm(chi_vec[:2]) > 1e-3:
        warnings.warn(f"event {event}: remnant spin has an in-plane component {chi_vec}")
    return dict(mass=entry["remnant_mass"], chi=float(chi_vec[2]), sxs_id=entry["sxs_id"])


@dataclass
class BHCoefficients:
    """
    Tabulated coefficients of the l = 2 Schwarzschild potential barrier, as a
    function of the angular frequency M omega (in units of the remnant mass).
    They were computed with the Black Hole Perturbation Toolkit
    (bhptoolkit.org, Mathematica) for [D25]; see [D25] Eqs. (A1)-(A3) and
    [M22] for their definition. In this code they enter as
      * C_in:  Y_in(omega) = Psi0(omega) C_in(omega)   (ingoing field at the horizon, ingoing_field)
      * D_in, D_out: barrier reflectivity R_BH = -(D_in/D_out) d(omega)/(4 c(omega))
                     and transmission factor 1/D_out of the transfer function.
    C_out is read for completeness but not used.
    """
    freq_C: np.ndarray
    C_in: np.ndarray
    C_out: np.ndarray
    freq_D: np.ndarray
    D_in: np.ndarray
    D_out: np.ndarray


def _read_coeff_file(filename):
    # [orig] Each file holds the datasets ['frequency', 'inc', 'ref'] (in this
    # order); 'inc' and 'ref' are arrays of (Re, Im) pairs. The original
    # implementation assigns "_in" = 'ref' (keys[2]) and "_out" = 'inc'
    # (keys[1]); we keep this. With this assignment |R_BH|^2 agrees with an
    # independent Regge-Wheeler integration of the l = 2 barrier to four digits
    # (see README, "Checks").
    with h5py.File(filename, "r") as f:
        keys = list(f.keys())
        freq = np.array(list(f[keys[0]]))
        c_in = np.array([re + 1j * im for re, im in list(f[keys[2]])])
        c_out = np.array([re + 1j * im for re, im in list(f[keys[1]])])
    return freq, c_in, c_out


def load_bh_coefficients():
    freq_C, C_in, C_out = _read_coeff_file(COEFF_DIR / "CCoeff.h5")
    freq_D, D_in, D_out = _read_coeff_file(COEFF_DIR / "DCoeff.h5")
    return BHCoefficients(freq_C, C_in, C_out, freq_D, D_in, D_out)


##############################################################################
# Time / frequency domain helpers
##############################################################################

def resample_uniform(time, waveform, t_uniform):
    """Cubic spline resampling of a complex waveform (same spline as fft_for_waveforms_CCE)."""
    tck_r = splrep(time, np.real(waveform), s=0)
    tck_i = splrep(time, np.imag(waveform), s=0)
    return BSpline(*tck_r)(t_uniform) + 1J * BSpline(*tck_i)(t_uniform)


def fft_for_waveforms_CCE(time, waveform, cut_time_start=-9000, cut_time_end=1200, tukey_alpha=0.08):
    """
    [orig, adapted] FFT of a CCE waveform on a uniform grid:
      * keep cut_time_start < t < cut_time_end (the start cut removes junk
        radiation at the beginning of the simulation);
      * resample with a cubic spline onto a uniform grid with
        dt = max(smallest sample spacing, 0.03 M);
      * multiply by a Tukey window (alpha = 0.08; leaves inspiral and merger
        untouched, tapers both ends);
      * spectrum = np.fft.ifft(..., norm="ortho"), angular frequencies
        2 pi fftfreq(N, dt).

    Changes w.r.t. the original:
      * cut_time_end is clamped to the last data sample. The original always
        built the grid up to t = 1200 M, which for the per-event data (which
        ends 430-890 M after the peak) means extrapolating the cubic spline.
      * The number of samples is made even, which interpolate_coeff assumes
        (it splits the fftfreq-ordered grid at N//2).

    Returns (angular frequencies, spectrum, uniform time grid).
    """
    cut_time_end = min(cut_time_end, time[-1])

    waveform_time = time[time > cut_time_start]
    waveform = waveform[time > cut_time_start]
    waveform = waveform[waveform_time < cut_time_end]
    waveform_time = waveform_time[waveform_time < cut_time_end]

    if np.min(np.diff(waveform_time)) > 0.03:
        dt = np.min(np.diff(waveform_time))
    else:
        dt = 0.03
    t_uniform = np.arange(waveform_time[0], cut_time_end, dt)
    t_uniform = t_uniform[: 2 * (len(t_uniform) // 2)]

    h = resample_uniform(waveform_time, waveform, t_uniform)

    # Tukey window: leaves inspiral and merger unchanged, tapers both ends
    window = signal.windows.tukey(h.shape[0], alpha=tukey_alpha, sym=True)
    h = h * window

    fft_data = np.fft.ifft(h, norm="ortho")
    fft_freq = np.fft.fftfreq(t_uniform.shape[0], dt) * 2 * np.pi
    return fft_freq, fft_data, t_uniform


def tukey_taper_start(t_uniform, tukey_alpha=0.08):
    """First time at which the Tukey window of fft_for_waveforms_CCE deviates from 1 at the late end."""
    return t_uniform[-1] - 0.5 * tukey_alpha * (t_uniform[-1] - t_uniform[0])


def interpolate_coeff(f, f_new, coeff):
    """
    [orig] Interpolate a tabulated coefficient onto an FFT frequency grid f_new
    (np.fft.fftfreq ordering, even length).

    The table is ordered from negative to positive frequency and contains four
    duplicated frequencies around |M omega| ~ 0.2 (where two tabulation ranges
    overlap); these entries (indices N/2+1652, N/2+1653, N/2-1653, N/2-1654) are
    removed first. Real and imaginary parts of the negative- and
    positive-frequency halves are then interpolated separately with cubic
    splines and reassembled in fftfreq order.
    """
    N_f = len(f)
    # remove redundance
    f = np.delete(f, [N_f//2+1652, N_f//2+1653, N_f//2-1653, N_f//2-1654])
    coeff = np.delete(coeff, [N_f//2+1652, N_f//2+1653, N_f//2-1653, N_f//2-1654])

    N_new = len(f)
    N = len(f_new)
    # split freq
    freq_n = f[:N_new//2]
    freq_p = f[N_new//2:]

    # split coeff
    coeff_n = coeff[:N_new//2]
    coeff_p = coeff[N_new//2:]

    tck_n_i = splrep(freq_n, np.imag(coeff_n), s=0)
    tck_n_r = splrep(freq_n, np.real(coeff_n), s=0)
    tck_p_i = splrep(freq_p, np.imag(coeff_p), s=0)
    tck_p_r = splrep(freq_p, np.real(coeff_p), s=0)

    int_coeff_n = 1J * BSpline(*tck_n_i)(f_new[N//2:]) + BSpline(*tck_n_r)(f_new[N//2:])
    int_coeff_p = 1J * BSpline(*tck_p_i)(f_new[:N//2]) + BSpline(*tck_p_r)(f_new[:N//2])

    return np.concatenate((int_coeff_p, int_coeff_n))


def c_coeff(freq):
    """
    [orig] Teukolsky-Starobinsky constant C = (l-1) l (l+1) (l+2) + 12 i omega
    for l = 2 (Schwarzschild), [D25] Eq. (A3).
    """
    return 24 + 12 * 1J * freq


def d_coeff(freq):
    """[orig] Teukolsky-Starobinsky constant D = 64 i omega (128 omega^2 + 8)(1 - 2 i omega), [D25] Eq. (A3)."""
    return 64 * 1J * freq * (128 * freq**2 + 8) * (1 - 2 * 1J * freq)


def low_pass(fft, freq, cut):
    """[orig, adapted] brick-wall low pass |omega| <= cut. (Returns a copy instead of modifying fft in place.)"""
    fft = fft.copy()
    fft[np.abs(freq) > cut] = 0
    return fft


def high_pass(fft, freq, cut):
    """[orig, adapted] brick-wall high pass |omega| >= cut. (Returns a copy instead of modifying fft in place.)"""
    fft = fft.copy()
    fft[np.abs(freq) < cut] = 0
    return fft


def planck_taper(v, v_shell, delta_v):
    """
    [orig, adapted] Planck taper ("Planck filter" F(v, Delta v, v_Sigma) of
    [D25] Eq. (A6)): 0 for v < v_shell - delta_v, 1 for v > v_shell, and
    1 / (exp(z) + 1) with z = delta_v/(v - v_shell) + delta_v/(v - v_shell + delta_v)
    in between. In the original, v_shell and delta_v were global variables
    (delta_v = 2/kappa = 8 M with kappa = 1/4); here they are arguments.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        z = delta_v / (v - v_shell) + delta_v / (v - v_shell + delta_v)
        F_v = np.zeros(len(v))
        F_v[v < v_shell] = 1 / (np.exp(z[v < v_shell]) + 1)
    F_v[v < (v_shell - delta_v)] = 0
    F_v[v > v_shell] = 1
    return F_v


##############################################################################
# Ingoing field at the horizon (echo seed)
##############################################################################

def ingoing_field(freq, fft_psi0, coeffs, cut=1e0):
    """
    [orig] Y_22^in(t) = FFT[ low_pass(Psi0(omega) C_in(omega), |omega| <= 1) ]:
    the radiation that falls towards the (would-be) horizon and seeds the
    echoes ([D25] Eqs. (A2), (A9): Y_in is obtained from the asymptotic Psi0 via
    the barrier coefficients). freq[0] must already be set to 1e-4 (as in the
    original) before calling.
    """
    C_lmw_in = interpolate_coeff(coeffs.freq_C, freq, coeffs.C_in)
    return np.fft.fft(low_pass(fft_psi0 * C_lmw_in, freq, cut), norm="ortho")


##############################################################################
# Quasi-normal modes: Kerr spectrum, fit, natural-log deformation
##############################################################################

def kerr_qnm_omegas(chi, mass, n_max, ell=2, m=2):
    """
    Complex Kerr QNM frequencies omega_{l m n} / M_f for n = 0..n_max, in units
    of M (time dependence exp(-i omega t), Im(omega) < 0), from the `qnm`
    package. As in the original implementation: s = 2 and a = |chi| (the
    frequencies are identical for s = +2 and s = -2). The mirror modes
    -conj(omega) are added in qnm_basis, so the sign of chi is covered.
    """
    omegas = []
    for n in range(n_max + 1):
        omega, _, _ = qnm.modes_cache(s=2, l=ell, m=m, n=n)(a=np.abs(chi))
        omegas.append(omega / mass)
    return np.array(omegas)


def qnm_basis(t, omegas, t_ref=0.0):
    """
    Basis of the ringdown model ([D25] Eq. (A5), plus a constant):
        columns = [1, exp(-i w_0 (t-t_ref)), exp(+i w_0^* (t-t_ref)), exp(-i w_1 (t-t_ref)), ...]
    i.e. a constant (memory / offset term), every overtone and its mirror mode.
    With t_ref = 0 this is the basis of the original implementation. A
    reference time t_ref makes the coefficients the amplitudes at t = t_ref.
    """
    tau = t - t_ref
    cols = [np.ones_like(tau, dtype=complex)]
    for omega in omegas:
        cols.append(np.exp(-1J * omega * tau))
        cols.append(np.exp(1J * np.conjugate(omega) * tau))
    return np.array(cols).T


def fit_qnm_amplitudes(t, field, t0, omegas, upper_t=100):
    """
    Linear least-squares fit of the QNM amplitudes for a fixed spectrum (fixed
    remnant) on the window t0 < t < upper_t; the coefficients are referenced to
    t0 (see qnm_basis).
    The original implementation additionally fitted (chi_f, M_f) with a
    Nelder-Mead minimisation of the residual (bounds 0 <= chi <= 1,
    0 <= M <= 1.1) around the linear fit; here the remnant is fixed to the SXS
    values (README, decision 1).
    Returns (coefficients, t_cut, model on t_cut, field on t_cut).
    """
    indices = (t > t0) * (t < upper_t)
    t_cut = t[indices]
    h_cut = field[indices]
    basis = qnm_basis(t_cut, omegas, t_ref=t0)
    coeffs = np.linalg.lstsq(basis, h_cut, rcond=None)[0]
    return coeffs, t_cut, basis @ coeffs, h_cut


def rereference_coeffs(coeffs, omegas, t_from, t_to):
    """
    Coefficients of qnm_basis referenced to t_from -> referenced to t_to
    (same function of time): A_n -> A_n exp(-i w_n (t_to - t_from)),
    B_n -> B_n exp(+i w_n^* (t_to - t_from)); the constant is unchanged.
    """
    out = np.array(coeffs, dtype=complex, copy=True)
    dt = t_to - t_from
    out[1::2] *= np.exp(-1J * omegas * dt)
    out[2::2] *= np.exp(1J * np.conjugate(omegas) * dt)
    return out


def mismatch_ringdown(h_NR, h_ring, t_cut):
    """
    [orig] Mismatch 1 - <h_NR, h_ring> / sqrt(<h_ring, h_ring><h_NR, h_NR>),
    <a, b> = Re int a b^* dt (Simpson), restricted to t_cut < 90 M.
    Used to determine the ringdown start v_Sigma ([D25] around Eq. (A5)).
    """
    T = 90  # M_f (it is supposed to be 90 M_f after the peak [which is at 0])
    times = t_cut

    h_NR = h_NR[times < T]
    h_ring = h_ring[times < T]
    init_time = times[times < T]

    up = simpson(np.real(h_NR * np.conjugate(h_ring)), x=init_time)
    part1 = simpson(np.real(h_ring * np.conjugate(h_ring)), x=init_time)
    part2 = simpson(np.real(h_NR * np.conjugate(h_NR)), x=init_time)
    down = np.sqrt(part1 * part2)
    return 1 - up / down


def scan_ringdown_start(t, field, omegas, t0_grid, upper_t=100):
    """
    Ringdown start time t0 (= v_Sigma of [D25], "Sigma_shell"): the t0 on
    t0_grid that minimises the mismatch between the field and its QNM fit
    ([D25] around Eq. (A5); the original scanned linspace(-20, 10, 121) with
    n = 5 overtones). t0 sets where the Planck taper of the echo seed starts.
    """
    mismatches = []
    for t0 in t0_grid:
        _, t_cut, model, data = fit_qnm_amplitudes(t, field, t0, omegas, upper_t)
        mismatches.append(mismatch_ringdown(data, model, t_cut))
    mismatches = np.array(mismatches)
    return t0_grid[np.argmin(mismatches)], mismatches


def natural_log_deformation(lambda_nh, Lambda_log, p, c_omega, c_tau, n_modes):
    """
    Eq. (14): delta_omega_n = c^(omega)_n lambda_NH Lambda_log^-p,
              delta_tau_n   = c^(tau)_n   lambda_NH Lambda_log^-p.
    c_omega, c_tau: scalar (same for all overtones) or one value per overtone.
    """
    eps = lambda_nh * Lambda_log ** (-p)
    c_omega = np.broadcast_to(np.asarray(c_omega, dtype=float), (n_modes,))
    c_tau = np.broadcast_to(np.asarray(c_tau, dtype=float), (n_modes,))
    return c_omega * eps, c_tau * eps


def deform_qnm_omegas(omegas, delta_omega, delta_tau):
    """
    Eq. (13): omega_n = omega_n^Kerr (1 + delta_omega_n), tau_n = tau_n^Kerr (1 + delta_tau_n),
    with omega the oscillation frequency Re(omega) and tau = -1/Im(omega) the damping time.
    The mirror modes -conj(omega) follow automatically in qnm_basis.
    """
    return np.real(omegas) * (1 + delta_omega) + 1J * np.imag(omegas) / (1 + delta_tau)


##############################################################################
# Modified ringdown
#
# Two constructions are implemented (see README, "Modified ringdown"):
#
#   "residual" (default):  h_mod = h_NR + Delta,
#       Delta(t) = sum_n [A_n(t_a)(e^{-i w'_n tau} - e^{-i w_n tau})
#                        + B_n(t_a)(e^{+i w'_n* tau} - e^{+i w_n* tau})],  tau = t - t_a,
#       switched on smoothly at t_a (default 5 M after the peak). The amplitudes
#       come from a fit with only the QNMs whose amplitudes are stable under
#       a change of the fit start (default n <= 1, fitted from t = 10 M).
#       The NR data are kept (lambda_NH = 0 reproduces them exactly); only the
#       difference between the deformed and the GR QNM content is added.
#       This follows the strategy of the parametrized EOB models, which keep the
#       GR waveform and change the QNM frequencies only in the ringdown part
#       (R. Brito, A. Buonanno, V. Raymond, arXiv:1805.00293;
#        A. Ghosh, R. Brito, A. Buonanno, arXiv:2104.01906), and restricts
#       the deformation to QNMs that are actually resolved in the NR data, as
#       recommended by the overtone-content studies of V. Baibhav et al.,
#       arXiv:2302.03050 (higher overtones fitted near the peak mostly absorb
#       nonlinear / non-modal content) and E. Coleman, E. Finch,
#       arXiv:2512.08098 (amplitudes of high overtones are strongly correlated
#       and poorly determined).
#
#   "replace" (legacy, first version of this pipeline): cut the NR waveform at
#       the scanned ringdown start t0 and replace it by the n <= 3 QNM fit with
#       the deformed frequencies (attach_ringdown). Kept for comparison: the
#       overtone amplitudes of that fit are not stable, and deforming them
#       produces a large spurious change right after t0.
##############################################################################

def qnm_amplitude_stability(t, field, omegas, t_fit, upper_t=100, step=5.0):
    """
    Stability of the fitted QNM amplitudes under a change of the fit start:
    fit from t_fit - step, t_fit and t_fit + step, re-reference all amplitudes
    to the peak (t = 0), and return per mode n the relative spread
    (max - min) / mean of sqrt(|A_n|^2 + |B_n|^2) (mode + mirror mode).
    A physical QNM has a start-independent amplitude; an overfitted one does not.
    """
    amps = []
    for start in (t_fit - step, t_fit, t_fit + step):
        coeffs = rereference_coeffs(fit_qnm_amplitudes(t, field, start, omegas, upper_t)[0], omegas, start, 0.0)
        amps.append(np.sqrt(np.abs(coeffs[1::2])**2 + np.abs(coeffs[2::2])**2))
    amps = np.array(amps)
    return (amps.max(axis=0) - amps.min(axis=0)) / amps.mean(axis=0)


def deformation_delta(t, t_a, coeffs, omegas, omegas_mod, t_ref, switch_width=5.0):
    """
    Delta(t) of the "residual" construction: deformed minus GR QNM content,
        Delta = w(t) [ model(t; omegas_mod) - model(t; omegas) ],
    where both models use the same amplitudes, referenced to t_a (so
    Delta(t_a) = 0), without the constant term, and w is a Planck taper from
    0 at t_a to 1 at t_a + switch_width (switch_width = 0: step at t_a).
    coeffs are the fit coefficients referenced to t_ref. Delta = 0 for t < t_a.
    """
    amps = rereference_coeffs(coeffs, omegas, t_ref, t_a)
    amps[0] = 0  # the constant (memory) term is not a QNM and is not deformed
    delta = np.zeros(len(t), dtype=complex)
    late = t >= t_a
    delta[late] = (qnm_basis(t[late], omegas_mod, t_a) - qnm_basis(t[late], omegas, t_a)) @ amps
    if switch_width > 0:
        delta *= planck_taper(t, t_a + switch_width, switch_width)
    return delta


def continue_with_qnm_model(t, data, coeffs, omegas, t_ref, t_cont, cont_width=50.0):
    """
    Continue a field beyond its trustworthy range with its GR QNM model
    (including the constant term): data for t < t_cont - cont_width, model for
    t > t_cont, Planck-taper blend in between. data may be NaN after t_cont.
    Needed because the NR data end 430-890 M after the peak, while the echoes
    need a longer time window.
    """
    w = planck_taper(t, t_cont, cont_width)
    out = np.array(data, dtype=complex, copy=True)
    use = w > 0
    model = qnm_basis(t[use], omegas, t_ref) @ coeffs
    out[use] = np.where(w[use] < 1, (1 - w[use]) * np.nan_to_num(out[use]) + w[use] * model, model)
    return out


def attach_ringdown(t, data, t0, coeffs, omegas, blend_width=5.0):
    """
    Legacy "replace" construction: original data for t < t0, QNM model (with
    the given, possibly deformed, spectrum; coefficients referenced to t0) for
    t >= t0 + blend_width, and a smooth transition (planck_taper) from data to
    model on [t0, t0 + blend_width]. Because the amplitudes are referenced to
    t0, the model with deformed QNMs equals the undeformed fit at t0; the blend
    removes the remaining jump (the fit residual at the edge of the fit window);
    blend_width = 0 gives a hard cut. data may contain NaN beyond the end of
    the NR data (which must lie after t0 + blend_width).
    """
    hybrid = np.array(data, dtype=complex, copy=True)
    late = t >= t0
    model = qnm_basis(t[late], omegas, t_ref=t0) @ coeffs
    if blend_width > 0:
        w = planck_taper(t[late], t0 + blend_width, blend_width)   # 0 at t0 -> 1 at t0 + blend_width
        mix = w < 1
        model[mix] = (1 - w[mix]) * hybrid[late][mix] + w[mix] * model[mix]
    hybrid[late] = model
    if np.any(~np.isfinite(hybrid)):
        raise ValueError("hybrid waveform undefined (data missing before t0 + blend_width)")
    return hybrid


def build_hybrid(method, t, data, coeffs, omegas, omegas_mod, t_ref, t_attach, switch_width,
                 t_cont=None, cont_width=50.0):
    """
    Modified waveform on the grid t for one deformed spectrum omegas_mod.
      method "residual": continue_with_qnm_model(data) + deformation_delta, switched on at t_attach = t_a.
      method "replace":  attach_ringdown at t_attach = t0 (coefficients must be referenced to t_ref = t0).
    data: field on t, NaN where no NR data exist.
    """
    if method == "residual":
        base = continue_with_qnm_model(t, data, coeffs, omegas, t_ref, t_cont, cont_width)
        return base + deformation_delta(t, t_attach, coeffs, omegas, omegas_mod, t_ref, switch_width)
    if method == "replace":
        if t_ref != t_attach:
            raise ValueError("replace: the coefficients must be referenced to the attachment time t0")
        return attach_ringdown(t, data, t_attach, coeffs, omegas_mod, switch_width)
    raise ValueError(f"unknown method '{method}'")


##############################################################################
# Kerr remnant: horizon, Hawking temperature, entropy, natural-log scale
##############################################################################

def kerr_horizon(mass, chi):
    """Eq. (3): r_pm = M (1 +- sqrt(1 - chi^2)); returns (r_plus, r_minus, a = chi M)."""
    root = np.sqrt(1 - chi**2)
    return mass * (1 + root), mass * (1 - root), chi * mass


def hawking_temperature(mass, chi):
    """Eq. (4): T_H = (r_+ - r_-) / (4 pi (r_+^2 + a^2))  (G = c = hbar = k_B = 1, units of 1/M)."""
    r_p, r_m, a = kerr_horizon(mass, chi)
    return (r_p - r_m) / (4 * np.pi * (r_p**2 + a**2))


def horizon_angular_velocity(mass, chi):
    """Omega_H = a / (2 M r_+), as in omega_tilde = omega - a m / (2 M r_+) below Eq. (16). Sign of chi kept."""
    r_p, _, a = kerr_horizon(mass, chi)
    return a / (2 * mass * r_p)


def bh_entropy(mass_solar, chi):
    """S_BH = A / (4 G hbar) (Eq. 1) of a Kerr BH of physical mass mass_solar [M_sun], A = 4 pi (r_+^2 + a^2)."""
    length = (const.G * const.M_sun / const.c**2).to(u.m).value * mass_solar
    planck_area = (const.hbar * const.G / const.c**3).to(u.m**2).value
    r_p, _, a = kerr_horizon(1.0, chi)
    area = 4 * np.pi * (r_p**2 + a**2) * length**2
    return area / (4 * planck_area)


def lambda_log(mass_solar, chi, zeta=1.0):
    """Eq. (5): Lambda_log = ln(zeta S_BH(M_z, chi))."""
    return np.log(zeta * bh_entropy(mass_solar, chi))


@dataclass
class Remnant:
    """Remnant in code units (M = total NR mass) plus its physical scale."""
    mass: float          # M_f in units of M
    chi: float           # dimensionless spin along z (signed)
    m_total_solar: float # physical (detector-frame) total mass M in M_sun

    @property
    def mass_solar(self):
        return self.mass * self.m_total_solar

    @property
    def T_H(self):
        return hawking_temperature(self.mass, self.chi)

    @property
    def Omega_H(self):
        return horizon_angular_velocity(self.mass, self.chi)

    def Lambda_log(self, zeta=1.0):
        return lambda_log(self.mass_solar, self.chi, zeta)


##############################################################################
# Echoes: wall reflectivity, barrier reflectivity, transfer function
##############################################################################

@dataclass
class EchoParameters:
    """Echo-sector parameters of Table I / Sec. IV."""
    alpha_echo: float = 0.4   # normalisation of the entropy-controlled delay, Eq. (10)
    gamma_echo: float = 1.0   # wall reflectivity amplitude, Eq. (16)
    phi_echo: float = 0.0     # phase offset / cavity phase (Table I)
    kappa_echo: float = 1.0   # model constant in Delta_ell, Eq. (9)
    zeta: float = 1.0         # entropy normalisation, Eq. (5) (cancels in t_echo)
    ell: int = 2
    m: int = 2

    def label(self):
        return f"alpha={self.alpha_echo:g}, gamma={self.gamma_echo:g}, phi={self.phi_echo:g}, kappa={self.kappa_echo:g}"


def echo_delay(remnant, params):
    """
    Eq. (10): t_echo = alpha_echo / (2 pi T_H) (Lambda_log + Delta_ell),
    Delta_ell = ln[kappa_echo / (zeta l(l+1))]  (Eq. 9). Identical to Eq. (15).
    Returned in units of M (T_H in code units, Lambda_log from the physical mass).
    """
    Delta_ell = np.log(params.kappa_echo / (params.zeta * params.ell * (params.ell + 1)))
    return params.alpha_echo / (2 * np.pi * remnant.T_H) * (remnant.Lambda_log(params.zeta) + Delta_ell)


def R_wall(omega, remnant, params, t_echo=None):
    """
    Quantum-horizon reflectivity, Eq. (16):
        R_wall = gamma_echo exp(-|omega_tilde| / (2 T_H)) exp(+i (omega_tilde t_echo + phi_echo)),
        omega_tilde = omega - m Omega_H.
    Two deliberate differences to the formula as printed:
      * |omega_tilde| in the Boltzmann factor. Without the modulus the factor
        grows exponentially for omega_tilde < 0. The Boltzmann reflectivity of
        the original implementation, R = exp(-i omega ln(gamma |omega|)/(pi T))
        exp(-|omega|/(2T)), uses |omega| in the same way.
      * exp(+i omega_tilde t_echo): in this code's Fourier convention a delay
        corresponds to +i omega t (the original reflectivity has the same
        sign). Checked numerically: the first echo arrives t_echo after the ringdown.
    """
    if t_echo is None:
        t_echo = echo_delay(remnant, params)
    omega_tilde = omega - params.m * remnant.Omega_H
    return (params.gamma_echo * np.exp(-np.abs(omega_tilde) / (2 * remnant.T_H))
            * np.exp(1J * (omega_tilde * t_echo + params.phi_echo)))


def R_BH_T(freq, D_lmw_in, D_lmw_out):
    """
    [orig] Reflectivity of the l = 2 Schwarzschild potential barrier,
    R_BH = -(D_in / D_out) d(omega) / (4 c(omega)) ([D25] App. A, [M22]).
    """
    return (-1) * D_lmw_in / D_lmw_out * d_coeff(freq) / (4 * c_coeff(freq))


def transfer_function(R_w, R_bh, D_lmw_out):
    """
    [orig, adapted] Full (all echoes summed) transfer function acting on the
    ingoing field, [D25] Eq. (A11) with l = m = 2 and gamma = 4:
        K = (-1)^(l+m+1) R / ((1 - R R_BH) gamma D_out) = -R / ((1 - R R_BH) 4 D_out).
    (gamma = 4 is the ECO value of [D25]; it is unrelated to gamma_echo of the
    PDF, which sits inside R.) Change: the reflectivity R is the natural-log
    wall R_wall instead of the original Boltzmann reflectivity. The original
    also contained a truncated n-echo sum, which was not used and is omitted.
    """
    return (-1) * R_w / ((1 - R_w * R_bh) * (4 * D_lmw_out))


def echo_strain(time, Y_in_hybrid, t0, remnant, params, coeffs, taper_width=8.0, t_hold=100.0,
                lp_cut=1e1, hp_cut=7e-5):
    """
    Echo part of the (2,2) strain at infinity, as in the original
    implementation ([D25] Eqs. (A6)-(A10)):
      1. seed = Y_in * planck_taper(v_shell = t0, delta_v = 8); the value at
         t = 100 M is held for all later times (the original's fix of a late-time
         numerical artefact of Y_in; [D25] Eq. (A7) writes it as a constant).
      2. Z_22^echo(omega) = K(omega) Y_seed(omega)
      3. h_echo(t) = -4 FFT[ high_pass(low_pass(Z / omega^2, 10), 7e-5) ]
         (division by omega^2 converts to strain; the filters and the factor
         -4 are those of the original implementation).
    time must be a uniform grid of even length. Returns (h_echo, t_echo).
    """
    seed = Y_in_hybrid * planck_taper(time, t0, taper_width)
    seed[time > t_hold] = seed[time < t_hold][-1]

    dt = time[1] - time[0]
    freq = np.fft.fftfreq(len(time), dt) * 2 * np.pi
    freq[0] = 1e-4  # as in the original: avoid division by zero

    D_lmw_in = interpolate_coeff(coeffs.freq_D, freq, coeffs.D_in)
    D_lmw_out = interpolate_coeff(coeffs.freq_D, freq, coeffs.D_out)

    t_echo = echo_delay(remnant, params)
    K = transfer_function(R_wall(freq, remnant, params, t_echo), R_BH_T(freq, D_lmw_in, D_lmw_out), D_lmw_out)
    # The coefficient tables end at |omega| = 3.96; beyond that the splines
    # extrapolate. R_wall is ~exp(-25) there, but to be safe K is set to zero.
    K[np.abs(freq) > np.max(np.abs(coeffs.freq_D))] = 0

    Z_22_inf_echo = K * np.fft.ifft(seed, norm="ortho")
    h_echo = -4 * np.fft.fft(high_pass(low_pass(1 / freq**2 * Z_22_inf_echo, freq, lp_cut), freq, hp_cut), norm="ortho")
    return h_echo, t_echo


##############################################################################
# Output helpers
##############################################################################

def ringdown_file(event):
    return RINGDOWN_DIR / f"ringdown_{normalize_event_id(event)}.h5"


def echo_file(event):
    return ECHO_DIR / f"echoes_{normalize_event_id(event)}.h5"


def load_ringdown(event):
    """
    Read a ringdown file and rebuild the full extended grid and the modified
    ingoing fields on it (needed for the FFT in compute_echoes.py), with
    exactly the construction used in compute_modified_ringdown.py.
    """
    with h5py.File(ringdown_file(event), "r") as f:
        res = dict(f.attrs)
        grid = dict(f["grid"].attrs)
        for group in ("data", "scan", "qnm", "hybrid"):
            res[group] = {k: f[group][k][()] for k in f[group]}
    t_ext = extended_grid(grid["t_start"], grid["dt"], int(grid["n_ext"]))
    pad = np.full(int(grid["n_ext"] - grid["n_data"]), np.nan, dtype=complex)
    Y_ext = np.concatenate((res["data"]["Y_in"], pad))
    res["t_ext"] = t_ext
    res["Y_in_hybrid_ext"] = [build_hybrid(res["method"], t_ext, Y_ext, res["qnm"]["coeffs_Y_in"],
                                           res["qnm"]["omegas_kerr"], om, res["t_ref"], res["t_attach"],
                                           res["switch_width"], res.get("t_cont"), res.get("cont_width", 50.0))
                              for om in res["qnm"]["omegas_mod"]]
    return res


def load_echoes(event):
    with h5py.File(echo_file(event), "r") as f:
        res = dict(f.attrs)
        for key in ("time", "lambda_nh", "t_echo", "h_echo"):
            res[key] = f[key][()]
        n_sets = len(res["t_echo"])
        values = {k: f["echo_params"][k][()] for k in f["echo_params"]}
    res["echo_sets"] = [EchoParameters(**{k: (int(v[j]) if k in ("ell", "m") else float(v[j]))
                                             for k, v in values.items()}) for j in range(n_sets)]
    return res


def extended_grid(t_start, dt, n):
    """Uniform grid t_start + dt * arange(n) (identical to np.arange on the overlap)."""
    return t_start + dt * np.arange(n)
