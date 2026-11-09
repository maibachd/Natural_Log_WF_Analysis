"""
Step 2 of the pipeline: echoes seeded by the natural-log deformed ringdown.

Requires the output of compute_modified_ringdown.py. The echo computation is
that of N. Deppe et al., PRD 111, 124035 (2025), arXiv:2411.05645, App. A
([D25], see nl_utils), with the natural-log wall reflectivity of the PDF.
For every event, every deformed ringdown (lambda_NH) and every set of echo
parameters:
  1. Take the modified ingoing field Y_in (built as in compute_modified_ringdown.py).
  2. Build the echo seed: Planck taper rising over [t0 - 8 M, t0] (t0 = v_Sigma,
     the scanned ringdown start), value held constant after t = 100 M.
  3. Multiply by the echo transfer function ([D25] Eq. (A11)), with the same
     barrier tables (D_in, D_out) as compute_modified_ringdown.py used for Y_in
         K = -R_wall / ((1 - R_wall R_BH) 4 D_out)
     with the Boltzmann reflectivity of the quantum horizon (Wang, Oshita,
     Afshordi, PRD 101, 024031 (2020); Eq. 16 of the PDF without its prefactor
     gamma_echo, see nl_utils.R_wall)
         R_wall = exp(-|omega_tilde|/(2 T_H)) exp(+i(omega_tilde t_echo + phi_echo)),
         t_echo = alpha_echo / (2 pi T_H) (Lambda_log + Delta_ell)           (Eq. 10),
         Delta_ell = ln[kappa_echo / (zeta l(l+1))],  kappa_echo = 1/(pi gamma^2),
     gamma = dissipation time in Planck times (Oshita, Afshordi, PLB 841, 137901
     (2023), Eqs. 19-20; default: their gamma = 8.5e-3, kappa_echo = 4390),
     where T_H, Omega_H and Lambda_log are those of the Kerr remnant, and the
     barrier reflectivity R_BH = -(D_in/D_out) D/(4 C) of the Kerr remnant
     (nl_utils.barrier_reflectivity_table; Teukolsky-Starobinsky constants of
     Xin et al., PRD 104, 104005 (2021)), or of [D25] for the chi = 0 tables.
  4. Convert to strain: h_echo = -4 FFT[Z / omega^2] (with the low/high-pass
     filters of the original implementation, see nl_utils.echo_strain).
The full (2,2) strain is h_total = h_hybrid + h_echo.

Diagnostics per echo set: max |R_wall R_BH| (>= 1: the echo sum diverges,
warning) and the echo before the ringdown relative to its maximum (FFT
wrap-around of slowly decaying ringing, warning above 3e-3).

As a consistency check, for lambda_NH = 0 the echo is also computed from the
original Y_in and the relative difference is printed. For the default
(residual) method the lambda_NH = 0 field equals the data wherever the seed is
non-constant, so the difference must vanish; for the legacy replace method it
measures the effect of replacing the data by the QNM fit.

Output: output/echoes/echoes_<event>.h5 (see README).

Example:
    python compute_echoes.py --events 1936 --echo alpha_echo=1 --echo alpha_echo=0.7 \
                             --echo alpha_echo=1,kappa_echo=0.3183
"""

import argparse
import warnings
from dataclasses import fields

import h5py
import numpy as np

import nl_utils as nl

PRECURSOR_TOL = 3e-3   # warn if the echo before the ringdown exceeds this fraction of its maximum

DEFAULT_ECHO_SETS = [nl.EchoParameters()]   # alpha_echo = 1, kappa_echo = KAPPA_ECHO_OA23


def parse_echo_set(text):
    """'alpha_echo=1,kappa_echo=4390' -> EchoParameters"""
    names = {f.name: f.type for f in fields(nl.EchoParameters)}
    kwargs = {}
    for item in text.split(","):
        key, value = item.split("=")
        key = key.strip()
        if key not in names:
            raise argparse.ArgumentTypeError(f"unknown echo parameter '{key}' (allowed: {', '.join(names)})")
        kwargs[key] = int(value) if key in ("ell", "m") else float(value)
    return nl.EchoParameters(**kwargs)


def compute_event(event, echo_sets, validate=True):
    event = nl.normalize_event_id(event)
    print(f"\n=== event {event} ===")
    rd = nl.load_ringdown(event)
    remnant = nl.Remnant(mass=rd["remnant_mass"], chi=rd["remnant_chi"], m_total_solar=rd["m_total_solar"])
    # same barrier tables as in compute_modified_ringdown.py (files written
    # before the Kerr tables existed used the chi = 0 tables)
    if rd.get("coeff_tables", "coefficients") == "coefficients":
        coeffs = nl.load_bh_coefficients()
    else:
        coeffs = nl.load_bh_coefficients(rd["coeff_chi"], mass=rd.get("coeff_mass", 1.0))
    print(f"barrier coefficients: {coeffs.source} (chi = {coeffs.chi:+.{nl.SPIN_COEFF_DIGITS}f}, "
          f"evaluated at {coeffs.mass:.5f} M omega)")
    t_ext, t0 = rd["t_ext"], rd["t0"]
    window = t_ext >= rd["hybrid"]["time"][0]
    lambdas = rd["qnm"]["lambda_nh"]
    print(f"T_H = {remnant.T_H:.5f}/M, Omega_H = {remnant.Omega_H:+.5f}/M, Lambda_log = {remnant.Lambda_log():.2f}, t0 = {t0:.2f} M")

    h_echo = np.zeros((len(lambdas), len(echo_sets), window.sum()), dtype=complex)
    t_echos = np.zeros(len(echo_sets))
    freq = np.fft.fftfreq(len(t_ext), t_ext[1] - t_ext[0]) * 2 * np.pi
    freq[0] = 1e-4
    in_table = np.abs(coeffs.mass * freq) <= np.max(np.abs(coeffs.freq_D))
    max_round_trip = np.zeros(len(echo_sets))
    precursor = np.zeros(len(echo_sets))
    for j, params in enumerate(echo_sets):
        # one round trip wall -> barrier -> wall; >= 1 means the echo sum diverges
        round_trip = np.abs(nl.cavity_round_trip(freq, remnant, params, coeffs))[in_table]
        k = np.argmax(round_trip)
        max_round_trip[j] = round_trip[k]
        if round_trip[k] >= 1:
            warnings.warn(f"event {event}, echo set '{params.label()}': |R_wall R_BH| = {round_trip[k]:.3f} >= 1 at "
                          f"M omega = {freq[in_table][k]:+.4f} (superradiant band of the remnant): the echoes grow "
                          "instead of decaying there (ergoregion instability), and the FFT result is not reliable")
        for i, Y_hyb in enumerate(rd["Y_in_hybrid_ext"]):
            h_e, t_echo = nl.echo_strain(t_ext, Y_hyb, t0, remnant, params, coeffs)
            h_echo[i, j] = h_e[window]
        t_echos[j] = t_echo
        if t0 + 2 * t_echo > t_ext[-1]:
            print(f"  note: second echo of set '{params.label()}' lies beyond t_end = {t_ext[-1]:.0f} M")
        # causality check: the echo must vanish before the ringdown. A signal
        # there is ringing that has not decayed by t_end and wraps around in the
        # (periodic) FFT, e.g. when |R_wall R_BH| is close to 1.
        before = t_ext[window] < t0 - 50
        precursor[j] = np.abs(h_echo[:, j][:, before]).max() / np.abs(h_echo[:, j]).max()
        if precursor[j] > PRECURSOR_TOL:
            warnings.warn(f"event {event}, echo set '{params.label()}': the echo before the ringdown (t < t0 - 50 M) "
                          f"reaches {precursor[j]:.1e} of its maximum (wrap-around of slowly decaying ringing); "
                          "increase --t-end of compute_modified_ringdown.py")
        print(f"  echo set {j} ({params.label()}): t_echo = {t_echo:.1f} M, "
              f"max|h_echo| = {np.abs(h_echo[:, j]).max():.3e}, max|R_wall R_BH| = {max_round_trip[j]:.3f}, "
              f"echo before the ringdown / max = {precursor[j]:.1e}")

    # ---- consistency check: lambda_NH = 0 vs. original ingoing field ---------
    if validate and 0.0 in lambdas:
        i0 = int(np.where(lambdas == 0.0)[0][0])
        n_data = len(rd["data"]["Y_in"])
        Y_orig = np.concatenate((rd["data"]["Y_in"], np.full(len(t_ext) - n_data, rd["data"]["Y_in"][-1])))
        h_e_orig, t_echo = nl.echo_strain(t_ext, Y_orig, t0, remnant, echo_sets[0], coeffs)
        first = (t_ext > t0 + 0.5 * t_echo) & (t_ext < t0 + 1.5 * t_echo)
        diff = np.linalg.norm((h_e_orig - nl.echo_strain(t_ext, rd["Y_in_hybrid_ext"][i0], t0, remnant,
                                                          echo_sets[0], coeffs)[0])[first])
        print(f"  check: first echo from original Y_in vs. lambda_NH = 0 ({rd['method']}): relative L2 difference = "
              f"{diff / np.linalg.norm(h_e_orig[first]):.2e}")

    nl.ECHO_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(nl.echo_file(event), "w") as f:
        f.attrs.update(dict(event=event, t0=t0, method=rd["method"], T_H=remnant.T_H, Omega_H=remnant.Omega_H,
                            Lambda_log=remnant.Lambda_log(), m_total_solar=remnant.m_total_solar,
                            coeff_tables=coeffs.source, coeff_chi=coeffs.chi, coeff_mass=coeffs.mass))
        f.create_dataset("time", data=t_ext[window])
        f.create_dataset("lambda_nh", data=lambdas)
        f.create_dataset("t_echo", data=t_echos)
        f.create_dataset("max_round_trip", data=max_round_trip)
        f.create_dataset("precursor", data=precursor)
        for field in fields(nl.EchoParameters):
            f.create_dataset(f"echo_params/{field.name}", data=[getattr(p, field.name) for p in echo_sets])
        f.create_dataset("h_echo", data=h_echo, compression="gzip")   # [lambda, echo set, time]
    print(f"written {nl.echo_file(event)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="+", default=nl.DEFAULT_EVENTS, help="SXS BBH numbers (default: all)")
    parser.add_argument("--echo", action="append", type=parse_echo_set, dest="echo_sets",
                        help="echo parameter set, e.g. 'alpha_echo=1,kappa_echo=4390,phi_echo=0' "
                             "(repeatable; default: alpha_echo = 1, kappa_echo = 55166/(4 pi) = 4390)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the consistency check (echo of the original Y_in vs. lambda_NH = 0)")
    args = parser.parse_args()

    for ev in args.events:
        compute_event(ev, args.echo_sets or DEFAULT_ECHO_SETS, validate=not args.no_validate)
