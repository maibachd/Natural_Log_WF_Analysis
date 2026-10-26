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
  3. Multiply by the echo transfer function ([D25] Eq. (A11))
         K = -R_wall / ((1 - R_wall R_BH) 4 D_out)
     with the natural-log quantum-horizon reflectivity (Eq. 16)
         R_wall = gamma_echo exp(-|omega_tilde|/(2 T_H)) exp(+i(omega_tilde t_echo + phi_echo)),
         t_echo = alpha_echo / (2 pi T_H) (Lambda_log + Delta_ell)           (Eq. 10),
     where T_H, Omega_H and Lambda_log are those of the Kerr remnant.
  4. Convert to strain: h_echo = -4 FFT[Z / omega^2] (with the low/high-pass
     filters of the original implementation, see nl_utils.echo_strain).
The full (2,2) strain is h_total = h_hybrid + h_echo.

As a consistency check, for lambda_NH = 0 the echo is also computed from the
original Y_in and the relative difference is printed. For the default
(residual) method the lambda_NH = 0 field equals the data wherever the seed is
non-constant, so the difference must vanish; for the legacy replace method it
measures the effect of replacing the data by the QNM fit.

Output: output/echoes/echoes_<event>.h5 (see README).

Example:
    python compute_echoes.py --events 1936 --echo alpha_echo=0.4 --echo alpha_echo=0.7 \
                             --echo alpha_echo=0.4,gamma_echo=0.5
"""

import argparse
from dataclasses import fields

import h5py
import numpy as np

import nl_utils as nl

DEFAULT_ECHO_SETS = [
    nl.EchoParameters(alpha_echo=0.4, gamma_echo=1.0),
    nl.EchoParameters(alpha_echo=0.7, gamma_echo=1.0),
    nl.EchoParameters(alpha_echo=0.4, gamma_echo=0.5),
]


def parse_echo_set(text):
    """'alpha_echo=0.4,gamma_echo=0.5' -> EchoParameters"""
    names = {f.name: f.type for f in fields(nl.EchoParameters)}
    kwargs = {}
    for item in text.split(","):
        key, value = item.split("=")
        key = key.strip()
        if key not in names:
            raise argparse.ArgumentTypeError(f"unknown echo parameter '{key}' (allowed: {', '.join(names)})")
        kwargs[key] = int(value) if key in ("ell", "m") else float(value)
    return nl.EchoParameters(**kwargs)


def compute_event(event, echo_sets, coeffs, validate=True):
    event = nl.normalize_event_id(event)
    print(f"\n=== event {event} ===")
    rd = nl.load_ringdown(event)
    remnant = nl.Remnant(mass=rd["remnant_mass"], chi=rd["remnant_chi"], m_total_solar=rd["m_total_solar"])
    t_ext, t0 = rd["t_ext"], rd["t0"]
    window = t_ext >= rd["hybrid"]["time"][0]
    lambdas = rd["qnm"]["lambda_nh"]
    print(f"T_H = {remnant.T_H:.5f}/M, Omega_H = {remnant.Omega_H:+.5f}/M, Lambda_log = {remnant.Lambda_log():.2f}, t0 = {t0:.2f} M")

    h_echo = np.zeros((len(lambdas), len(echo_sets), window.sum()), dtype=complex)
    t_echos = np.zeros(len(echo_sets))
    for j, params in enumerate(echo_sets):
        for i, Y_hyb in enumerate(rd["Y_in_hybrid_ext"]):
            h_e, t_echo = nl.echo_strain(t_ext, Y_hyb, t0, remnant, params, coeffs)
            h_echo[i, j] = h_e[window]
        t_echos[j] = t_echo
        if t0 + 2 * t_echo > t_ext[-1]:
            print(f"  note: second echo of set '{params.label()}' lies beyond t_end = {t_ext[-1]:.0f} M")
        print(f"  echo set {j} ({params.label()}): t_echo = {t_echo:.1f} M, "
              f"max|h_echo| = {np.abs(h_echo[:, j]).max():.3e}")

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
                            Lambda_log=remnant.Lambda_log(), m_total_solar=remnant.m_total_solar))
        f.create_dataset("time", data=t_ext[window])
        f.create_dataset("lambda_nh", data=lambdas)
        f.create_dataset("t_echo", data=t_echos)
        for field in fields(nl.EchoParameters):
            f.create_dataset(f"echo_params/{field.name}", data=[getattr(p, field.name) for p in echo_sets])
        f.create_dataset("h_echo", data=h_echo, compression="gzip")   # [lambda, echo set, time]
    print(f"written {nl.echo_file(event)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="+", default=nl.DEFAULT_EVENTS, help="SXS BBH numbers (default: all)")
    parser.add_argument("--echo", action="append", type=parse_echo_set, dest="echo_sets",
                        help="echo parameter set, e.g. 'alpha_echo=0.4,gamma_echo=0.5,phi_echo=0' "
                             "(repeatable; defaults: alpha 0.4/0.7 with gamma 1, alpha 0.4 with gamma 0.5)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the consistency check (echo of the original Y_in vs. lambda_NH = 0)")
    args = parser.parse_args()

    coeffs = nl.load_bh_coefficients()
    for ev in args.events:
        compute_event(ev, args.echo_sets or DEFAULT_ECHO_SETS, coeffs, validate=not args.no_validate)
