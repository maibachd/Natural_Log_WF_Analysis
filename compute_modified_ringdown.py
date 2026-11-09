"""
Step 1 of the pipeline: natural-log deformed ringdown.

For every event:
  1. Load the (2,2) mode of the strain h and of Psi0 (CCE data of the event folder),
     with t = 0 at the peak of |h|.
  2. Compute the ingoing field at the horizon, Y_22^in = FFT[Psi0 C_in]
     ([D25] App. A, see nl_utils), which later seeds the echoes. C_in is taken
     from the Kerr barrier tables for the remnant spin chi_f
     (coefficients/chi_<chi_f>/, computed with Mathematica on first use, see
     nl_utils.load_bh_coefficients), evaluated at M_f omega (the tables are in
     units of the remnant mass); --schwarzschild-coefficients uses the chi = 0
     tables of [D25] instead, evaluated at M omega as in the original.
  3. Take the Kerr QNM spectrum omega^Kerr_22n(M_f, chi_f) of the remnant
     (SXS metadata, remnant_metadata.json) and determine the ringdown start t0
     (= v_Sigma of [D25]) as the minimum of the QNM-fit mismatch of Y_in over a
     grid of start times. t0 is where the echo seed starts (compute_echoes.py).
  4. Fit the QNM amplitudes of Y_in and of h.
  5. Deform the spectrum according to Eqs. (13)-(14) of the PDF,
         omega_n -> omega_n (1 + delta_omega_n),  tau_n -> tau_n (1 + delta_tau_n),
         delta_{omega,tau}_n = c^{(omega,tau)}_n lambda_NH Lambda_log^-p,
     with Lambda_log = ln(zeta S_BH) of the remnant (Eq. 5), for every lambda_NH.
  6. Build the modified h and Y_in on an extended time grid (up to --t-end,
     needed to fit late echoes into the time window).

Steps 4 and 6 depend on --method (details and references: README, "Modified ringdown"):

  residual (default)
     4. Fit only QNMs whose amplitudes are stable (default n <= 1) from
        t_fit = 10 M after the peak; check the stability by refitting from
        t_fit -+ 5 M.
     6. h_mod = h_NR + Delta, where Delta is the difference between the deformed
        and the GR QNM content with the same amplitudes, switched on smoothly at
        t_a = 5 M after the peak. lambda_NH = 0 reproduces the NR data exactly.
        Beyond the trustworthy end of the data the fields are continued by
        their GR QNM fit.

  replace (legacy)
     4. Fit n <= 3 overtones from the scanned t0.
     6. Cut the data at t0 and attach the fit with the deformed QNMs (Planck
        blend over 5 M). The overtone amplitudes of this fit are not stable,
        and deforming them produces a large spurious change right after t0.

Output: output/ringdown/ringdown_<event>.h5 (see README for its layout).

Example:
    python compute_modified_ringdown.py --events 1936 --lambda-nh -1 0 1
"""

import argparse
import warnings

import h5py
import numpy as np

import nl_utils as nl


def compute_event(event, args):
    event = nl.normalize_event_id(event)
    print(f"\n=== event {event} ({args.method}) ===")

    # ---- 1. data -------------------------------------------------------------
    data = nl.load_event(event)
    rem_md = nl.load_remnant(event)
    remnant = nl.Remnant(mass=rem_md["mass"], chi=rem_md["chi"], m_total_solar=args.m_total)
    print(f"remnant ({rem_md['sxs_id']}): M_f = {remnant.mass:.5f} M, chi_f = {remnant.chi:+.4f}")
    if args.schwarzschild_coefficients:
        coeffs = nl.load_bh_coefficients()
    else:
        coeffs = nl.load_bh_coefficients(remnant.chi, mass=remnant.mass)
    print(f"barrier coefficients: {coeffs.source} (chi = {coeffs.chi:+.{nl.SPIN_COEFF_DIGITS}f}, "
          f"evaluated at {coeffs.mass:.5f} M omega)")
    if args.schwarzschild_coefficients and abs(remnant.chi) > nl.SCHWARZSCHILD_SPIN_WARNING:
        warnings.warn(f"event {event}: |chi_f| = {abs(remnant.chi):.2f}, but the barrier coefficients are those of "
                      "Schwarzschild (--schwarzschild-coefficients), so Y_in and the echo transfer function are only "
                      "approximate for this remnant.")

    # ---- 2. frequency domain and ingoing field ([D25] App. A) ----------------
    freq, fft_psi0, t_data = nl.fft_for_waveforms_CCE(data["time"], data["psi0"])
    freq[0] = 1e-4  # as in the original implementation
    Y_in_data = nl.ingoing_field(freq, fft_psi0, coeffs)
    h_data = nl.resample_uniform(data["time"], data["h"], t_data)
    dt = t_data[1] - t_data[0]
    t_tukey = nl.tukey_taper_start(t_data)
    if t_tukey < args.upper_t:
        warnings.warn(f"event {event}: the Tukey window of the Psi0 FFT starts at "
                      f"t = {t_tukey:.0f} M, inside the fit window (< {args.upper_t} M)")
    print(f"uniform grid: t = [{t_data[0]:.1f}, {t_data[-1]:.1f}] M, dt = {dt:.3f} M, N = {len(t_data)}")

    # ---- 3. ringdown start t0 (start of the echo seed) -----------------------
    omegas_scan = nl.kerr_qnm_omegas(remnant.chi, remnant.mass, args.n_scan)
    t0_grid = np.linspace(args.t0_min, args.t0_max, args.t0_num)
    t0, scan = nl.scan_ringdown_start(t_data, Y_in_data, omegas_scan, t0_grid, args.upper_t)
    print(f"ringdown start (min. mismatch of Y_in, n <= {args.n_scan}): t0 = {t0:.2f} M, mismatch = {scan.min():.2e}")
    if t0 in (t0_grid[0], t0_grid[-1]):
        warnings.warn(f"event {event}: t0 = {t0} lies on the edge of the scanned interval; consider widening it")

    # ---- 4. QNM amplitude fits of Y_in and h ---------------------------------
    omegas_kerr = nl.kerr_qnm_omegas(remnant.chi, remnant.mass, args.n_overtones)
    if args.method == "residual":
        t_ref, t_attach = args.t_fit, args.t_attach
    else:
        t_ref, t_attach = t0, t0
    fields = {"Y_in": Y_in_data, "h": h_data}
    ringdown = (t_data >= t0) & (t_data < args.upper_t)
    fits = {}
    for name, field in fields.items():
        coeffs_fit, t_cut, model, field_cut = nl.fit_qnm_amplitudes(t_data, field, t_ref, omegas_kerr, args.upper_t)
        fit = dict(coeffs=coeffs_fit, mismatch=nl.mismatch_ringdown(field_cut, model, t_cut),
                   scale=np.max(np.abs(field[ringdown])))   # signal peak in the ringdown region [t0, upper_t)
        msg = f"QNM fit of {name:4s} (n <= {args.n_overtones}, from t = {t_ref:.2f} M): mismatch = {fit['mismatch']:.2e}"
        if args.method == "residual":
            fit["stability"] = nl.qnm_amplitude_stability(t_data, field, omegas_kerr, t_ref, args.upper_t,
                                                          args.stability_step)
            msg += ", amplitude spread under t_fit -+ {:g} M: ".format(args.stability_step) + \
                   ", ".join(f"n={n}: {s:.2f}" for n, s in enumerate(fit["stability"]))
            tol = np.where(np.arange(len(omegas_kerr)) == 0, args.stability_tol_fundamental,
                           args.stability_tol_overtones)
            unstable = np.flatnonzero(fit["stability"] > tol)
            if unstable.size:
                warnings.warn(f"event {event}, {name}: QNM amplitude(s) n = {unstable.tolist()} change by more than "
                              f"the tolerance when the fit start is moved by {args.stability_step:g} M "
                              f"(spread {np.round(fit['stability'][unstable], 2)}); their deformation is uncertain")
        else:
            fit["jump"] = np.abs(field_cut[0] - model[0]) / np.max(np.abs(field_cut))
            msg += f", relative jump at t0 = {fit['jump']:.2e}"
        msg += f", max QNM amplitude / signal = {np.max(np.abs(coeffs_fit[1:])) / fit['scale']:.1f}"
        fits[name] = fit
        print(msg)

    # ---- 5. natural-log deformation ------------------------------------------
    Lambda = remnant.Lambda_log(args.zeta)
    n_modes = args.n_overtones + 1
    lambdas = np.array(args.lambda_nh, dtype=float)
    deltas_omega, deltas_tau, omegas_mod = [], [], []
    for lam in lambdas:
        d_om, d_tau = nl.natural_log_deformation(lam, Lambda, args.p, args.c_omega, args.c_tau, n_modes)
        deltas_omega.append(d_om)
        deltas_tau.append(d_tau)
        omegas_mod.append(nl.deform_qnm_omegas(omegas_kerr, d_om, d_tau))
    print(f"Lambda_log = {Lambda:.2f} (M_f = {remnant.mass_solar:.3e} M_sun), "
          f"Lambda_log^-p = {Lambda ** -args.p:.4f}")

    # ---- 6. modified waveforms on an extended grid ---------------------------
    n_ext = int(np.ceil((args.t_end - t_data[0]) / dt)) + 1
    n_ext += n_ext % 2  # even length, needed by interpolate_coeff
    t_ext = nl.extended_grid(t_data[0], dt, n_ext)
    pad = np.full(n_ext - len(t_data), np.nan, dtype=complex)
    window = t_ext >= args.t_save_min
    # Residual method: the data are used up to the start of the Tukey window of
    # the Psi0 FFT (Y_in is tapered after that) and continued by the GR fit.
    t_cont = t_tukey if args.method == "residual" else None
    if t_cont is not None and t_cont - args.cont_width < args.upper_t:
        warnings.warn(f"event {event}: the continuation by the QNM fit starts at t = {t_cont - args.cont_width:.0f} M, "
                      f"inside the fit window (< {args.upper_t} M)")
    check = (t_ext >= t_attach) & (t_ext < args.upper_t)

    hybrids = {name: [] for name in fields}
    deviations = []
    for lam, om, d_om, d_tau in zip(lambdas, omegas_mod, deltas_omega, deltas_tau):
        dev = {}
        for name, field in fields.items():
            hyb = nl.build_hybrid(args.method, t_ext, np.concatenate((field, pad)), fits[name]["coeffs"],
                                  omegas_kerr, om, t_ref, t_attach, args.switch_width, t_cont, args.cont_width)
            gr = nl.build_hybrid(args.method, t_ext, np.concatenate((field, pad)), fits[name]["coeffs"],
                                 omegas_kerr, omegas_kerr, t_ref, t_attach, args.switch_width, t_cont, args.cont_width)
            # size of the deformation relative to the ringdown peak. For the replace
            # method, overfitted (nearly cancelling) overtone amplitudes make this blow up.
            dev[name] = np.max(np.abs(hyb - gr)[check]) / fits[name]["scale"]
            hybrids[name].append(hyb[window])
        deviations.append([dev["h"], dev["Y_in"]])
        print(f"  lambda_NH = {lam:+.2f}: delta_omega_0 = {d_om[0]:+.4f}, delta_tau_0 = {d_tau[0]:+.4f}, "
              f"omega_220 = {om[0]:.5f} (Kerr {omegas_kerr[0]:.5f}); "
              f"max|deformed - GR| / ringdown peak: h {dev['h']:.2f}, Y_in {dev['Y_in']:.2f}")
        if max(dev.values()) > args.max_deviation:
            warnings.warn(f"event {event}, lambda_NH = {lam}: the deformed ringdown deviates from the GR one by up to "
                          f"{max(dev.values()):.1f} x the ringdown peak; check the QNM amplitudes (overfitting?)")

    # ---- save ----------------------------------------------------------------
    nl.RINGDOWN_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(nl.ringdown_file(event), "w") as f:
        attrs = dict(
            event=event, sxs_id=rem_md["sxs_id"], remnant_mass=remnant.mass, remnant_chi=remnant.chi,
            m_total_solar=args.m_total, zeta=args.zeta, p=args.p, Lambda_log=Lambda, t_peak=data["t_peak"],
            method=args.method, n_overtones=args.n_overtones, n_scan=args.n_scan, upper_t=args.upper_t,
            t0=t0, t_ref=t_ref, t_attach=t_attach, switch_width=args.switch_width,
            mismatch_h=fits["h"]["mismatch"], mismatch_Y_in=fits["Y_in"]["mismatch"],
            coeff_tables=coeffs.source, coeff_chi=coeffs.chi, coeff_mass=coeffs.mass,
        )
        if args.method == "residual":
            attrs.update(t_fit=args.t_fit, t_cont=t_cont, cont_width=args.cont_width)
        else:
            attrs.update(jump_h=fits["h"]["jump"], jump_Y_in=fits["Y_in"]["jump"])
        f.attrs.update(attrs)
        g = f.create_group("grid")
        g.attrs.update(dict(t_start=t_data[0], dt=dt, n_data=len(t_data), n_ext=n_ext))
        g = f.create_group("data")
        g.create_dataset("time", data=t_data)
        g.create_dataset("h", data=h_data)
        g.create_dataset("Y_in", data=Y_in_data)
        g = f.create_group("scan")
        g.create_dataset("t0_grid", data=t0_grid)
        g.create_dataset("mismatch", data=scan)
        g = f.create_group("qnm")
        g.create_dataset("omegas_kerr", data=omegas_kerr)
        g.create_dataset("coeffs_h", data=fits["h"]["coeffs"])
        g.create_dataset("coeffs_Y_in", data=fits["Y_in"]["coeffs"])
        if args.method == "residual":
            g.create_dataset("stability_h", data=fits["h"]["stability"])
            g.create_dataset("stability_Y_in", data=fits["Y_in"]["stability"])
        g.create_dataset("lambda_nh", data=lambdas)
        g.create_dataset("c_omega", data=np.broadcast_to(np.asarray(args.c_omega, float), (n_modes,)))
        g.create_dataset("c_tau", data=np.broadcast_to(np.asarray(args.c_tau, float), (n_modes,)))
        g.create_dataset("delta_omega", data=np.array(deltas_omega))
        g.create_dataset("delta_tau", data=np.array(deltas_tau))
        g.create_dataset("omegas_mod", data=np.array(omegas_mod))
        g.create_dataset("deviation_h_Y_in", data=np.array(deviations))
        g = f.create_group("hybrid")
        g.create_dataset("time", data=t_ext[window])
        g.create_dataset("h", data=np.array(hybrids["h"]), compression="gzip")
        g.create_dataset("Y_in", data=np.array(hybrids["Y_in"]), compression="gzip")
    print(f"written {nl.ringdown_file(event)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="+", default=nl.DEFAULT_EVENTS, help="SXS BBH numbers (default: all)")
    parser.add_argument("--method", choices=["residual", "replace"], default="residual",
                        help="construction of the modified ringdown (default: residual; replace = legacy version)")
    parser.add_argument("--lambda-nh", nargs="+", type=float, default=[-1.0, 0.0, 1.0],
                        help="no-hair amplitudes lambda_NH (Eq. 6/14); 0 = undeformed GR")
    parser.add_argument("--p", type=float, default=0.5, help="no-hair scaling exponent p (default 1/2)")
    parser.add_argument("--c-omega", nargs="+", type=float, default=[1.0],
                        help="c^(omega)_n of Eq. (14): one value for all overtones or one per overtone")
    parser.add_argument("--c-tau", nargs="+", type=float, default=[1.0],
                        help="c^(tau)_n of Eq. (14): one value for all overtones or one per overtone")
    parser.add_argument("--m-total", type=float, default=1e6,
                        help="detector-frame total mass of the binary in M_sun (sets S_BH, Lambda_log)")
    parser.add_argument("--zeta", type=float, default=1.0, help="entropy normalisation zeta, Eq. (5)")
    parser.add_argument("--n-overtones", type=int, default=None,
                        help="highest overtone n of the ringdown model that is deformed "
                             "(default: 1 for residual, 3 for replace)")
    parser.add_argument("--t-attach", type=float, default=5.0,
                        help="residual: time [M after peak] at which the deformation is switched on (t_a)")
    parser.add_argument("--t-fit", type=float, default=10.0,
                        help="residual: start [M after peak] of the QNM amplitude fit")
    parser.add_argument("--stability-step", type=float, default=5.0,
                        help="residual: the amplitude fit is repeated from t_fit -+ this value [M] to test stability")
    parser.add_argument("--stability-tol-fundamental", type=float, default=0.1,
                        help="residual: warn if the relative spread of the n = 0 amplitude exceeds this")
    parser.add_argument("--stability-tol-overtones", type=float, default=0.5,
                        help="residual: warn if the relative spread of an n >= 1 amplitude exceeds this")
    parser.add_argument("--switch-width", "--blend-width", type=float, default=5.0,
                        help="width [M] of the smooth switch-on of the deformation (residual) or of the "
                             "data-to-model blend after t0 (replace); 0 = step")
    parser.add_argument("--cont-width", type=float, default=50.0,
                        help="residual: width [M] of the blend from the data to the GR QNM continuation, ending "
                             "where the Tukey window of the Psi0 FFT starts")
    parser.add_argument("--max-deviation", type=float, default=0.5,
                        help="warn if the deformed ringdown deviates from the GR one by more than this "
                             "fraction of the ringdown peak")
    parser.add_argument("--n-scan", type=int, default=3,
                        help="highest overtone of the QNM model used in the t0 scan (the original used 5)")
    parser.add_argument("--schwarzschild-coefficients", action="store_true",
                        help="use the chi = 0 barrier tables of [D25] (coefficients/{C,D}Coeff.h5) for every event "
                             "instead of the Kerr tables for the remnant spin")
    parser.add_argument("--t0-min", type=float, default=-20.0, help="start of the t0 scan [M]")
    parser.add_argument("--t0-max", type=float, default=10.0, help="end of the t0 scan [M]")
    parser.add_argument("--t0-num", type=int, default=121, help="number of t0 values")
    parser.add_argument("--upper-t", type=float, default=100.0, help="end of the QNM fit windows [M]")
    parser.add_argument("--t-end", type=float, default=3000.0,
                        help="end of the extended time grid [M after peak]; must contain the echoes")
    parser.add_argument("--t-save-min", type=float, default=-1000.0,
                        help="modified waveforms are stored from this time on [M]")
    args = parser.parse_args()
    if args.n_overtones is None:
        args.n_overtones = 1 if args.method == "residual" else 3
    n_modes = args.n_overtones + 1
    for name in ("c_omega", "c_tau"):
        val = getattr(args, name)
        if len(val) not in (1, n_modes):
            parser.error(f"--{name.replace('_', '-')} needs 1 or {n_modes} values")
        setattr(args, name, val[0] if len(val) == 1 else val)
    if args.method == "residual" and args.t_fit - args.stability_step < args.t_attach:
        print(f"note: the stability test fits from t = {args.t_fit - args.stability_step:g} M, "
              f"before t_a = {args.t_attach:g} M")
    return args


if __name__ == "__main__":
    args = parse_args()
    for ev in args.events:
        compute_event(ev, args)
