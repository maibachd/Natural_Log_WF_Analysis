"""
Step 3: test plots for one event (requires the outputs of
compute_modified_ringdown.py and compute_echoes.py).

Figure 1 (output/plots/ringdown_<event>.png): original NR ringdown vs. the
    modified ringdowns for the different lambda_NH (lambda_NH = 0 is GR: the
    NR data for the default residual method, the QNM fit for the legacy replace
    method). Top: Re h_22; bottom: |h_22 - h_const| on a log scale (h_const =
    constant/memory term of the QNM fit), which shows the changed oscillation
    frequencies and damping times.

Figure 2 (output/plots/echoes_<event>.png): full (2,2) strain
    h_total = h_hybrid + h_echo for the different echo parameter sets (one
    deformed ringdown, --lambda-echo), plus a zoom on the first echo for all
    lambda_NH (fixed echo parameters), which shows how the ringdown deformation
    propagates into the echoes.

Example:
    python plot_test_run.py --event 1936
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import nl_utils as nl

LAMBDA_COLORS = {-1.0: "tab:blue", 0.0: "0.35", 1.0: "tab:red"}
ECHO_COLORS = ["tab:purple", "tab:green", "tab:orange", "tab:brown", "tab:pink"]


def lambda_color(lam):
    return LAMBDA_COLORS.get(float(lam), None)


def lambda_label(lam, rd):
    if lam == 0:
        gr = "= NR data" if rd["method"] == "residual" else "GR QNM fit"
        return rf"$\lambda_{{\rm NH}}=0$ ({gr})"
    d = rd["qnm"]["delta_omega"][list(rd["qnm"]["lambda_nh"]).index(lam)][0]
    return rf"$\lambda_{{\rm NH}}={lam:+g}$ ($\delta\omega=\delta\tau={d:+.3f}$)"


def plot_ringdown(rd, event):
    t_data, h_data = rd["data"]["time"], rd["data"]["h"]
    th, h_hyb = rd["hybrid"]["time"], rd["hybrid"]["h"]
    t_a, width = rd["t_attach"], rd["switch_width"]
    h_const = rd["qnm"]["coeffs_h"][0]  # constant (memory) term of the QNM fit

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=dict(hspace=0.08))
    for ax, func in ((axes[0], np.real), (axes[1], lambda x: np.abs(x - h_const))):
        ax.plot(t_data, func(h_data), color="k", lw=2.2, label="NR (CCE) data")
        for lam, h in zip(rd["qnm"]["lambda_nh"], h_hyb):
            ls = "--" if lam == 0 else "-"
            ax.plot(th, func(h), ls=ls, lw=1.4, color=lambda_color(lam), label=lambda_label(lam, rd))
        ax.axvline(t_a, color="0.5", lw=0.8, ls=":")
        ax.axvspan(t_a, t_a + width, color="0.85", lw=0)
    axes[0].set_xlim(-60, 130)
    axes[0].set_ylabel(r"Re $h_{22}$  [$r/M$]")
    axes[0].legend(loc="upper right", fontsize=9)
    if rd["method"] == "residual":
        axes[0].text(t_a + width + 1, 0.03, r"$\leftarrow$ deformation switched on at $t_a$", fontsize=9,
                     color="0.4", ha="left", transform=axes[0].get_xaxis_transform())
    else:
        axes[0].text(t_a - 1, 0.03, r"$t_0$ (+ blend) $\rightarrow$", fontsize=9, color="0.4", ha="right",
                     transform=axes[0].get_xaxis_transform())
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-6, 1)
    axes[1].set_ylabel(r"$|h_{22} - h_{\rm const}|$")
    axes[1].set_xlabel(r"$t - t_{\rm peak}$  [$M$]")
    if rd["method"] == "residual":
        how = (f"NR data + deformation of the QNMs n ≤ {rd['n_overtones']} "
               f"(amplitudes fitted from t = {rd['t_fit']:g} M)")
    else:
        how = f"QNM fit (n ≤ {rd['n_overtones']}) with deformed spectrum replaces the data after t0"
    axes[0].set_title(
        f"SXS:BBH:{event} ringdown: {how}\n"
        rf"$M_f={rd['remnant_mass']:.4f}M$, $\chi_f={rd['remnant_chi']:+.3f}$, "
        rf"$\Lambda_{{\rm log}}={rd['Lambda_log']:.1f}$ ($M={rd['m_total_solar']:.0e}M_\odot$), $p={rd['p']:g}$",
        fontsize=11)
    return fig


def plot_echoes(rd, ec, event, lambda_echo):
    th = rd["hybrid"]["time"]
    lambdas = list(ec["lambda_nh"])
    i_lam = lambdas.index(lambda_echo)
    h_hyb = rd["hybrid"]["h"]
    t0 = rd["t0"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw=dict(hspace=0.35))

    # (a) full signal and (b) zoom on the echo amplitude: different echo parameters
    t_max = min(th[-1], t0 + 3.2 * ec["t_echo"].max())
    for ax in axes[:2]:
        for j, params in enumerate(ec["echo_sets"]):
            h_tot = h_hyb[i_lam] + ec["h_echo"][i_lam, j]
            ax.plot(th, np.real(h_tot), lw=1.0, color=ECHO_COLORS[j % len(ECHO_COLORS)],
                    label=rf"$\alpha_{{\rm echo}}={params.alpha_echo:g}$, $\kappa_{{\rm echo}}={params.kappa_echo:.4g}$"
                          rf" ($t_{{\rm echo}}={ec['t_echo'][j]:.0f}M$)")
        ax.set_xlim(-200, t_max)
        ax.set_ylabel(r"Re $h_{22}$")
    amp = np.abs(ec["h_echo"][i_lam]).max()
    late = np.real(h_hyb[i_lam][th > 150]).mean()
    axes[1].set_ylim(late - 1.3 * amp, late + 1.3 * amp)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"SXS:BBH:{event}: deformed ringdown ($\\lambda_{{\\rm NH}}={lambda_echo:+g}$) + echoes "
                      f"(different echo parameters)", fontsize=11)
    axes[1].set_title("zoom on the echoes", fontsize=10)
    axes[1].set_xlabel(r"$t - t_{\rm peak}$  [$M$]")

    # (c) first echo for the different ringdown deformations, first echo parameter set
    j = 0
    t_e = ec["t_echo"][j]
    for i, lam in enumerate(lambdas):
        ls = "--" if lam == 0 else "-"
        axes[2].plot(th, np.real(ec["h_echo"][i, j]), ls=ls, lw=1.3, color=lambda_color(lam),
                     label=lambda_label(lam, rd))
    axes[2].set_xlim(t0 + t_e - 60, t0 + t_e + 140)
    axes[2].set_ylabel(r"Re $h^{\rm echo}_{22}$")
    axes[2].set_xlabel(r"$t - t_{\rm peak}$  [$M$]")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_title(f"first echo for different ringdown deformations "
                      f"({ec['echo_sets'][j].label()})", fontsize=10)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event", default="1936")
    parser.add_argument("--lambda-echo", type=float, default=1.0,
                        help="lambda_NH of the deformed ringdown used in the echo-parameter comparison")
    args = parser.parse_args()

    event = nl.normalize_event_id(args.event)
    rd = nl.load_ringdown(event)
    ec = nl.load_echoes(event)
    nl.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plot_ringdown(rd, event)
    fig.savefig(nl.PLOT_DIR / f"ringdown_{event}.png", dpi=150, bbox_inches="tight")
    fig = plot_echoes(rd, ec, event, args.lambda_echo)
    fig.savefig(nl.PLOT_DIR / f"echoes_{event}.png", dpi=150, bbox_inches="tight")
    print(f"written {nl.PLOT_DIR / f'ringdown_{event}.png'} and {nl.PLOT_DIR / f'echoes_{event}.png'}")
