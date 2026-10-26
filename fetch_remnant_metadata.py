"""
Fetch the remnant properties of the SXS events used in this project and store
them in ``remnant_metadata.json``.

The remnant mass M_f and dimensionless spin chi_f define the Kerr reference
spectrum omega^Kerr_lmn(M_f, chi_f) of Eq. (13) in The_Natural_Log_Project.pdf.
The original echo implementation ([D25], see nl_utils) read the same
quantities (as |chi_f| and M_f) directly from the SXS catalogue with the ``sxs``
package; here they are cached (with the full spin vector, so that the sign of
chi_f along the orbital axis is kept) so that the main scripts do not need the
``sxs`` package or network access.

Only needed when adding new events (requires ``pip install sxs`` and internet):

    python fetch_remnant_metadata.py                  # all default events
    python fetch_remnant_metadata.py 1936 0205        # selected events
"""

import argparse
import json

import numpy as np

from nl_utils import DEFAULT_EVENTS, REMNANT_FILE, normalize_event_id


def fetch(events):
    import sxs  # imported here: optional dependency

    out = {}
    for event in events:
        sim = sxs.load(f"SXS:BBH:{event}")
        md = sim.metadata
        out[event] = dict(
            sxs_id=f"SXS:BBH:{event}",
            sxs_version=str(getattr(sim, "version", "")),
            reference_mass_ratio=float(md.reference_mass_ratio),
            reference_dimensionless_spin1=[float(x) for x in md.reference_dimensionless_spin1],
            reference_dimensionless_spin2=[float(x) for x in md.reference_dimensionless_spin2],
            reference_total_mass=float(md.reference_mass1 + md.reference_mass2),
            remnant_mass=float(md.remnant_mass),
            remnant_dimensionless_spin=[float(x) for x in md.remnant_dimensionless_spin],
        )
        print(f"{event}: M_f = {md.remnant_mass:.5f}, chi_f = {np.round(md.remnant_dimensionless_spin, 4)}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", nargs="*", default=DEFAULT_EVENTS, help="SXS BBH numbers")
    args = parser.parse_args()

    events = [normalize_event_id(e) for e in args.events]
    data = {}
    if REMNANT_FILE.exists():
        data = json.loads(REMNANT_FILE.read_text())
    data.update(fetch(events))
    REMNANT_FILE.write_text(json.dumps(dict(sorted(data.items())), indent=2))
    print(f"written {REMNANT_FILE}")
