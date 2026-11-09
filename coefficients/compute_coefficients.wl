(* ::Package:: *)

(* compute_coefficients.wl

   Tabulates the barrier coefficients CCoeff.h5 and DCoeff.h5 for a Kerr
   remnant of dimensionless spin chi, with the Teukolsky package of the
   Black Hole Perturbation Toolkit (bhptoolkit.org). This is the chapter
   "Same Coefficients for the Teukolsky Equation" of Coeff_QG_Mem.nb, made
   non-interactive and with the spin as a parameter:

     DCoeff.h5:  TeukolskyRadial[-2, 2, 2, chi, omega]["Up"]["Amplitudes"]
     CCoeff.h5:  TeukolskyRadial[+2, 2, 2, chi, omega]["Up"]["Amplitudes"]

   each with the datasets "frequency" (M omega, M = remnant mass),
   "inc" (= "Incidence") and "ref" (= "Reflection"), on the frequency grid of
   the notebook. For chi = 0 this reproduces coefficients/CCoeff.h5 and
   coefficients/DCoeff.h5 exactly.

   Differences to the notebook:
     * The notebook obtains omega < 0 from omega > 0 by complex conjugation,
       which uses the chi = 0 symmetry R(-omega) = R(omega)^*. For chi != 0 and
       fixed m = 2 this symmetry does not hold (it maps m -> -m), so the
       negative frequencies are computed directly.
     * Both amplitudes are taken from a single TeukolskyRadial call per
       frequency, and the frequencies are evaluated on parallel kernels.

   In addition, Eigenvalue.h5 holds the eigenvalue of the spin-weighted
   spheroidal harmonic, SpinWeightedSpheroidalEigenvalue[-2, 2, 2, chi omega]
   (datasets "frequency", "eigenvalue"; BHPT convention, which already
   contains a^2 omega^2 - 2 a m omega, i.e. eigenvalue + 2 = Q of Xin et al.,
   PRD 104, 104005 (2021), Eq. (53)). It is needed for the Starobinsky
   constant C of a Kerr remnant (nl_utils.starobinsky_constant).

   Usage (normally called by nl_utils.load_bh_coefficients):
     wolframscript -file compute_coefficients.wl <chi> <output directory> [eigenvalues-only]

   The output directory must exist. With "eigenvalues-only" only Eigenvalue.h5
   is written (a few seconds). The script stops with a non-zero exit code if
   any amplitude cannot be computed or is not finite. *)

args = $ScriptCommandLine;
If[Length[args] < 3,
  Print["usage: wolframscript -file compute_coefficients.wl <chi> <output directory> [eigenvalues-only]"]; Exit[2]];
eigenvaluesOnly = Length[args] >= 4 && args[[4]] === "eigenvalues-only";
chi = ToExpression[args[[2]]];
outDir = args[[3]];
If[!(NumericQ[chi] && Abs[chi] < 1), Print["chi must be a number with |chi| < 1, got ", args[[2]]]; Exit[2]];
If[!DirectoryQ[outDir], Print["output directory does not exist: ", outDir]; Exit[2]];
chi = N[chi];

(* frequency grid of Coeff_QG_Mem.nb ("A list of omega>0 for evaluation" and
   "A list of omega<0 for evaluation"). The first two ranges overlap around
   M omega ~ 0.2, which gives the duplicated frequencies that
   nl_utils.interpolate_coeff removes. *)
omegatabp = Join[10^Range[-4, -0.698, 0.002], 10^Range[-0.6999, 0.017, 0.001], 10^Range[0.018, 0.6, 0.02]];
omegataball = Join[Reverse[-omegatabp], omegatabp];

Needs["SpinWeightedSpheroidalHarmonics`"];
eigen = SpinWeightedSpheroidalEigenvalue[-2, 2, 2, chi #] & /@ omegataball;
If[!VectorQ[eigen, NumberQ] || Max[Abs[Im[eigen]]] > 0,
  Print["spheroidal eigenvalues could not be computed"]; Exit[1]];
file = FileNameJoin[{outDir, "Eigenvalue.h5"}];
Export[file, {"Datasets" -> {"frequency" -> omegataball, "eigenvalue" -> Re[eigen]},
   "DataFormat" -> {Automatic, Automatic}}, "Rules"];
Print["written ", file];
If[eigenvaluesOnly, Exit[0]];

LaunchKernels[];
ParallelNeeds["Teukolsky`"];
Needs["Teukolsky`"];

amplitudes[s_, w_] := Module[{amp},
  amp = Quiet[TeukolskyRadial[s, 2, 2, chi, w]["Up"]["Amplitudes"]];
  {amp["Incidence"], amp["Reflection"]}];
DistributeDefinitions[amplitudes, chi];

Print["chi = ", chi, ", ", Length[omegataball], " frequencies, ", $KernelCount, " kernels"];
Do[
  t0 = AbsoluteTime[];
  table = ParallelMap[amplitudes[s, #] &, omegataball, Method -> "FinestGrained"];
  If[!MatrixQ[table, NumericQ] || !And @@ (NumberQ /@ N[Flatten[table]]),
    Print["s = ", s, ": some amplitudes could not be computed"]; Exit[1]];
  file = FileNameJoin[{outDir, If[s == -2, "DCoeff.h5", "CCoeff.h5"]}];
  Export[file, {"Datasets" -> {"frequency" -> omegataball, "inc" -> table[[All, 1]], "ref" -> table[[All, 2]]},
     "DataFormat" -> {Automatic, Automatic, Automatic}}, "Rules"];
  Print["s = ", s, ": written ", file, " (", Round[AbsoluteTime[] - t0], " s)"],
  {s, {-2, 2}}];
CloseKernels[];
Exit[0];
