================================================================
Equilibrium generation for the 16 dynamics of the first submission
Producing the <tag>_<stem>_equilibrium_data.csv files from networks
================================================================

This folder holds the scripts that produced the equilibrium CSVs of the
16 dynamics that the paper shares with its first submission (the input
data folder equilibria_submitted/, see ../../README.md). The CSVs are
the authoritative outputs; this folder documents and preserves the code
that produced them. Producing them took days of CPU time, and the scripts
are not meant to be re-run.

The scripts are unchanged copies of the code that was run. They read the
network edge lists from, and write the CSVs into, the folder they are run
from. ../../s07_variability/gen_hetero.py and
../../s09_sentinel_count/gen_ba.py import several of them and point that
folder elsewhere.

Data model
----------
  - 14 networks (tags): dolphin, prox, metabolic, gkk, lfr, er, ba, hk,
    euroroad, email, flybi, reactome, route_views, spanish.
  - 16 dynamics (stems): sis, sir, msis, msir, doublewell, mutualistic,
    genereg, wc, glv, noymeir, eutroph, vegturb, and the parameter-2
    variants msis2, msir2, eutroph2, vegturb2.
  - Output file: <tag>_<stem>_equilibrium_data.csv. Header row
    `p,x_1,...,x_N`; M = 51 rows (one per swept control-parameter value).
  - The manuscript display names MSIS1/MSIS2 etc. correspond to the
    unsuffixed stem (=*1) and the *2-suffixed stem. So `<tag>_msis_...`
    is MSIS1 and `<tag>_msis2_...` is MSIS2.
  - gLV uses an analytical equilibrium, not ODE integration.

----------------------------------------------------------------
Why so many scripts (the layering)
----------------------------------------------------------------
The current CSVs are a LAYERED result. A base pass wrote all dynamics,
then per-dynamics "regenerate" passes overwrote individual stems with
refined sweep ranges / warm-start policies, and finally two late passes
(gLV eigenvalue rule, WC <k>-rescaling for v89) overwrote those two
stems again. The live producer of each (tag, stem) CSV is whichever
script wrote it LAST; the table at the bottom records that.

----------------------------------------------------------------
LIVE producer scripts (in this folder) -- one by one
----------------------------------------------------------------

verify_networks.py
  PREREQUISITE. Loads the networks, stored as *.rds files in the folder
  networks-from-Neil/ (the network collection of our previous study,
  not distributed here), builds an undirected graph without self-loops or
  duplicate edges, extracts the largest connected component, relabels the
  nodes 0..N-1 in sorted order of their original IDs, and writes
  <tag>_network.edgelist for all 14 networks (plus a verification CSV).
  All other scripts read those edgelists.

generate_data.py
  Base generator: `python3 generate_data.py <tag>` produces all 12 base
  dynamics on one network with the auto-trim protocol, plus a
  <tag>_bifurcation_ranges.csv summary. This is the foundation pass and
  remains the LIVE producer for several stems on the two largest
  networks (route_views, spanish) that the v3 regenerate scripts below
  do not cover.

generate_msis2_msir2.py
  Parameter-2 metapopulation variants MSIS2 (D_I=0.2) and MSIR2 (D_I=0.2)
  for 13 networks (the 11 small nets + reactome + route_views). MSIS2
  reverse-warm, MSIR2 cold-start, adaptive beta_max widening. Imports
  run_all_12_dynamics_v9.py helpers. (route_views IS included here;
  spanish is NOT -- spanish MSIS2/MSIR2 come from spanish_param_variants.py.)

generate_eut2_veg2.py
  Parameter-2 ecological variants EUT2 (Hill p=2) and VEG2 (Hill p=1)
  for 13 networks (the 11 small nets + reactome + route_views). Forward
  warm-start (sweep_forward_warm); has a sanity guard (flags DOUBTFUL and
  writes nothing if eq < -1 or > 100). (spanish EUT2/VEG2 come from
  spanish_param_variants.py.)

spanish_param_variants.py
  LIVE producer of all four parameter-2 variants (MSIS2, MSIR2, EUT2,
  VEG2) for the spanish network only. Reuses the generate_msis2_msir2.py
  and generate_eut2_veg2.py helpers, restricted to tag="spanish".

run_all_12_dynamics_v9.py
  HELPER MODULE (not a standalone entry point): RHS factories, Jacobian
  sparsity patterns, and sweep drivers used by the two generate_*2*
  scripts above. Encodes the LSODA (rtol 1e-7 / atol 1e-9, small nets)
  vs BDF+sparse-Jacobian (rtol 1e-4 / atol 1e-6, large/stiff) solver
  tiers.

regenerate_sis_dw_v3.py
  Regenerates SIS and DW for the 12 smaller networks (incl. reactome)
  under the v3 per-network rescaled ranges (SIS: beta_max =
  1/(0.15 alpha_max); DW: D_max <k> held constant). Cold-start, LSODA.
  (It overwrites the prior CSVs in place; it does NOT write a backup.)

regenerate_reactome_v3.py
  Thin wrapper that runs the regenerate_sis_dw_v3 protocol for reactome
  only (split out to avoid duplicate work during parallel runs).

regenerate_sir_v3.py
  SIR for the 12 smaller networks, cold-start, t_final=400.

regenerate_msis_v3.py
  MSIS for the 12 smaller networks. Reverse warm-start: at the largest
  beta the IC is rho_S=0.9, rho_I=0.1, then the solution is tracked
  backward to smaller beta. beta_max is set by the adaptive widening rule
  (Methods sec:autotrim): start at beta_max=1/(0.15 alpha_max), probe at
  beta_max/2 from the reverse-warm IC, and double beta_max until the mean
  rho_I at the probe reaches THRESHOLD=0.1. NOTE: an earlier shipped copy
  of this script had dropped the widening step (it ran at a fixed
  SIS-matched beta_max); the step has been restored here with THRESHOLD=0.1
  so the script reproduces the MSIS CSVs, which were produced with 0.1
  widening.

regenerate_msir_v3.py
  MSIR for the 12 smaller networks, cold-start with adaptive beta_max
  widening (accept when mean rho_R >= 0.1 at the half-max probe;
  THRESHOLD updated from 0.02 to 0.1 to match the on-disk CSVs and the
  Methods, sec:autotrim).

regenerate_mut_gen_wc_v3.py
  MUT, GEN, WC for the 12 smaller networks (incl. reactome), under the v3
  per-network range rescaling. Cold-start with fixed initial conditions
  at every sweep point (MUT: 0.001, the low-Allee branch; GEN: 2.0;
  WC: 0.1) -- NOT warm-start. It writes ONLY MUT/GEN/WC; the diffusive
  ecological dynamics NM/EUT/VEG are NOT produced here -- they come from
  the base generate_data.py pass (and are not overwritten for the small
  networks).

regenerate_gen_revwarm.py
  GEN reverse-warm-start extended to ALL 14 networks; the LIVE producer
  of GEN for route_views and spanish (the smaller-network GEN comes from
  regenerate_mut_gen_wc_v3.py, except reactome -- see next).

regenerate_reactome_gen_revwarm.py
  GEN on reactome, reverse warm-start. LIVE producer of reactome GEN.

regenerate_glv_eps.py
  gLV for all 14 networks via the analytical coexistence equilibrium
  with the leading-eigenvalue-proximity sweep rule
  (D in [0.05 D_c, D_c - 0.5/alpha_max]). LIVE producer of every gLV CSV.

regenerate_wc_kbar_v89.py
  THE v89 WC RE-CALIBRATION. Regenerates WC on all 14 networks so that
  D_max <k> (not D_max alpha_max) is held constant across networks
  (constant = 6 * 5.13 = 30.77, calibrated on dolphin). This is the most
  recent equilibrium change and is the LIVE producer of every WC CSV.

----------------------------------------------------------------
Run order to reproduce all CSVs from scratch
----------------------------------------------------------------
  1. python3 verify_networks.py                 # all 14 *_network.edgelist
  2. for tag in <14 tags>: python3 generate_data.py $tag   # base 12 dynamics
  3. python3 generate_msis2_msir2.py            # MSIS2, MSIR2 (13 nets, no spanish)
     python3 generate_eut2_veg2.py              # EUT2, VEG2 (13 nets, no spanish)
     python3 spanish_param_variants.py          # MSIS2/MSIR2/EUT2/VEG2 (spanish only)
  4. python3 regenerate_sis_dw_v3.py            # SIS, DW (12 small) + reactome via
     python3 regenerate_reactome_v3.py          #   the wrapper
     python3 regenerate_sir_v3.py               # SIR (12 small)
     python3 regenerate_msis_v3.py              # MSIS (12 small)
     python3 regenerate_msir_v3.py              # MSIR (12 small)
     python3 regenerate_mut_gen_wc_v3.py        # MUT/GEN/WC/NM/EUT/VEG (12 small)
  5. python3 regenerate_gen_revwarm.py          # GEN on route_views, spanish (14-net)
     python3 regenerate_reactome_gen_revwarm.py # GEN on reactome
  6. python3 regenerate_glv_eps.py              # gLV (all 14)
  7. python3 regenerate_wc_kbar_v89.py          # WC (all 14)  <-- last, v89

----------------------------------------------------------------
Per (tag, stem) live producer
----------------------------------------------------------------
Legend: v3=*_v3.py | base=generate_data.py | genrw=regenerate_gen_revwarm.py
        rxgenrw=regenerate_reactome_gen_revwarm.py | glv=regenerate_glv_eps.py
        wc89=regenerate_wc_kbar_v89.py | m2=generate_msis2_msir2.py
        e2=generate_eut2_veg2.py | spv=spanish_param_variants.py
        | (-) not generated

  stem        | 12 small nets        | reactome   | route_views | spanish
  ------------|----------------------|------------|-------------|--------
  sis         | v3                   | v3         | base        | base
  doublewell  | v3                   | v3         | base        | base
  sir         | v3                   | base       | base        | base
  msis        | v3                   | v3/base    | base        | base
  msir        | v3                   | v3/base    | base        | base
  mutualistic | v3                   | base       | base        | base
  genereg     | v3                   | rxgenrw    | genrw       | genrw
  wc          | wc89                 | wc89       | wc89        | wc89
  glv         | glv                  | glv        | glv         | glv
  noymeir     | base                 | base       | base        | base
  eutroph     | base                 | base       | base        | base
  vegturb     | base                 | base       | base        | base
  msis2       | m2                   | m2         | m2          | spv
  msir2       | m2                   | m2         | m2          | spv
  eutroph2    | e2                   | e2         | e2          | spv
  vegturb2    | e2                   | e2         | e2          | spv

(The "12 small nets" column = dolphin, prox, metabolic, gkk, lfr, er,
ba, hk, euroroad, email, flybi -- plus reactome where the v3 scripts
cover it. Entries marked v3/base or base for the large nets reflect that
generate_data.py is the last writer there because the v3 regenerate
scripts target the 12 smaller networks. This table is a best-effort
reconstruction from script scope and CSV modification times; if you need
certainty for a specific cell, check the mtime of that CSV against the
scripts above.)
