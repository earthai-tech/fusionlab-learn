#!/usr/bin/env bash
set -euo pipefail

# SM3 synthetic identifiability — 5 regimes, 50 realisations each
# identify=tau (recommended)
#
# Outputs:
#   results/sm3_tau_suite_<timestamp>/
#     sm3_tau_<reg>_50/   (each has sm3_synth_runs.csv + sm3_synth_summary.csv)
#     logs/<reg>.log
#     combined/sm3_summary_combined.csv
#     combined/sm3_summary_combined.json

TS="$(date +%Y%m%d-%H%M%S)"
SUITE_ROOT="results/sm3_tau_suite_${TS}"
LOGDIR="${SUITE_ROOT}/logs"
COMBDIR="${SUITE_ROOT}/combined"

mkdir -p "${SUITE_ROOT}" "${LOGDIR}" "${COMBDIR}"

NREAL=50
SEED=123

# Shared "strong" settings
NYEARS=25
TSTEPS=5
FH=3
VALTAIL=5

EPOCHS=80
BATCH=16
LR="1e-3"

NOISE="0.02"
LOAD="step"

TAU_MIN="0.3"
TAU_MAX="10.0"
TAU_SPREAD="0.35"
SS_SPREAD="0.45"

ALPHA="1.0"
HD_FACTOR="0.6"
THICK_CAP="30.0"
KAPPA_B="1.0"
GAMMA_W="9810.0"

SCENARIO="base"   # or: tau_only_derive_k

echo "============================================================"
echo "[SM3] suite root: ${SUITE_ROOT}"
echo "============================================================"

for reg in none base anchored closure_locked data_relaxed; do
  OUTDIR="${SUITE_ROOT}/sm3_tau_${reg}_50"
  LOGFILE="${LOGDIR}/${reg}.log"

  echo "============================================================"
  echo "[RUN] ident=tau  regime=${reg}"
  echo "      outdir=${OUTDIR}"
  echo "      log=${LOGFILE}"
  echo "============================================================"

  # Use 'tee' so you get console output AND a saved log
  python nat.com/sm3_synthetic_identifiability.py \
    --outdir "${OUTDIR}" \
    --n-realizations "${NREAL}" \
    --identify tau \
    --ident-regime "${reg}" \
    --scenario "${SCENARIO}" \
    --n-years "${NYEARS}" \
    --time-steps "${TSTEPS}" \
    --forecast-horizon "${FH}" \
    --val-tail "${VALTAIL}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --noise-std "${NOISE}" \
    --load-type "${LOAD}" \
    --seed "${SEED}" \
    --tau-min "${TAU_MIN}" \
    --tau-max "${TAU_MAX}" \
    --tau-spread-dex "${TAU_SPREAD}" \
    --Ss-spread-dex "${SS_SPREAD}" \
    --alpha "${ALPHA}" \
    --hd-factor "${HD_FACTOR}" \
    --thickness-cap "${THICK_CAP}" \
    --kappa-b "${KAPPA_B}" \
    --gamma-w "${GAMMA_W}" \
    --nx 21 --Lx-m 5000 --h-right 0.0 \
    --start-realisation 1 \
    2>&1 | tee "${LOGFILE}"
done

echo "============================================================"
echo "[COLLECT] building combined summary table..."
echo "============================================================"

python nat.com/sm3_collect_summaries.py \
  --suite-root "${SUITE_ROOT}" \
  --out-csv "${COMBDIR}/sm3_summary_combined.csv" \
  --out-json "${COMBDIR}/sm3_summary_combined.json"

echo " Suite completed."
echo "   Combined CSV:  ${COMBDIR}/sm3_summary_combined.csv"
echo "   Combined JSON: ${COMBDIR}/sm3_summary_combined.json"