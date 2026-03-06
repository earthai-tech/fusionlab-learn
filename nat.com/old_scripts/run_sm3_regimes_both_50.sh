#!/usr/bin/env bash
set -euo pipefail

# SM3 synthetic identifiability — 5 regimes, 50 realisations each
# identify=both (harder; requires spatial signal)
#
# Outputs:
#   results/sm3_both_suite_<timestamp>/
#     sm3_both_<reg>_50/   (each has sm3_synth_runs.csv + sm3_synth_summary.csv)
#     logs/<reg>.log
#     combined/sm3_summary_combined.csv
#     combined/sm3_summary_combined.json

TS="$(date +%Y%m%d-%H%M%S)"
SUITE_ROOT="results/sm3_both_suite_${TS}"
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
K_SPREAD="0.6"

ALPHA="1.0"
HD_FACTOR="0.6"
THICK_CAP="30.0"
KAPPA_B="1.0"
GAMMA_W="9810.0"

SCENARIO="base"

# 1D domain settings 
NX=21
LX_M=5000
H_RIGHT="0.0"

echo "============================================================"
echo "[SM3] suite root: ${SUITE_ROOT}"
echo "============================================================"
echo "[SM3] 1D domain: nx=${NX}  Lx_m=${LX_M}  h_right=${H_RIGHT}"
echo "============================================================"

for reg in none base anchored closure_locked data_relaxed; do
  OUTDIR="${SUITE_ROOT}/sm3_both_${reg}_50"
  LOGFILE="${LOGDIR}/${reg}.log"

  echo "============================================================"
  echo "[RUN] ident=both  regime=${reg}"
  echo "      outdir=${OUTDIR}"
  echo "      log=${LOGFILE}"
  echo "============================================================"

  python nat.com/sm3_synthetic_identifiability.py \
    --outdir "${OUTDIR}" \
    --n-realizations "${NREAL}" \
    --identify both \
    --ident-regime "${reg}" \
    --scenario "${SCENARIO}" \
    --nx "${NX}" \
    --Lx-m "${LX_M}" \
    --h-right "${H_RIGHT}" \
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
    --K-spread-dex "${K_SPREAD}" \
    --alpha "${ALPHA}" \
    --hd-factor "${HD_FACTOR}" \
    --thickness-cap "${THICK_CAP}" \
    --kappa-b "${KAPPA_B}" \
    --gamma-w "${GAMMA_W}" \
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