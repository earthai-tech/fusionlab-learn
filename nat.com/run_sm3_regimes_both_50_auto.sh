#!/usr/bin/env bash
set -euo pipefail

# SM3 synthetic identifiability — selectable regimes
# identify=both (harder; needs spatial signal)

usage() {
  cat <<'USAGE'
Usage: run_sm3_regimes_both_50_auto.sh [options]

Options:
  --device {auto|cpu|gpu}   Device selection (default: auto)
  --epochs N                Training epochs (default: 40)
  --batch N                 Batch size (default: 64)
  --patience N              EarlyStopping patience (default: 5)
  --fast {0|1}              Disable heavy diagnostics callbacks (default: 1)

  --disable-freeze          Alias for --fast 1
  --enable-freeze           Alias for --fast 0

  --nreal N                 Number of realisations (default: 50)
  --seed N                  Base seed (default: 123)

  --regime NAME             Run only this regime (repeatable)
  --regimes CSV             Run regimes from comma list (e.g., base,anchored)
  --regime-ids CSV          Run regimes by 1-based ids (e.g., 2,4)
  --list-regimes            Print regime ids + names and exit

  --suite-root PATH          Use this suite root (enables resume)
  --resume-latest            Resume the most recent suite
  --start-realisation N      1-based start index to pass to python (default: 1)
  
  --help                    Show this help

Env overrides:
  DEVICE EPOCHS BATCH PATIENCE FAST NREAL SEED
  REGIMES (csv) REGIME_IDS (csv)
  SUITE_ROOT RESUME_LATEST START_REALISATION

CPU threading env:
  NTHREADS TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS
USAGE
}

# ------------------------------
# Regime catalog (1-based ids)
# ------------------------------
ALL_REGIMES=("none" "base" "anchored" "closure_locked" "data_relaxed")

list_regimes() {
  echo "Available regimes:"
  local i
  for i in "${!ALL_REGIMES[@]}"; do
    echo "  $((i+1))  ${ALL_REGIMES[$i]}"
  done
}

is_valid_regime() {
  local r="$1"
  local x
  for x in "${ALL_REGIMES[@]}"; do
    if [[ "$x" == "$r" ]]; then
      return 0
    fi
  done
  return 1
}

# ------------------------------
# Defaults (can be overridden)
# ------------------------------
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-64}"
PATIENCE="${PATIENCE:-5}"
FAST="${FAST:-1}"
NREAL="${NREAL:-50}"
SEED="${SEED:-123}"

# optional env-driven regime selection
REGIMES_CSV="${REGIMES:-}"
REGIME_IDS_CSV="${REGIME_IDS:-}"

SUITE_ROOT_IN="${SUITE_ROOT:-}"
RESUME_LATEST="${RESUME_LATEST:-0}"
START_REALISATION="${START_REALISATION:-1}"

# selection by repeated --regime
SEL_REGIMES=()

# ------------------------------
# Parse CLI args
# ------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --fast) FAST="$2"; shift 2;;

    --disable-freeze) FAST="1"; shift;;
    --enable-freeze) FAST="0"; shift;;

    --nreal) NREAL="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;

    --regime) SEL_REGIMES+=("$2"); shift 2;;
    --regimes) REGIMES_CSV="$2"; shift 2;;
    --regime-ids) REGIME_IDS_CSV="$2"; shift 2;;
    --list-regimes) list_regimes; exit 0;;

    --suite-root) SUITE_ROOT_IN="$2"; shift 2;;
    --resume-latest) RESUME_LATEST="1"; shift;;
    --start-realisation) START_REALISATION="$2"; shift 2;;
    
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

# ------------------------------
# Build final regime runlist
#   Priority:
#     1) repeated --regime
#     2) --regimes CSV / REGIMES env
#     3) --regime-ids CSV / REGIME_IDS env
#     4) default: all
# ------------------------------
RUN_REGIMES=()

# 1) from repeated --regime
if [[ "${#SEL_REGIMES[@]}" -gt 0 ]]; then
  RUN_REGIMES=("${SEL_REGIMES[@]}")
fi

# 2) from names CSV
if [[ "${#RUN_REGIMES[@]}" -eq 0 && -n "${REGIMES_CSV}" ]]; then
  IFS=',' read -r -a _tmp <<< "${REGIMES_CSV}"
  RUN_REGIMES=("${_tmp[@]}")
fi

# 3) from ids CSV
if [[ "${#RUN_REGIMES[@]}" -eq 0 && -n "${REGIME_IDS_CSV}" ]]; then
  IFS=',' read -r -a _ids <<< "${REGIME_IDS_CSV}"
  for id in "${_ids[@]}"; do
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] Non-integer regime id: '$id'" >&2
      list_regimes >&2
      exit 2
    fi
    if [[ "$id" -lt 1 || "$id" -gt "${#ALL_REGIMES[@]}" ]]; then
      echo "[ERROR] Regime id out of range: '$id'" >&2
      list_regimes >&2
      exit 2
    fi
    RUN_REGIMES+=("${ALL_REGIMES[$((id-1))]}")
  done
fi

# 4) default all
if [[ "${#RUN_REGIMES[@]}" -eq 0 ]]; then
  RUN_REGIMES=("${ALL_REGIMES[@]}")
fi

# validate + dedupe while preserving order
DEDUPED=()
declare -A _seen=()
for r in "${RUN_REGIMES[@]}"; do
  if ! is_valid_regime "$r"; then
    echo "[ERROR] Unknown regime: '$r'" >&2
    list_regimes >&2
    exit 2
  fi
  if [[ -z "${_seen[$r]+x}" ]]; then
    _seen[$r]=1
    DEDUPED+=("$r")
  fi
done
RUN_REGIMES=("${DEDUPED[@]}")

# ------------------------------
# Device detection / setup
# ------------------------------
has_gpu=0
gpu_count=0
if [[ "$DEVICE" != "cpu" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
      gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
      if [[ "${gpu_count}" -ge 1 ]]; then
        has_gpu=1
      fi
    fi
  fi
fi

if [[ "$DEVICE" == "gpu" && "$has_gpu" -ne 1 ]]; then
  echo "[ERROR] DEVICE=gpu requested, but no NVIDIA GPU detected." >&2
  exit 3
fi

if [[ "$DEVICE" == "auto" ]]; then
  if [[ "$has_gpu" -eq 1 ]]; then
    DEVICE="gpu"
  else
    DEVICE="cpu"
  fi
fi

if [[ "$DEVICE" == "gpu" ]]; then
  echo "[SM3] Using GPU (detected ${gpu_count})"
  export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
  export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
else
  echo "[SM3] Using CPU"
  export CUDA_VISIBLE_DEVICES=""

  NTHREADS="${NTHREADS:-$(nproc)}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NTHREADS}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NTHREADS}"
  export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$NTHREADS}"
  export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

  echo "[SM3] CPU threads: OMP=${OMP_NUM_THREADS} TF(intra)=${TF_NUM_INTRAOP_THREADS} TF(inter)=${TF_NUM_INTEROP_THREADS}"
fi

SUITE_PREFIX="sm3_both_suite"

if [[ -n "${SUITE_ROOT_IN}" ]]; then
  SUITE_ROOT="${SUITE_ROOT_IN}"
elif [[ "${RESUME_LATEST}" == "1" ]]; then
  SUITE_ROOT="$(ls -1dt results/${SUITE_PREFIX}_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${SUITE_ROOT}" ]]; then
    echo "[ERROR] --resume-latest requested but no existing suite found: results/${SUITE_PREFIX}_*" >&2
    exit 4
  fi
else
  TS="$(date +%Y%m%d-%H%M%S)"
  SUITE_ROOT="results/${SUITE_PREFIX}_${TS}"
fi
LOGDIR="${SUITE_ROOT}/logs"
COMBDIR="${SUITE_ROOT}/combined"

mkdir -p "${SUITE_ROOT}" "${LOGDIR}" "${COMBDIR}"

# Shared settings
NYEARS=25
TSTEPS=5
FH=3
VALTAIL=5

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

EXTRA_FLAGS=()
if [[ "$FAST" == "1" ]]; then
  EXTRA_FLAGS+=("--disable-freeze")
fi

echo "============================================================"
echo "[SM3] suite root: ${SUITE_ROOT}"
echo "[SM3] identify: both"
echo "[SM3] device: ${DEVICE}"
echo "[SM3] regimes: ${RUN_REGIMES[*]}"
echo "[SM3] nreal=${NREAL}  epochs=${EPOCHS}  batch=${BATCH}  patience=${PATIENCE}  fast=${FAST}"
echo "[SM3] 1D domain: nx=${NX}  Lx_m=${LX_M}  h_right=${H_RIGHT}"
echo "============================================================"

for reg in "${RUN_REGIMES[@]}"; do
  OUTDIR="${SUITE_ROOT}/sm3_both_${reg}_${NREAL}"
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
    --patience "${PATIENCE}" \
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
    --start-realisation "${START_REALISATION}" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee -a "${LOGFILE}"
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