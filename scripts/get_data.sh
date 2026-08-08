#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
# Downloads private Kaggle dataset and organizes into data/raw/
set -euo pipefail

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  export $(cat "${PROJECT_ROOT}/.env" | grep -v "^#" | xargs)
fi

KAGGLE_DATASET="haha34/agri-bot"
DATA_DIR="data"
DOWNLOAD_DIR="${DATA_DIR}/.download"
ZIP_PATH="${DOWNLOAD_DIR}/dataset.zip"
EXPECTED_TOP_FOLDERS=("blank" "pages" "raw" "ground-truth-ocr")
EXPECTED_SUBFOLDERS=("Bharater-Krishi-Babyasthar-Parichay" "Krishi-Bigyan" "Krishi-Darpan")

log() { echo "[get_data] $*"; }
die() { echo "[get_data] ERROR: $*" >&2; exit 1; }

command -v kaggle >/dev/null 2>&1 || die "kaggle CLI not found. Install with: pip install kaggle"

has_env_creds=false
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_API_KEY:-}" ]]; then
  has_env_creds=true
elif [[ -n "${KAGGLE_USERNAME:-}" || -n "${KAGGLE_API_KEY:-}" ]]; then
  die "Only one of KAGGLE_USERNAME / KAGGLE_API_KEY is set — both are required together."
fi

if ! ${has_env_creds}; then
  die "No Kaggle credentials found in .env. See .env.example for setup."
fi

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
mkdir -p "${DOWNLOAD_DIR}"

if [[ -f "${ZIP_PATH}" ]]; then
  log "Zip already present at ${ZIP_PATH}, skipping download."
else
  log "Downloading ${KAGGLE_DATASET} from Kaggle..."
  kaggle datasets download -d "${KAGGLE_DATASET}" -p "${DOWNLOAD_DIR}" --force
  # Kaggle names the file after the slug; normalize to a known name.
  DOWNLOADED_ZIP=$(find "${DOWNLOAD_DIR}" -maxdepth 1 -name "*.zip" | head -n 1)
  [[ -n "${DOWNLOADED_ZIP}" ]] || die "Download finished but no .zip file was found in ${DOWNLOAD_DIR}."
  mv "${DOWNLOADED_ZIP}" "${ZIP_PATH}"
fi

# ---------------------------------------------------------------------------
# 2. Extract
# ---------------------------------------------------------------------------
log "Extracting archive..."
mkdir -p "${DATA_DIR}"
unzip -o -q "${ZIP_PATH}" -d "${DATA_DIR}"

for wrapper in "${DATA_DIR}"/*/; do
  wrapper="${wrapper%/}"
  wrapper_name=$(basename "${wrapper}")
  [[ "${wrapper_name}" == ".download" ]] && continue

  is_wrapper=false
  for f in "${EXPECTED_TOP_FOLDERS[@]}"; do
    [[ -d "${wrapper}/${f}" ]] && is_wrapper=true && break
  done

  if ${is_wrapper}; then
    log "Flattening wrapper directory: ${wrapper}"
    rsync -a "${wrapper}/" "${DATA_DIR}/"
    rm -rf "${wrapper}"
  fi
done

rm -rf "${DOWNLOAD_DIR}"

log "Done. Dataset is ready under ${DATA_DIR}/"
