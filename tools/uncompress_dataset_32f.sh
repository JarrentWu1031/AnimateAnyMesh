#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-.}"
DEST="${2:-./extracted}"

# Default archive path prefix to strip from extracted file paths
# (can be overridden via --strip-prefix)
STRIP_PREFIX_DEFAULT="mnt/nas_jianchong/wzj/AnimateAnyMesh_v1_Data/obj4d_dt4d_all_5w_merged_filtered_32f_clips_caption_no_pad_f"
STRIP_PREFIX="$STRIP_PREFIX_DEFAULT"

# Optional argument to override the prefix to strip
if [[ "${3:-}" == "--strip-prefix" ]]; then
  STRIP_PREFIX="${4:-$STRIP_PREFIX_DEFAULT}"
fi

# Ensure tar is available
if ! command -v tar >/dev/null 2>&1; then
  echo "Error: tar command not found." >&2
  exit 1
fi

# Detect availability of zstd (or tar -I)
HAVE_ZSTD=0
TAR_HAS_I=0
if command -v zstd >/dev/null 2>&1; then
  HAVE_ZSTD=1
fi
if tar --help 2>/dev/null | grep -qE -- "-I, --use-compress-program"; then
  TAR_HAS_I=1
fi
if [[ $HAVE_ZSTD -eq 0 && $TAR_HAS_I -eq 0 ]]; then
  echo "Error: zstd not found and tar does not support -I. Please install zstd or use GNU tar with -I." >&2
  exit 1
fi

mkdir -p "$DEST"

# Build tar path-stripping options:
# Prefer GNU tar --transform; fallback to --strip-components if unavailable.
TAR_TRANSFORM_ARGS=()
if tar --help 2>/dev/null | grep -q -- "--transform"; then
  # First strip a possible leading "./", then strip the specified prefix
  TAR_TRANSFORM_ARGS=( --transform="s|^\\./||" --transform="s|^${STRIP_PREFIX%/}/||" )
else
  # Fallback: compute component count for --strip-components
  IFS="/" read -r -a _parts <<< "${STRIP_PREFIX%/}"
  STRIP_N=${#_parts[@]}
  TAR_TRANSFORM_ARGS=( "--strip-components=$STRIP_N" )
  echo "Note: current tar does not support --transform; falling back to --strip-components=$STRIP_N." >&2
fi

# Collect all part groups matching *_part_* (strip the _part_ suffix to get bases)
# Support both Linux (GNU find) and environments without -printf.
if find "$SRC" -maxdepth 1 -type f -name "*_part_*" -printf "" >/dev/null 2>&1; then
  mapfile -t BASES < <(
    find "$SRC" -maxdepth 1 -type f -name "*_part_*" -printf "%f\n" \
    | sed -E "s/_part_.*$//" | sort -u
  )
else
  mapfile -t BASES < <(
    ls -1 "$SRC" 2>/dev/null | grep -E "_part_[^/]+$" | sed -E "s/_part_.*$//" | sort -u
  )
fi

if ((${#BASES[@]} == 0)); then
  echo "No *_part_* split files found under: $SRC"
  exit 0
fi

echo "Searching split archives in: $SRC"
echo "Extracting into: $DEST"
echo "Will strip archive path prefix: ${STRIP_PREFIX%/}/"
echo

for base in "${BASES[@]}"; do
  # Find all parts for this base and sort by numeric suffix (00, 01, 02, ...)
  if find "$SRC" -maxdepth 1 -type f -name "${base}_part_*" -printf "" >/dev/null 2>&1; then
    mapfile -t PARTS < <(
      find "$SRC" -maxdepth 1 -type f -name "${base}_part_*" -printf "%p\n" | sort -V
    )
  else
    mapfile -t PARTS < <(ls -1 "$SRC"/"${base}"_part_* 2>/dev/null | sort -V)
  fi

  if ((${#PARTS[@]} == 0)); then
    echo "Skip ${base}: no parts found."
    continue
  fi

  echo "==> Processing part group: ${base}"
  printf "    Parts:\n"
  printf "    - %s\n" "${PARTS[@]}"

  # Streamed extraction:
  # Concatenate parts -> decompress with zstd -> extract with tar (stripping prefix)
  if [[ $HAVE_ZSTD -eq 1 ]]; then
    cat "${PARTS[@]}" | zstd -d --stdout | tar -xvf - -C "$DEST" "${TAR_TRANSFORM_ARGS[@]}"
  else
    cat "${PARTS[@]}" | tar -I zstd -xvf - -C "$DEST" "${TAR_TRANSFORM_ARGS[@]}"
  fi

  echo "==> Done: ${base}"
  echo
done

echo "All extractions completed."