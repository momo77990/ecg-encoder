#!/usr/bin/env bash
# 把项目根目录下的 archive.zip 解压到 data/。
# archive.zip 来源: Kaggle https://www.kaggle.com/datasets/shayanfazeli/heartbeat
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$PROJECT_ROOT/archive.zip"
DEST="$PROJECT_ROOT/data"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: $ARCHIVE not found." >&2
  exit 1
fi

mkdir -p "$DEST"

# -n: 不覆盖已有文件（幂等）
unzip -n "$ARCHIVE" -d "$DEST"

echo "Done. Files in $DEST:"
ls -lh "$DEST"
