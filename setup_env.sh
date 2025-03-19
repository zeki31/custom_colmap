#!/bin/bash

set -e

# このスクリプトの絶対パスを取得
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# .venv/bin/activate ファイルが存在しない場合はエラー終了
if [[ ! -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    echo "Error: Initialize uv and venv first." >&2
    exit 1
fi

PYTHONPATH=$PROJECT_DIR

# 一時ファイルを作成して置換を実行
update_file() {
    local file=$1
    local tmp_file="${file}.tmp"
    if grep -q "^export PYTHONPATH=" "$file"; then
        awk -v path="$PYTHONPATH" '{sub(/^export PYTHONPATH=.*/, "export PYTHONPATH=" path)}1' "$file" > "$tmp_file"
        mv "$tmp_file" "$file"
        echo "Updated PYTHONPATH in $file"
    else
        echo "export PYTHONPATH=$PYTHONPATH" >> "$file"
        echo "Set PYTHONPATH to $file"
    fi
}

# .env に対して適用
touch "$PROJECT_DIR/.env"
update_file "$PROJECT_DIR/.env"

# .venv/bin/activate に対して適用
update_file "$PROJECT_DIR/.venv/bin/activate"
