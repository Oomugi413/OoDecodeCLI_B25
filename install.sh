#!/usr/bin/env bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
data_dir=${XDG_DATA_HOME:-"$HOME/.local/share"}
applications_dir="$data_dir/applications"
desktop_path="$applications_dir/OoDecodeCLI_B25.desktop"

mkdir -p "$applications_dir"

# The repository path is substituted into the launcher so file managers can
# pass dropped files through the desktop-entry %F field.
sed "s|@PROJECT_DIR@|$script_dir|g" \
  "$script_dir/OoDecodeCLI_B25.desktop.in" > "$desktop_path"
chmod 0644 "$desktop_path"

echo "インストールしました: $desktop_path"
echo "アプリ一覧の更新が必要な環境では、ログアウト/ログイン後に確認してください。"
