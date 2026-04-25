# FDST Dataset Download & Verification Guide

## Overview

This guide walks you through downloading and verifying the **FDST (Fudan-ShanghaiTech) Dataset** using `rclone` from Google Drive. The dataset is split into `train_data` and `test_data` folders.

---

## Prerequisites

### 1. Install rclone (no sudo required)

```bash
curl https://rclone.org/install.sh | bash
```

If the script fails due to permissions, install manually:

```bash
curl -L "https://downloads.rclone.org/rclone-current-linux-amd64.zip" -o rclone.zip
unzip rclone.zip
mkdir -p ~/.local/bin
cp rclone-*-linux-amd64/rclone ~/.local/bin/
export PATH="$PATH:$HOME/.local/bin"
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:

```bash
rclone version
```

---

### 2. Configure rclone with Google Drive

```bash
rclone config
```

Follow these prompts:

| Prompt | Response |
|--------|----------|
| New remote | `n` |
| Name | `gdrive` |
| Storage type | `drive` |
| Client ID | *(leave blank, press Enter)* |
| Client Secret | *(leave blank, press Enter)* |
| Scope | `1` (full access) |
| Root folder ID | *(leave blank, press Enter)* |
| Service account | *(leave blank, press Enter)* |
| Advanced config | `n` |
| Auto config | `n` *(since you are on a remote server)* |

When prompted, run the following **on your local machine** (with a browser):

```bash
rclone authorize "drive" "<token-shown-on-screen>"
```

Paste the resulting token back into the server terminal. Then:

| Prompt | Response |
|--------|----------|
| Shared Drive | `n` |
| Confirm | `y` |
| Quit | `q` |

---

## Set Your Base Path

Set the root directory where the dataset will be downloaded. Replace with your desired path:

```bash
export FDST_ROOT="/your/path/here"
# Example:
# export FDST_ROOT="/scratch/username/FDST_Data"

mkdir -p "$FDST_ROOT/train_data"
mkdir -p "$FDST_ROOT/test_data"
```

---

## Download Dataset

### Download Train Data

```bash
nohup rclone copy gdrive:train_data "$FDST_ROOT/train_data" \
  --drive-root-folder-id 1zFjDGr9dpqjZ9ZPsXKbHeBUR7xbbDvSN \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --buffer-size 64M \
  --retries 10 \
  --retries-sleep 30s \
  --drive-pacer-min-sleep 100ms \
  --exclude ".DS_Store" \
  --exclude "._*" \
  -v > "$FDST_ROOT/rclone_train.log" 2>&1 &

echo "Train download PID: $!"
```

Monitor progress:

```bash
tail -f "$FDST_ROOT/rclone_train.log"
```

---

### Download Test Data

```bash
nohup rclone copy gdrive:test_data "$FDST_ROOT/test_data" \
  --drive-root-folder-id 1F9Tgj_sPBO8yEdL1SKjX3siNUsy3TlcM \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --buffer-size 64M \
  --retries 10 \
  --retries-sleep 30s \
  --drive-pacer-min-sleep 100ms \
  --exclude ".DS_Store" \
  --exclude "._*" \
  -v > "$FDST_ROOT/rclone_test.log" 2>&1 &

echo "Test download PID: $!"
```

Monitor progress:

```bash
tail -f "$FDST_ROOT/rclone_test.log"
```

---

## Verify Dataset

Once downloads complete, run this verification script to compare local files against remote and detect missing, partial, or extra files.

```bash
cat << 'EOF' > "$FDST_ROOT/verify.sh"
#!/bin/bash

BASE="$FDST_ROOT"
TRAIN_ID="1zFjDGr9dpqjZ9ZPsXKbHeBUR7xbbDvSN"
TEST_ID="1F9Tgj_sPBO8yEdL1SKjX3siNUsy3TlcM"

verify_split() {
    local SPLIT=$1
    local FOLDER_ID=$2
    local SPLIT_DIR="$BASE/${SPLIT}_data"

    echo ""
    echo "=============================="
    echo " Verifying: $SPLIT data"
    echo "=============================="

    # Fetch remote file list (excluding Mac metadata)
    rclone ls gdrive: --drive-root-folder-id "$FOLDER_ID" \
        | grep -v ".DS_Store" \
        | grep -v " \._" \
        > "/tmp/remote_${SPLIT}.txt"

    echo "Remote files: $(wc -l < /tmp/remote_${SPLIT}.txt)"
    echo "Local files:  $(find "$SPLIT_DIR" -type f | grep -v ".partial" | wc -l)"
    echo ""

    ok=0; incomplete=0; partial_folders=0

    for folder in "$SPLIT_DIR"/*/; do
        folder_name=$(basename "$folder")
        local_count=$(ls "$folder" | grep -v ".partial" | wc -l)
        remote_count=$(grep " ${folder_name}/" "/tmp/remote_${SPLIT}.txt" | wc -l)
        partial_count=$(find "$folder" -name "*.partial" | wc -l)

        if [ "$partial_count" -gt 0 ]; then
            echo "  HAS PARTIAL: $folder_name (local=$local_count, remote=$remote_count, partials=$partial_count)"
            ((partial_folders++))
        elif [ "$local_count" -lt "$remote_count" ]; then
            echo "  INCOMPLETE:  $folder_name (local=$local_count, remote=$remote_count)"
            ((incomplete++))
        else
            ((ok++))
        fi
    done

    echo ""
    echo "  Complete folders:          $ok"
    echo "  Incomplete folders:        $incomplete"
    echo "  Folders with .partial:     $partial_folders"
}

verify_split "train" "$TRAIN_ID"
verify_split "test"  "$TEST_ID"

echo ""
echo "=============================="
echo " SUMMARY"
echo "=============================="
echo "Train folders: $(ls $BASE/train_data | wc -l)"
echo "Test folders:  $(ls $BASE/test_data | wc -l)"
echo "Total files:   $(find $BASE/train_data $BASE/test_data -type f | grep -v '.partial' | wc -l)"
EOF

chmod +x "$FDST_ROOT/verify.sh"
bash "$FDST_ROOT/verify.sh"
```

---

## Re-download Incomplete Folders

If the verify script reports incomplete or partial folders, redownload them:

```bash
# Replace SPLIT with 'train' or 'test', FOLDER_NAME with folder number, FOLDER_ID with the relevant ID
rclone copy gdrive:FOLDER_NAME "$FDST_ROOT/SPLIT_data/FOLDER_NAME" \
  --drive-root-folder-id FOLDER_ID \
  --transfers 4 \
  --checkers 8 \
  --size-only \
  --retries 10 \
  --retries-sleep 30s \
  --exclude ".DS_Store" \
  --exclude "._*" \
  -v
```

To completely redownload a folder:

```bash
rm -rf "$FDST_ROOT/SPLIT_data/FOLDER_NAME"
# Then rerun the rclone copy command above
```

---

## Notes

- `.DS_Store` and `._*` files are Mac metadata files uploaded by the dataset creator — they are automatically excluded and can be safely ignored.
- `*.partial` files are incomplete downloads — delete them and redownload the folder.
- If your server session disconnects during download, just rerun the same `nohup rclone copy` command — rclone will skip already-downloaded files automatically.
- Keep downloads to `--transfers 4` to avoid Google Drive rate limiting.