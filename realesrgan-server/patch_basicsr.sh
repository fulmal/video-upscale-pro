#!/bin/bash
# Patch basicsr untuk kompatibilitas torchvision >= 0.16
# (mengganti import lama 'functional_tensor' yang sudah dihapus)

DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="$DIR/.venv/lib/python3.9/site-packages/basicsr/data/degradations.py"

if [ ! -f "$TARGET" ]; then
    echo "❌ basicsr tidak ditemukan: $TARGET"
    exit 1
fi

if grep -q "functional_tensor" "$TARGET"; then
    sed -i '' \
      's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
      "$TARGET"
    echo "✅ Patch berhasil: functional_tensor → functional"
else
    echo "✓ Sudah di-patch sebelumnya, tidak perlu diulang"
fi
