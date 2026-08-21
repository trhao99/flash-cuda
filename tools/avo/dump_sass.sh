#!/usr/bin/env bash
# dump_sass.sh <so> <function_name_regex> <outfile>  -- extract one kernel's SASS (mangled-name match).
SO="$1"; RE="$2"; OUT="${3:-/tmp/avo/sass_out.txt}"
cuobjdump -sass "$SO" 2>/dev/null | awk -v re="$RE" '
  /Function :/ { inf = ($0 ~ re) }
  inf { print }' > "$OUT"
echo "wrote $(grep -cE "/\*[0-9a-f]+\*/" "$OUT") SASS instrs -> $OUT"
