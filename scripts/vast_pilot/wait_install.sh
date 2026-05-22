#!/usr/bin/env bash
while pgrep -f 'pip install.*flash_attn' >/dev/null 2>&1; do
  echo still_building_flash
  sleep 90
done
tail -25 /workspace/install.log
python3 -c 'import flash_attn_interface; print("flash ok")'
