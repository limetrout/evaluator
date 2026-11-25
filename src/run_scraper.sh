#!/bin/bash
# Keeps the scraper alive: restarts it if it exits early or makes no progress for 8 minutes.
cd "$(dirname "$0")/.."
while true; do
  python3 src/01_scrape_mtgo.py 1 >> data/scrape.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    n=$(ls data/raw/mtgo | wc -l); s=$(stat -c %Y data/scrape.log data/raw/mtgo_index.json | sort -n | tail -1)
    if [ $(( $(date +%s) - s )) -gt 900 ] && [ "$n" = "${last:-x}" ]; then echo "watchdog: restarting (stalled)" >> data/scrape.log; kill $pid; fi
    last=$n
  done
  grep -q "^done" <(tail -1 data/scrape.log) && break
done
echo "ALL DONE" >> data/scrape.log
