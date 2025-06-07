#!/bin/bash

# Define source and destination directories
SRC="/ssd_scratch/pratyush.jena/Aug_LineTR/annotation_tool/images_train/"
DEST="/nas/pratyush.jena/Aug_LineTR/annotation_tool/images_train/"

# Log file
LOGFILE="/ssd_scratch/pratyush.jena/Aug_LineTR/sync_log.txt"

# Time interval in seconds between syncs
INTERVAL=7200

echo "[START] Starting sync loop from $SRC to $DEST" | tee -a "$LOGFILE"

while true
do
    echo "[SYNC] $(date): Running rsync..." | tee -a "$LOGFILE"
    rsync -av --update --progress "$SRC" "$DEST" | tee -a "$LOGFILE"
    echo "[SLEEP] Sleeping for $INTERVAL seconds..." | tee -a "$LOGFILE"
    sleep "$INTERVAL"
done
