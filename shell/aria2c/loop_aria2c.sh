#!/bin/bash
# per https://stackoverflow.com/questions/37295861/aria2c-timeout-error-on-large-file-but-downloads-most-of-it
############################################################
aria2c -j5 -i list.txt -c --save-session out.txt
has_error=`wc -l < out.txt`

while [ $has_error -gt 0 ]
do
    echo "still has $has_error errors, rerun aria2 to download ..."
    aria2c -j5 -i list.txt -c --save-session out.txt
    has_error=`wc -l < out.txt`
    sleep 10
done
