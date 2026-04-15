#!/bin/bash
set -e
cd /app/torch2trt_src && python3 setup.py install
cd /app/whisper_trt_src && python3 setup.py install
exec "$@"
