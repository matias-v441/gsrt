#!/usr/bin/sh
# export NERFBASELINES_REGISTER="$PWD/nerfbaselines/gsrt_method_spec.py"
make && python nbs_viewer.py $1