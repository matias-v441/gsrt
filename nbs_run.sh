#!/usr/bin/sh
export NERFBASELINES_REGISTER="$PWD/nbs_method/gsrt_method_spec.py"
make && python nbs_viewer.py $1