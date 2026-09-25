#!/bin/bash
cd $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
[ -e .venv ] || python3 -m venv .venv;
.venv/bin/python3 -m pip install -r requirements.txt
