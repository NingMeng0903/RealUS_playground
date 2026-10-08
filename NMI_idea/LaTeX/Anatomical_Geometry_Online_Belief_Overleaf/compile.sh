#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
support_dir="${HOME}/.local/share/latex-support/texlive2021"
if [[ -d "${support_dir}/texmf" ]]; then
  export PATH="${support_dir}/bin:${PATH}"
  export TEXMFHOME="{${support_dir}/texmf,${HOME}/texmf}"
  export TEXFORMATS="${support_dir}/var/web2c//:"
fi
mkdir -p build
latexmk -xelatex -outdir=build -halt-on-error "$@" main.tex
