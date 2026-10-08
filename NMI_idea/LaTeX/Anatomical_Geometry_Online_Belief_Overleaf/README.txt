Overleaf instructions
====================
1. Upload this whole folder (or the zip) as a new Overleaf project.
2. Compiler: XeLaTeX. The included latexmkrc also requests XeLaTeX.
3. Main document: main.tex.
4. references.bib and figures/ are required. main.bbl is included as a fallback.

Why the earlier copy failed:
- That source used fontspec and therefore cannot be compiled with pdfLaTeX.
- Copying only main.tex also omits bibliography and figure files.

This Overleaf version uses the TeX-Live Fandol Chinese font set and avoids host-specific Noto/Liberation font declarations.
