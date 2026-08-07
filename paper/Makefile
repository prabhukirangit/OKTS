# Build the OKTS preprint.
#
#   make results   # regenerate all numbers/tables/figures from the eval harness
#   make pdf       # compile main.tex -> main.pdf (needs a local TeX engine)
#   make all       # results, then pdf
#   make clean     # remove LaTeX build artifacts (keeps results/ and main.pdf)
#
# No local TeX engine? The sources are self-contained and compile as-is on
# Overleaf or arXiv's own build. See SUBMISSION.md.

PYTHON ?= python
LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)

.PHONY: all results pdf clean

all: results pdf

results:
	$(PYTHON) make_results.py

pdf:
ifneq ($(LATEXMK),)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else ifneq ($(PDFLATEX),)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main || true
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
else
	@echo "No local TeX engine (latexmk/pdflatex) found."
	@echo "Upload main.tex + refs.bib + results/ to Overleaf or arXiv, which compile it."
	@echo "See SUBMISSION.md for the step-by-step."
	@exit 1
endif

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.fdb_latexmk main.fls main.synctex.gz
