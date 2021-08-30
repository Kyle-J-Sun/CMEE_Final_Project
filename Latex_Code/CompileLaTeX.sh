#!/bin/bash

# to remove the extensions from the input file
name=${1%.*}

bash multiblocks_wordcounter.sh final_project.tex

pdflatex $1
pdflatex $1
bibtex $name
pdflatex $1
pdflatex $1
# evince $name.pdf &

## Cleanup
rm *~
rm *.aux
rm *.dvi
rm *.log
rm *.nav
rm *.out
rm *.snm
rm *.toc
rm *.bbl
rm *.blg
rm *.fls
rm *.fdb_latexmk
rm *.run.xml
rm *.gz
rm *.bcf
rm .DS_Store
rm *.lof
# rm *.sum
