#!/bin/bash

bash CompileLaTeX.sh final_project.tex

bash CompileLaTeX2.sh appendices.tex

gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=merged_project.pdf final_project.pdf appendices.pdf
