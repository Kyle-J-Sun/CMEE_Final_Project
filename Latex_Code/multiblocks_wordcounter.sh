#!/bin/bash
name=${1%.*}
wc=0
for v in $(basename ./blocks/*.tex); do
  if [ $v == "introduction.tex" -o $v == "methods.tex" -o $v == "results.tex" -o $v == "discussion.tex" -o $v == "conclusion.tex" ]
  then
    num=$(texcount -1 -sum ./blocks/$v)
    wc=$(($wc+num))
  else
    continue
  fi
done

echo $wc > $name.sum

echo "Total word count has been saved"
