\usepackage[top = 0.6 in, bottom = 0.6 in, left = 1 in, right = 1 in]{geometry}

\usepackage{
  graphicx,
  anyfontsize,
  setspace,
  listings,
  xcolor,
  float,
  amsmath, % replace with CJK for sharelatex.com
  subcaption,
  lineno,
  tabularx,
  url,
  hyperref,
  wrapfig
}

\usepackage[nottoc]{tocbibind}
\usepackage[titles]{tocloft}
\setlength\cftparskip{3pt}

\usepackage{verbatim}

\newcommand{\quickwordcount}{%
  \immediate\write18{texcount -1 -sum -merge -q introduction.tex methods.tex results.tex discussion.tex conclusion.tex > words.sum }%
  \input{words.sum}%
}

\setlength{\parindent}{0em}
\setlength{\parskip}{1em}
\renewcommand{\baselinestretch}{1.5}
\newcommand{\wordcount}{\input{final_project.sum}}
