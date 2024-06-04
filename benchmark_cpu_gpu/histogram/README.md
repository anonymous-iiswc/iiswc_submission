Histogram

Compilation instructions:

  - GPU: 'nvcc -o hist hist.cu'
  - CPU: 'make'

Execution instructions:

  - GPU: './hist', can specify specific files with -i flag
  - CPU: './hist <bitmap filename>'

Input files:

Testing can be done with any 24-bit .bmp file. Sample input can be found from the Phoenix [GitHub](https://github.com/fasiddique/DRAMAP-Phoenix/tree/main) or the direct [link](http://csl.stanford.edu/~christos/data/histogram.tar.gz).
