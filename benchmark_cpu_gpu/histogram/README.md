Histogram

Compilation instructions:

  - GPU: 'nvcc -o hist hist.cu'
  - CPU: 'make'

Execution instructions:

  - GPU: './hist', can specify specific files with -i flag
  - CPU: './hist <bitmap filename>'

Input files:

Testing can be done with any 24-bit .bmp file.
