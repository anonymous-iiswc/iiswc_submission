This benchmark application implements the supervised ML algorithm K-Nearest Neighbors. 
The parameters you can provide to this program include the number of threads, k, and the target index of the data column you wish to analyze.
You may also change the dataset to include by adjusting the first few lines in the main function.

A note about PIM research:

In our CPU implementation of KNN, we do not perform any feature scaling on our data, which results in poorer KNN clasifcation accuracy. 
We made this choice to allign the CPU implementation with the BitSIMD implementation since the BitSIMD simulator does not support floating-point operations
and feature scaling produces floating-point data values.
However, our CPU implementation still completes the algorithm end-to-end, providing useful results for benchmarking purposes despite innaccurate results.