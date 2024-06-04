/*
 * Copyright 2011-2017 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain metric values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <callback_metric_any_function.cu>
bool runPass(int argc, char **argv);
 int main(int argc, char *argv[])
{
	 FUNC_PTR func=runPass;
	 //printf("hi");

	return main_function_metric(argc, argv, func);
}

