# -*- coding: utf-8 -*-
"""ray.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HulCkopip47XGbCMKOznaErtDc5vBc20
"""

!pip install ray

import time

# Define a complex mathematical operation
def complex_operation(n):
    result = 0
    for i in range(n):
        result += (i ** 2) + (i // 2) + (i % 3)  # Simulating a complex operation
    return result

# Parameters
n = 10000000  # Performing the complex operation 10,000,000 times

# Sequential computation
start_time = time.time()
result_seq = complex_operation(n)
total_time_seq = time.time() - start_time

# Output the results
print(f"Without Ray: {result_seq}\nTotal time: {total_time_seq} seconds")

ray.shutdown()

import time
import ray

# Define a complex mathematical operation
@ray.remote
def complex_operation(n):
    result = 0
    for i in range(n):
        result += (i ** 2) + (i // 2) + (i % 3)  # Simulating a complex operation
    return result

# Parameters
n = 10000000  # Performing the complex operation 10,000,000 times

# Parallel computation using Ray
ray.init()
start_time = time.time()
result_ray = ray.get(complex_operation.remote(n))
total_time_ray = time.time() - start_time

# Output the results
print(f"Ray result: {result_ray}\nTotal time: {total_time_ray} seconds")
ray.shutdown()

