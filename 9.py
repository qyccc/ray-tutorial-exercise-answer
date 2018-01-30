from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=4, num_gpus=2, redirect_output=True)

@ray.remote(num_gpus=1)
def f():
    time.sleep(0.5)
    return ray.get_gpu_ids()

start_time = time.time()

gpu_ids = ray.get([f.remote() for _ in range(3)])

end_time = time.time()

for i in range(len(gpu_ids)):
    assert len(gpu_ids[i]) == 1

assert end_time - start_time > 1

print('Sucess! The test passed.')

@ray.remote(num_gpus=1)
class Actor(object):
    def __init__(self):
        pass

    def get_gpu_ids(self):
        return ray.get_gpu_ids()

actor = Actor.remote()

gpu_ids = ray.get(actor.get_gpu_ids.remote())

assert len(gpu_ids) == 1

print('Sucess! The test passed.')
