from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=6, redirect_output=True)

@ray.remote
def f(i):
    np.random.seed(5 + i)
    x = np.random.uniform(0, 4)
    time.sleep(x)
    return i, time.time()

# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# This launches 6 tasks, each of which takes a random amount of time to
# complete.
result_ids = [f.remote(i) for i in range(6)]
# Get one batch of tasks. Instead of waiting for a fixed subset of tasks, we
# should instead use the first 3 tasks that finish.
initial_results , remaining_ids = ray.wait(result_ids, 3, None)
initial_results= ray.get(initial_results)

end_time = time.time()
duration = end_time - start_time

remaining_results = ray.get(remaining_ids)

assert len(initial_results) == 3
assert len(remaining_results) == 3
print(initial_results+remaining_results)
initial_indices = [result[0] for result in initial_results]
initial_times = [result[1] for result in initial_results]
remaining_indices = [result[0] for result in remaining_results]
remaining_times = [result[1] for result in remaining_results]
print(initial_indices+remaining_indices)
assert set(initial_indices + remaining_indices) == set(range(6))

assert duration < 1.5, ('The initial batch of ten tasks was retrieved in '
                        '{} seconds. This is too slow.'.format(duration))

assert duration > 0.8, ('The initial batch of ten tasks was retrieved in '
                        '{} seconds. This is too slow.'.format(duration))

# Make sure the initial results actually completed first.
assert max(initial_times) < min(remaining_times)

print('Success! The example took {} seconds.'.format(duration))