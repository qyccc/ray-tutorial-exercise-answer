from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=9, redirect_output=True)
@ray.remote
def compute_gradient(data):
    time.sleep(0.03)
    return 1

@ray.remote
def train_model(hyperparameters):
    result = 0
    for i in range(10):
        # EXERCISE: After you turn "compute_gradient" into a remote function,
        # you will need to call it with ".remote". The results must be retrieved
        # with "ray.get" before "sum" is called.
        result += sum(ray.get([compute_gradient.remote(j) for j in range(2)]))
    return result
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# Run some hyperparaameter experiments.
results = []
for hyperparameters in [{'learning_rate': 1e-1, 'batch_size': 100},
                        {'learning_rate': 1e-2, 'batch_size': 100},
                        {'learning_rate': 1e-3, 'batch_size': 100}]:
    results.append(train_model.remote(hyperparameters))
results=ray.get(results)
end_time = time.time()
duration = end_time - start_time
assert results == [20, 20, 20]
assert duration < 0.5, ('The experiments ran in {} seconds. This is too '
                         'slow.'.format(duration))
assert duration > 0.3, ('The experiments ran in {} seconds. This is too '
                        'fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
import ray.experimental.ui as ui
ui.task_timeline()