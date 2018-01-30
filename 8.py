from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import ray
import time

ray.init(num_cpus=4, redirect_output=True)
neural_net_weights = {'variable{}'.format(i): np.random.normal(size=1000000)
                      for i in range(50)}



@ray.remote
def use_weights(weights, i):
    return i


# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()
neural_net_weights_id= ray.put(neural_net_weights)
results = ray.get([use_weights.remote(neural_net_weights_id, i)
                   for i in range(20)])

end_time = time.time()
duration = end_time - start_time

assert results == list(range(20))
assert duration < 1, ('The experiments ran in {} seconds. This is too '
                      'slow.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))