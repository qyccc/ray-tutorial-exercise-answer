from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=5, redirect_output=True)

@ray.remote
def f():
    time.sleep(np.random.uniform(0, 5))
    return time.time()



# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

result_ids = [f.remote() for _ in range(10)]

# Get the results.
results = []
for result_id in result_ids:
    result1,result_ids1=ray.wait(result_ids,1,None)
    result = ray.get(result1)
    results.append(result)
    #print('Processing result which finished after {} seconds.'
     #     .format(result - start_time))

end_time = time.time()
duration = end_time - start_time

assert results == sorted(results), ('The results were not processed in the '
                                    'order that they finished.')

print('Success! The example took {} seconds.'.format(duration))