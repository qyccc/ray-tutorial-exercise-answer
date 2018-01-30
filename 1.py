from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time

ray.init(num_cpus=4, redirect_output=True)


import ray.experimental.ui as ui
ui.task_timeline()

@ray.remote
def remote_function(i):
    time.sleep(1)
    return i


time.sleep(2.0)
start_time1 = time.time()

results = []
for i in range(4):
    results.append(remote_function.remote(i))
results=ray.get(results)
end_time1 = time.time()
duration1 = end_time1 - start_time1

assert results == [0, 1, 2, 3], 'Did you remember to call ray.get?'
assert duration1 < 1.1, ('The loop took {} seconds. This is too slow.'
                        .format(duration1))
assert duration1 > 1, ('The loop took {} seconds. This is too fast.'
                      .format(duration1))

print('Success! The example took {} seconds.'.format(duration1))

