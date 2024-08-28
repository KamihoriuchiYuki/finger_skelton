import numpy as np
import random as rand
num_node = 21
finger = np.zeros((1, num_node))
for i in range(num_node):
    finger[0, i] = rand.uniform(0, 100)

depth_order = np.zeros((1, num_node), dtype=int)
for i in range(num_node):
    if i != 0:
        depth_order[0, i] = i
        che = 0
    for j in range(i):
        if che == 0:
            if finger[0, depth_order[0, j]] > finger[0, i]:
                che = 1
                depth_order[0, j], depth_order[0, i] = depth_order[0, i], depth_order[0, j]
        else:
            depth_order[0, j], depth_order[0, i] = depth_order[0, i], depth_order[0, j]
        print(finger)
        print(depth_order)

print(finger)
print(depth_order)