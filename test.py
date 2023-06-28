import random

import queue

q=queue.Queue()
for i in range(10):
    if i==5:
        while not q.empty():
            a=q.get()
            print(a)
    q.put(i)
while not q.empty():
    a = q.get()
    print(a)


