"""
sametime模块的示例程序。
"""

import os
import time
from multiprocessing import Queue
from threading import Thread

from omicron.sametime import Actor, ExecutorPool, StopOnSightActor


def before_start():
    print(f"before_start is called in process {os.getpid()}")


def before_end():
    print(f"before_end is called in process {os.getpid()}")


def work(i):
    print(f"sleep and work {i} in {os.getpid()}")
    time.sleep(i)


def shutdown(pool):
    time.sleep(5)
    print("trying to exit")
    pool.shutdown()


if __name__ == "__main__":
    pool = ExecutorPool(before_start, before_end, max_workers=3)
    for i in range(6):
        pool.submit(Actor(work, (i,)))

    t = Thread(target=shutdown, args=(pool,))
    t.start()
    pool.join()
