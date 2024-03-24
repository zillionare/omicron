import logging
import uuid
from multiprocessing import Process, Queue
from typing import Any, Callable, List, Optional, Union

from psutil import cpu_count

logger = logging.getLogger(__name__)

class Actor:
    """自带执行体的任务。

    本类及其子类必须是可串行化(serializable)的。
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or uuid.uuid4().hex[-6:]

    def run(self, results_queue: Queue):
        raise NotImplementedError


class GeneralActor:
    def __init__(
        self, target: Callable, args: tuple[Any, ...], name: Optional[str] = None
    ):
        self.target = target
        self.args = args
        self.name = name or uuid.uuid4().hex[-6:]

    def run(self, results_queue):
        result = self.target(*self.args)
        if results_queue is not None:
            results_queue.put_nowait(result)


class StopOnSightActor(Actor):
    """执行体在见到此任务后，将执行退出进程操作"""

    def __init__(self):
        super().__init__("stop-on-sight")

class Executor(Process):
    """通用任务执行体

    启动后，从job_queue中接受任务并执行，直到遇到StopOnSightJob时退出。
    在启动后，执行任何任务之前，如果指定了before_start，则执行初始化；在退出前，如果指定了before_end，则执行退出操作。
    """

    def __init__(
        self,
        job_queue: Queue,
        before_start: Optional[Callable] = None,
        before_end: Optional[Callable] = None,
        result_queue: Optional[Queue] = None,
    ):
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.before_start = before_start
        self.before_end = before_end

        super().__init__()

    def run(self):
        if self.before_start is not None:
            self.before_start()

        while True:
            job = self.job_queue.get()
            if isinstance(job, StopOnSightActor):
                break
            else:
                job.run(self.result_queue)

        if self.before_end is not None:
            self.before_end()


class ExecutorPool:
    def __init__(
        self,
        before_start: Optional[Callable] = None,
        before_end: Optional[Callable] = None,
        max_workers: Optional[int] = None,
    ):
        self.executors: List[Process] = []
        self.jobs_q = Queue()
        self.results_q = Queue()
        if max_workers is None:
            n = cpu_count()
        else:
            n = max_workers

        for _ in range(n):  # type: ignore
            executor = Executor(
                job_queue=self.jobs_q,
                before_start=before_start,
                before_end=before_end,
                result_queue=self.results_q,
            )
            executor.start()
            self.executors.append(executor)

        logger.info("%s executors are created and started", len(self.executors))

    def submit(self, jobs: Union[Actor, List[Actor]]):
        if isinstance(jobs, Actor):
            jobs = [jobs]

        for job in jobs:
            self.jobs_q.put_nowait(job)
            if not isinstance(job, StopOnSightActor):
                logger.debug("Job %s submitted", job.name)

    @property
    def results(self):
        return self.results_q

    def shutdown(self):
        """关闭进程池"""
        for i in range(len(self.executors)):
            self.submit(StopOnSightActor())

    def join(self):
        """确保主进程等待子进程结束后再退出。"""
        for executor in self.executors:
            logger.info("executor %s exited", executor.pid)
            executor.join()
            executor.close()
