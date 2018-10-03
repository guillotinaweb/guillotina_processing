import multiprocessing as mp
import random
import sys
import threading
import time
import warnings
from abc import abstractmethod
from contextlib import closing
from multiprocessing.pool import ThreadPool

import numpy as np
import six
import asyncio


class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.
    # Notes
    `Sequence` are a safer way to do multiprocessing. This structure guarantees
    that the network will only train once on each sample per epoch which is not
    the case with generators.
    # Examples
    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np
        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.
        class CIFAR10Sequence(Sequence):
            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size
            def __len__(self):
                return int(np.ceil(len(self.x) / float(self.batch_size)))
            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None


def init_pool(seqs):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = seqs


def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.
    To allow multiple Sequences to be used at the same time, we use `uid` to
    get a specific one. A single Sequence would cause the validation to
    overwrite the training Sequence.
    # Arguments
        uid: int, Sequence identifier
        i: index
    # Returns
        The value at index `i`.
    """
    return _SHARED_SEQUENCES[uid][i]


class SequenceEnqueuer(object):
    """Base class to enqueue inputs.
    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.
    # Examples
    ```python
        enqueuer = SequenceEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.close()
    ```
    The `enqueuer.get()` should be an infinite stream of datas.
    """
    def __init__(self, sequence,
                 use_multiprocessing=False):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing

        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = mp.Value('i', 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.
        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            # We do not need the init since it's threads.
            self.executor_fn = lambda _: ThreadPool(workers)
        self.workers = workers
        self.queue = asyncio.Queue(max_queue_size)
        self.stop_signal = asyncio.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _send_sequence(self):
        """Send current Iterable to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None

    @abstractmethod
    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        raise NotImplementedError

    @abstractmethod
    def _get_executor_init(self, workers):
        """Get the Pool initializer for multiprocessing.
        # Returns
            Function, a Function to initialize the pool
        """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError


class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """
    def __init__(self, sequence, use_multiprocessing=False, shuffle=False):
        super(OrderedEnqueuer, self).__init__(sequence, use_multiprocessing)
        self.shuffle = shuffle

    def _get_executor_init(self, workers):
        """Get the Pool initializer for multiprocessing.
        # Returns
            Function, a Function to initialize the pool
        """
        return lambda seqs: mp.Pool(workers,
                                    initializer=init_pool,
                                    initargs=(seqs,))

    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    async def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)

            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(
                        executor.apply_async(get_index, (self.uid, i)), block=True)

                # Done with the current epoch, waiting for the final batches
                self._wait_queue()

                if self.stop_signal.is_set():
                    # We're done
                    return

            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self._send_sequence()  # Update the pool

    async def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            raise


def init_pool_generator(gens, random_seed=None):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = gens

    if random_seed is not None:
        ident = mp.current_process().ident
        np.random.seed(random_seed + ident)


def next_sample(uid):
    """Get the next value from the generator `uid`.
    To allow multiple generators to be used at the same time, we use `uid` to
    get a specific one. A single generator would cause the validation to
    overwrite the training generator.
    # Arguments
        uid: int, generator identifier
    # Returns
        The next value of generator `uid`.
    """
    return six.next(_SHARED_SEQUENCES[uid])


class GeneratorEnqueuer(SequenceEnqueuer):
    """Builds a queue out of a data generator.
    The provided generator can be finite in which case the class will throw
    a `StopIteration` exception.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each worker.
    """

    def __init__(self, sequence, use_multiprocessing=False, wait_time=None,
                 random_seed=None):
        super(GeneratorEnqueuer, self).__init__(sequence, use_multiprocessing)
        self.random_seed = random_seed
        if wait_time is not None:
            warnings.warn('`wait_time` is not used anymore.',
                          DeprecationWarning)

    def _get_executor_init(self, workers):
        """Get the Pool initializer for multiprocessing.
        # Returns
            Function, a Function to initialize the pool
        """
        return lambda seqs: mp.Pool(workers,
                                    initializer=init_pool_generator,
                                    initargs=(seqs, self.random_seed))

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        self._send_sequence()  # Share the initial generator
        with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
            while True:
                if self.stop_signal.is_set():
                    return
                self.queue.put(
                    executor.apply_async(next_sample, (self.uid,)), block=True)

    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except StopIteration:
            # Special case for finite generators
            last_ones = []
            while self.queue.qsize() > 0:
                last_ones.append(self.queue.get(block=True))
            # Wait for them to complete
            list(map(lambda f: f.wait(), last_ones))
            # Keep the good ones
            last_ones = [future.get() for future in last_ones if future.successful()]
            for inputs in last_ones:
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            if 'generator already executing' in str(e):
                raise RuntimeError(
                    "Your generator is NOT thread-safe."
                    "Keras requires a thread-safe generator when"
                    "`use_multiprocessing=False, workers > 1`."
                    "For more information see issue #1638.")
            six.reraise(*sys.exc_info())