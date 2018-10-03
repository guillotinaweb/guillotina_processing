import numpy as np
from keras.utils.generic_utils import unpack_singleton
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import Progbar
import asyncio


async def evaluate_generator(
        model, generator,
        steps=None,
        verbose=1):
    """See docstring for `Model.evaluate_generator`."""
    model._make_test_function()

    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    steps_done = 0
    outs_per_batch = []
    batch_sizes = []

    if steps is None:
        steps = len(generator)

    output_generator = generator.async_next

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = await output_generator()

        if not hasattr(generator_output, '__len__'):
            raise ValueError('Output of generator should be a tuple '
                             '(x, y, sample_weight) '
                             'or (x, y). Found: ' +
                             str(generator_output))
        if len(generator_output) == 2:
            x, y = generator_output
            sample_weight = None
        elif len(generator_output) == 3:
            x, y, sample_weight = generator_output
        else:
            raise ValueError('Output of generator should be a tuple '
                             '(x, y, sample_weight) '
                             'or (x, y). Found: ' +
                             str(generator_output))
        outs = model.test_on_batch(x, y, sample_weight=sample_weight)
        outs = to_list(outs)
        outs_per_batch.append(outs)

        if x is None or len(x) == 0:
            # Handle data tensors support when no input given
            # step-size = 1 for data tensors
            batch_size = 1
        elif isinstance(x, list):
            batch_size = x[0].shape[0]
        elif isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
        else:
            batch_size = x.shape[0]
        if batch_size == 0:
            raise ValueError('Received an empty batch. '
                             'Batches should contain '
                             'at least one item.')
        steps_done += 1
        batch_sizes.append(batch_size)
        if verbose == 1:
            progbar.update(steps_done)

    generator.on_epoch_end()
    averages = []
    for i in range(len(outs)):
        if i not in stateful_metric_indices:
            averages.append(np.average([out[i] for out in outs_per_batch],
                                       weights=batch_sizes))
        else:
            averages.append(np.float64(outs_per_batch[-1][i]))
    return unpack_singleton(averages)
