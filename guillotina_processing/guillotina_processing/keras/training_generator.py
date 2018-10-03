"""Part of the training engine related to Python generators of array data.
"""
from guillotina_processing.keras.evaluate_generator import evaluate_generator
from guillotina_processing.keras.enqueuer import Sequence
from keras.utils.generic_utils import to_list
from keras import callbacks as cbks
import asyncio


async def fit_generator(
        model,
        generator,
        steps_per_epoch=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        class_weight=None,
        shuffle=True,
        initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    epoch = initial_epoch

    do_validation = bool(validation_data)
    model._make_train_function()
    if do_validation:
        model._make_test_function()

    if steps_per_epoch is None:
        steps_per_epoch = len(generator)

    # Prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    output_generator = generator.async_next

    callback_model.stop_training = False
    # Construct epoch logs.
    epoch_logs = {}
    while epoch < epochs:
        for m in model.stateful_metric_functions:
            m.reset_states()
        callbacks.on_epoch_begin(epoch)
        steps_done = 0
        batch_index = 0
        while steps_done < steps_per_epoch:
            generator_output = await output_generator()

            if not hasattr(generator_output, '__len__'):
                raise ValueError('Output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))

            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                raise ValueError('Output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
            # build batch logs
            batch_logs = {}
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
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_size
            callbacks.on_batch_begin(batch_index, batch_logs)

            outs = model.train_on_batch(x, y,
                                        sample_weight=sample_weight,
                                        class_weight=class_weight)

            outs = to_list(outs)
            for l, o in zip(out_labels, outs):
                batch_logs[l] = o

            callbacks.on_batch_end(batch_index, batch_logs)

            batch_index += 1
            steps_done += 1

            # Epoch finished.
            if steps_done >= steps_per_epoch and do_validation:
                val_outs = await evaluate_generator(
                    model,
                    validation_data,
                    validation_steps)
                val_outs = to_list(val_outs)
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o

            if callback_model.stop_training:
                break

        generator.on_epoch_end()
        callbacks.on_epoch_end(epoch, epoch_logs)
        epoch += 1
        if callback_model.stop_training:
            break

    callbacks.on_train_end()
    return model.history
