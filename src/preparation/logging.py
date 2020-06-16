import os
from typing import Any, Union
from enum import Enum

from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import *


def custom_global_step_from_engine(trainer, evaluator, evaluator_type, verbose=True, custom_print=None, loss_name="nll"):
    def wrapper(_, event_name):
        if verbose:
            metrics = evaluator.state.metrics
            epoch = trainer.state.epoch
            epoch_string = "" if evaluator_type == "Test" else "Epoch: {:03d}".format(
                epoch)
            suffix = "\n" if evaluator_type == "Test" else ""

            message = "{} Results - {} ".format(evaluator_type, epoch_string)
            if metrics.get("accuracy", None):
                message += "Avg accuracy: {:.2f} ".format(metrics["accuracy"])

            avg_loss = metrics.get(loss_name)
            message += "Avg loss: {:.2f} ".format(avg_loss)

            if metrics.get("precision", None):
                message += "Avg precision: {:.2f} ".format(
                    metrics["precision"])

            if metrics.get("recall", None):
                message += "Avg recall: {:.2f}".format(metrics["recall"])
            
            message += suffix
            
            if custom_print is not None:
                custom_print(message)
            else:
                print(message)

        return trainer.state.get_event_attrib_value(event_name)

    return wrapper


def create_tb_logger(model, optimizer, trainer, train_evaluator, val_evaluator, test_evaluator, **kwargs):

    verbose = kwargs.get("verbose", True)
    custom_print = kwargs.get("custom_print", None)
    loss_name = kwargs.get("loss_name", None)
    del kwargs["verbose"]
    del kwargs["custom_print"]
    del kwargs["loss_name"]
    
    # Create a logger
    tb_logger = TensorboardLogger(**kwargs)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(
                         tag="training", output_transform=lambda loss: {'loss': loss}),
                     event_name=Events.ITERATION_COMPLETED)

    # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
    # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
    # of the `trainer` instead of `train_evaluator`.
    tb_logger.attach(train_evaluator,
                     log_handler=OutputHandler(tag="training",
                                               metric_names=[
                                                   loss_name, "accuracy", "precision", "recall"],
                                               global_step_transform=custom_global_step_from_engine(trainer, train_evaluator, "[   Train  ]", verbose=verbose, custom_print=custom_print, loss_name=loss_name)),
                     event_name=Events.EPOCH_COMPLETED)

    # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
    # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach(val_evaluator,
                     log_handler=OutputHandler(tag="validation",
                                               metric_names=[
                                                   loss_name, "accuracy", "precision", "recall"],
                                               global_step_transform=custom_global_step_from_engine(trainer, val_evaluator, "[Validation]", verbose=verbose, custom_print=custom_print, loss_name=loss_name)),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(test_evaluator,
                     log_handler=OutputHandler(tag="test",
                                               metric_names=[
                                                   loss_name, "accuracy", "precision", "recall"],
                                               global_step_transform=custom_global_step_from_engine(trainer, test_evaluator, "Test", custom_print=custom_print, loss_name=loss_name)),
                     event_name=Events.COMPLETED)

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer),
                     event_name=Events.ITERATION_STARTED)

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(trainer,
                     log_handler=WeightsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED)

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(trainer,
                     log_handler=WeightsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED)

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(trainer,
                     log_handler=GradsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED)

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(trainer,
                     log_handler=GradsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED)

    return tb_logger
