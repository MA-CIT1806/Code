import torch
import numpy as np

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
import src.preparation.datasets_helpers as dh_helpers


class MyBatch():
    def __init__(self, x=None, y=None, edge_index=None, batch=None, num_nodes=None, device=None):
        self.x = x[0].to(device)
        self.y = y[0].to(device)
        self.edge_index = edge_index[0].to(device)
        self.batch = batch[0].to(device)
        self.num_nodes = num_nodes[0].to(device)


def transform_batch(extraction_target, batch, device):
    if extraction_target == "window":
        return batch.to(device)
    elif extraction_target == "sequence":
        return MyBatch(**batch, device=device)


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False, extraction_target="window",
                              output_transform=lambda x, y, y_pred, loss: loss.item()):

    def _update(engine, batch):
        batch = transform_batch(extraction_target, batch, device)
        model.train()
        optimizer.zero_grad()
        response = model(batch)

        loss = loss_fn(response, batch.y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        return output_transform(batch.x, batch.y, response, loss)

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                pred_collector_function=None, extraction_target="window",
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    metrics = metrics or {}

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = transform_batch(extraction_target, batch, device)
            response = model(batch)

            if pred_collector_function is not None:
                pred_collector_function(response)
            
            return output_transform(batch.x, batch.y, response)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
