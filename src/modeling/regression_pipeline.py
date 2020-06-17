import torch
import torch.nn.functional as F
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Precision, Recall, ConfusionMatrix
from ignite.handlers import EarlyStopping, TerminateOnNan, Checkpoint, DiskSaver, global_step_from_engine
from src.modeling.classification_pipeline import ClassificationPipeline
from src.preparation.datasets import GraphDataset
from src.modeling.train import create_supervised_trainer, create_supervised_evaluator
from src.preparation.logging import create_tb_logger
from src.modeling.utils import LocalSaveHandler


class RegressionPipeline(ClassificationPipeline):
    
    def _prepare_trained_model(self, model, optimizer):
        """Load a model from checkpoint."""
    
        model, optimizer = self._load_checkpoint(model, optimizer, torch.load(self.trained_model_checkpoint))

        return model, optimizer    

    def _run_training(self, train_data, valid_data=[], test_data=[], tb_log_dir=None, split_num=None, verbose=True):

        # setup
        epochs = self.training_config.get("epochs")
        optimizer_config = self.training_config.get("optimizer_config")
        early_stopping = self.training_config.get("early_stopping")
        checkpoint_saving = self.training_config.get("checkpoint_saving")
        graph_dataset_config = self.training_config.get("graph_dataset_config")
        device = self.training_config.get("device")
        extraction_target = self.training_config.get(
            "extraction_target")

        graph_dataset_config["device"] = device
        graph_dataset_config["extraction_target"] = extraction_target

        if split_num is not None:
            tb_log_dir += "-Split{}".format(split_num+1)

        # prepare graph-sets
        graph_dataset = GraphDataset(train_data, valid_data=valid_data,
                                     test_data=test_data, graph_dataset_config=graph_dataset_config)

        train_loader, val_loader, test_loader = graph_dataset.get_loaders()

        worker_init_fn = graph_dataset.init_fn

        # create model, optimizer, loss
        model = self.model_class(self.model_config)
        model = model.to(device)

        optimizer = self.optimizer_class(
            model.parameters(), **optimizer_config)

        # apparently, we have to do this
        self.model = model
        self.optimizer = optimizer

        # load model from checkpoint if available
        if self.trained_model_checkpoint is not None:
            self.custom_print("load transfer-learning checkpoint...")
            model, optimizer = self._prepare_trained_model(model, optimizer)

        loss = self.loss_class()
        loss_name = "mse"
            
        evaluator_settings = {
            "device": device,
            "extraction_target": extraction_target,
            "pred_collector_function": lambda x: self._pred_collector_function(x),
            "metrics": {
                loss_name: Loss(loss)
            }
        }

        ## configure trainer ##
        trainer = create_supervised_trainer(
            model, optimizer, loss, device=device, extraction_target=extraction_target)

        ###############################################
        ## configure evaluators for each data source ##
        train_evaluator = create_supervised_evaluator(model,
                                                      **evaluator_settings)

        val_evaluator = create_supervised_evaluator(model,
                                                    **evaluator_settings)

        test_evaluator = create_supervised_evaluator(model,
                                                     **evaluator_settings)

        # configure behavior for early stopping
        if early_stopping is not None:
            stopper = EarlyStopping(
                patience=early_stopping, score_function=self.score_function, trainer=trainer)
            val_evaluator.add_event_handler(Events.COMPLETED, stopper)

        # configure behavior for checkpoint saving
        if checkpoint_saving is not None:
            save_handler = None
            if self.test_mode and self.validation_mode:
                self.custom_print("Use LocalSaveHandler...")
                save_handler = LocalSaveHandler(self)
            else:
                self.custom_print("Use IgniteSaveHandler...")
                save_handler = DiskSaver(self.save_path, create_dir=True,
                                         require_empty=False)

            saver = Checkpoint(
                {
                    "model_state_dict": model,
                    "optimizer_state_dict": optimizer
                },
                save_handler,
                filename_prefix='{}_best'.format(self.dataset.name),
                score_name="val_loss",
                score_function=self.score_function,
                global_step_transform=global_step_from_engine(trainer),
                n_saved=1)
            train_evaluator.add_event_handler(Events.COMPLETED, saver)

        @trainer.on(Events.STARTED)
        def log_training_start(trainer):
            self.custom_print("Split: {}".format(split_num+1))

        @trainer.on(Events.COMPLETED)
        def log_training_complete(trainer):
            """Trigger evaluation on test set if training is completed."""

            epoch = trainer.state.epoch
            suffix = "(Early Stopping)" if epoch < epochs else ""

            self.custom_print("Finished after {:03d} epochs! {}".format(
                epoch, suffix))

            embedding_list = []

            def _graph_embedding_function(tensor, idx):
                while idx >= len(embedding_list):
                    embedding_list.append([])
                embedding_list[idx].append(tensor.cpu().detach().numpy())

            if self.test_mode and self.validation_mode:
                checkpoint_dict = self.best_model_checkpoint
                self.custom_print("Load best model checkpoint by validation loss... Epoch: {}".format(
                    checkpoint_dict["epoch"]))
                model, optimizer = self._load_checkpoint(
                    self.model, self.optimizer, checkpoint_dict["checkpoint"])

            self.model.graph_embedding_function = _graph_embedding_function
            self.persist_pred = True

            if not self.test_mode:
                return

            test_evaluator.run(test_loader)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            """Compute evaluation metric values after each epoch."""
            train_evaluator.run(train_loader)
            
            if hasattr(self.model, "node_counter"):
                self.custom_print(self.model.node_counter)

            if self.validation_mode:
                val_evaluator.run(val_loader)

        # terminate training if Nan values are produced
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        
        # create tensorboard-logger
        tb_logger = create_tb_logger(model,
                                     optimizer,
                                     trainer,
                                     train_evaluator,
                                     val_evaluator,
                                     test_evaluator,
                                     log_dir=tb_log_dir,
                                     verbose=verbose,
                                     custom_print=self.custom_print,
                                     loss_name=loss_name
                                     )

        with torch.autograd.detect_anomaly():
            trainer.run(train_loader, max_epochs=epochs)

        tb_logger.close()

        if not self.test_mode:
            return 0, 0

        test_acc = test_evaluator.state.metrics["accuracy"]
        test_loss = test_evaluator.state.metrics["mse"]

        return test_acc, test_loss
