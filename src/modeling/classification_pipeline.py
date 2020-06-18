import torch
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Precision, Recall, ConfusionMatrix
from ignite.handlers import EarlyStopping, TerminateOnNan, Checkpoint, DiskSaver, global_step_from_engine
import numpy as np
from src.modeling.train import create_supervised_trainer, create_supervised_evaluator
from src.modeling.models_tf import ProposalRLLL, ProposalRLGRU
from src.preparation.logging import create_tb_logger
from src.modeling.evaluation import create_confusion_matrix, create_temporal_softmax_data
from src.preparation.splitting import LeaveOneGroupOutWrapper, TransferLearningWrapper
from src.preparation.datasets import GraphDataset, MultiAnomalyDataset
from src.utils import setup_tb_log_dir
import datetime
import random
import re
import json
from src.modeling.utils import LocalSaveHandler
from src.modeling.models import Proposal


class ClassificationPipeline():
    def __init__(self, datasets, model_settings={}, **kwargs):

        if isinstance(datasets, list):
            if len(datasets) == 1:
                print("Use SingleAnomalyDataset...")
                self.dataset = datasets[0]
            else:
                print("Use MultiAnomalyDataset...")
                self.dataset = MultiAnomalyDataset(datasets)

        print(self.dataset)

        self.data_transform_steps = kwargs.get("data_transform_steps", {})

        self.model_class = model_settings.get("model_class")
        self.optimizer_class = model_settings.get("optimizer_class")
        self.loss_class = model_settings.get("loss_class")
        self.save_path = model_settings.get("save_path")

        self.score_function = kwargs.get("score_function", None)

        self.model_config = kwargs.get("model_config", {})
        self.training_config = kwargs.get("training_config", {})

        self.model_config["num_classes"] = self.dataset.num_classes

        self.pred_list = []
        self.persist_pred = False

        self.trained_model_checkpoint = kwargs.get(
            "trained_model_checkpoint", None)

    def _prepare_metric_dict(self, metrics):
        """Prepare a metric dict for tensorboard."""

        new_dict = dict()
        for key, value in metrics.items():
            if "cm" not in key:
                new_dict["hparam/{}".format(key)] = value

        return new_dict

    def _prepare_hparam_dict(self, hparams):
        """Prepare a hyperparameter dict for tensorboard."""

        new_dict = dict()
        for key, value in hparams.items():
            if isinstance(value, (float, int)):
                new_dict[key] = value

        return new_dict

    def _pred_collector_function(self, pred):
        """Collect predictions on test-set."""

        if self.persist_pred:
            self.pred_list.append(pred)

    def _get_collected_predictions(self, class_labels):
        """Prepare collected predictions, determine predicted classes."""

        pred_labels_raw = np.concatenate(
            [pd.cpu().detach().numpy() for pd in self.pred_list], axis=0)
        pred_labels = np.array(
            [list(class_labels)]).reshape(-1, 1)[np.argmax(pred_labels_raw, axis=1)]

        self.pred_list = []
        self.persist_pred = False

        return pred_labels, pred_labels_raw

    def _load_checkpoint(self, model, optimizer, checkpoint):
        """Load a model from checkpoint."""

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    def _prepare_trained_model(self, model, optimizer, device):
        """Load a model from checkpoint."""

        additional_parameters = None

        if isinstance(model, Proposal):
            model, optimizer = self._load_checkpoint(
                model, optimizer, torch.load(self.trained_model_checkpoint))

            model.to(device)

            for param in model.parameters():
                param.requires_grad = False

            additional_parameters = model.prepare_fine_tuning()

        elif isinstance(model, (ProposalRLLL, ProposalRLGRU)):
            model.sub_module1, optimizer = self._load_checkpoint(
                model.sub_module1, optimizer, torch.load(self.trained_model_checkpoint))

            for param in model.sub_module1.parameters():
                param.requires_grad = False

            model.sub_module1.fc_task1 = None
            model.sub_module1.fc_task2 = None

            additional_parameters = model.prepare_fine_tuning()

        torch.cuda.empty_cache()

        if isinstance(additional_parameters, list):
            print("prepare fine-tuning...")
            for p in additional_parameters:
                optimizer.add_param_group({"params": p.parameters()})

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
            model, optimizer = self._prepare_trained_model(
                model, optimizer, device)

        loss = self.loss_class()

        def output_transform(x): return x

        evaluator_settings = {
            "device": device,
            "extraction_target": extraction_target,
            "pred_collector_function": lambda x: self._pred_collector_function(x),
            "metrics": {
                'accuracy': Accuracy(output_transform=output_transform),
                'nll': Loss(loss),
                'precision': Precision(average=True, output_transform=output_transform),
                'recall': Recall(average=True, output_transform=output_transform)
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

        evaluator_settings["metrics"]["cm"] = ConfusionMatrix(
            self.dataset.num_classes, average="samples", output_transform=output_transform)
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
            val_evaluator.add_event_handler(Events.COMPLETED, saver)

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

            y_true = graph_dataset.y_test.cpu().detach().numpy()

            y_pred, y_pred_raw = self._get_collected_predictions(
                self.dataset.class_name_mapping.keys())

            # add testset pr-curves
            for i, label in enumerate(self.dataset.class_name_mapping.values()):
                temp_label_true = np.zeros(y_true.shape)
                temp_label_true[y_true == i] = 1
                tb_logger.writer.add_pr_curve(
                    "test_pr_curve_[{}]".format(label),
                    temp_label_true,
                    np.exp(y_pred_raw[:, i])
                )

            # add temporal softmax plots
            temporal_softmax_data = create_temporal_softmax_data(
                y_true, y_pred_raw, self.dataset.class_name_mapping)
            for anomaly_name, dict_list in temporal_softmax_data.items():
                for list_idx, list_elem in enumerate(dict_list):
                    tb_logger.writer.add_scalars("test_temporal_softmax_Split{}_{}".format(
                        split_num+1, anomaly_name), list_elem, list_idx)

            # add model hyperparameters and testset metric results
            tb_logger.writer.add_hparams(
                hparam_dict=self._prepare_hparam_dict(
                    dict(**optimizer_config, batch_size=graph_dataset.batch_size)),
                metric_dict=self._prepare_metric_dict(
                    test_evaluator.state.metrics),
                name="session"
            )

            # add testset confusion matrices
            tb_logger.writer.add_figure(
                "test_cm",
                create_confusion_matrix(
                    y_true, y_pred, self.dataset.class_name_mapping.values())
            )

            # add testset-embeddings
            for i in range(len(embedding_list)):
                tb_logger.writer.add_embedding(
                    np.concatenate(embedding_list[i], axis=0),
                    metadata=y_true,
                    tag="test_embedding_{}".format(i))

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            """Compute evaluation metric values after each epoch."""

            train_evaluator.run(train_loader)

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
                                     custom_print=lambda x: self.custom_print(
                                         x),
                                     loss_name="nll"
                                     )

        with torch.autograd.detect_anomaly():
            trainer.run(train_loader, max_epochs=epochs)

        tb_logger.close()

        if not self.test_mode:
            return 0, 0

        test_acc = test_evaluator.state.metrics["accuracy"]
        test_loss = test_evaluator.state.metrics["nll"]

        return test_acc, test_loss

    def custom_print(self, text):
        """Redirect print-output to additional file."""

        print(text)
        with open('{}.txt'.format(self.unique_log_name), mode='a') as file_object:
            if isinstance(text, dict):
                text = json.dumps(text)
            print(text, file=file_object)

    def run(self, dataset_names, model_name, folds=5, test=True, validation=True, use_split_idx=None, verbose=True, wrapper=LeaveOneGroupOutWrapper):
        """
        Trains a model on provided data. It can be declared whether a train/test, train/validation or train/validation/test split is required.
        """

        splitter = wrapper(self.dataset)

        self.test_mode = test
        self.validation_mode = validation

        # setup a unique name for suffixes or prefixes in the process
        self.unique_log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # setup a directory for tensorboard-logs
        tb_log_dir = setup_tb_log_dir(
            self.unique_log_name, dataset_names, model_name)

        test_acc = []
        test_loss = []

        todo_splits = list(range(folds)) if not isinstance(
            use_split_idx, int) else [use_split_idx]

        for i in todo_splits:
            test_data = []
            valid_data = []
            end_message = "No more folds in this dataset. Finished after {} folds.".format(
                str(i + 1))

            if test and validation:
                train_data, test_data = splitter.get_split(i)
                # All possible leave one group out folds were processed. Move on to next dataset
                if not train_data:
                    print(end_message)
                    break
                train_data, valid_data = LeaveOneGroupOutWrapper(train_data).get_split(
                    random.randint(0, len(set(train_data.get_groups())) - 1), data_transform_steps=self.data_transform_steps)
            elif not test and validation:
                train_data, valid_data = splitter.get_split(
                    i, data_transform_steps=self.data_transform_steps)
                # All possible leave one group out folds were processed. Move on to next dataset
                if not train_data:
                    print(end_message)
                    break

            elif not validation and test:
                train_data, test_data = splitter.get_split(
                    i, data_transform_steps=self.data_transform_steps)
                # All possible leave one group out folds were processed. Move on to next dataset
                if not train_data:
                    print(end_message)
                    break

            else:
                print("think about this again...")

            acc, loss = self._run_training(train_data, valid_data=valid_data, test_data=test_data,
                                           tb_log_dir=tb_log_dir, split_num=i, verbose=verbose)

            test_acc.append(acc)
            test_loss.append(loss)

        return sum(test_acc) / len(test_acc), sum(test_loss) / len(test_loss)
