import abc
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict


class BaseModel(nn.Module):
    def __init__(self, seed=10101):
        self._metrics = defaultdict(list)
        self._epoch = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_inputs(cls, iterator):
        pass

    @abc.abstractmethod
    def _compute_metrics(self, target_y, pred_y, loss, save=True):
        pass

    @abc.abstractmethod
    def fit(self, optimizer, loss_fn, data_loader, validation_data_loader, num_epochs, logger, verbose=True):
        pass

    @classmethod
    def to_np(cls, x):
        # convert Variable to numpy array
        return x.data.cpu().numpy()

    @classmethod
    def to_var(cls, x, use_gpu=True):
        if torch.cuda.is_available() and use_gpu:
            x = x.cuda()
        return Variable(x)

    @classmethod
    def to_tensor(cls, x):
        # noinspection PyUnresolvedReferences
        tensor = torch.from_numpy(x).float()
        return tensor

    def _avg_metrics(self):
        res = {}
        for k, v in self._metrics.items():
            res[k] = sum(v) / len(v)
        return res

    @classmethod
    def show_env_info(cls):
        print('__Python VERSION:', sys.version)
        print('__CUDA VERSION')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        print("OS: ", sys.platform)
        print("PyTorch: ", torch.__version__)
        print("Numpy: ", np.__version__)
        use_cuda = torch.cuda.is_available()
        print("CUDA is available", use_cuda)

    def _log_data(self, logger, data_dict):
        for tag, value in data_dict.items():
            logger.scalar_summary(tag, value, self._epoch + 1)

    def _log_grads(self, logger):
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, self.to_np(value), self._epoch + 1)
            logger.histo_summary(tag + '/grad', self.to_np(value.grad), self._epoch + 1)

    def _log_and_reset(self, logger, data=None):
        if not data:
            averaged = self._avg_metrics()
            self._log_data(logger, averaged)
            self._log_grads(logger)
            self._metrics = defaultdict(list)
        else:
            self._log_data(logger, data)

    def evaluate(self, loader, loss_fn=None, switch_to_eval=False, **kwargs):
        if switch_to_eval:
            self.eval()
        iterator = iter(loader)
        iter_per_epoch = len(loader)
        all_predictions = np.array([])
        all_targets = np.array([])
        losses = []
        for i in range(iter_per_epoch):
            inputs, targets = self._get_inputs(iterator)
            pred_y, extra = self.predict(inputs, **kwargs)
            target_y = self.to_np(targets).squeeze()
            if loss_fn:
                loss = loss_fn(pred_y, targets)
                losses.append(loss.data[0])
            all_targets = np.append(all_targets, target_y)
            all_predictions = np.append(all_predictions, self.to_np(pred_y))
        computed_metrics = self._compute_metrics(all_targets, all_predictions, losses, save=False)
        if switch_to_eval:
            # switch back to train
            self.train()
        return computed_metrics

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        """
        Load model
        :param path: string path to file containing model
        :return: instance of this class
        :rtype: BaseModel
        """
        model = torch.load(path)
        if isinstance(model, dict):
            model = model['state_dict']
        return model
