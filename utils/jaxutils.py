from typing import Any, Optional


class EarlyStopping:
    """
    This beautiful piece of code was briefly modified from [jaxchem's](https://github.com/deepchem/jaxchem/blob/master/jaxchem/utils/early_stop.py#L8-L55).

    Early stops the training if score doesn't improve after a given patience.
    """
    def __init__(self,
                 patience: int = 10,
                 delta: int = 0,
                 greater_is_better: bool = True):
        """
        Parameters
        ----------
        patience : int
            How long to wait after last time validation loss improved, default to be 10.
        delta : float
            Minimum change in the monitored quantity to qualify as an improvement, default to be 0.
        is_greater_better : bool
            Whether the greater score is better or not default to be True.
        """
        self.patience = patience
        self.delta = delta
        self.greater_is_better = greater_is_better
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_checkpoint = None
        self.stop = False
        self.__tmp_best_score = 0.0

    def update(self, score: float, checkpoint: Any = None):
        """Update early stopping counter.
        Parameters
        ----------
        score : float
            validation score per epoch.
        checkpoint : Any
            all parameters and states of training model.
        """
        tmp_score = score if self.greater_is_better else -score
        if self.best_score is None:
            self.__tmp_best_score = tmp_score
            self.best_score = score
            self.best_checkpoint = checkpoint
        elif tmp_score < self.__tmp_best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.__tmp_best_score = tmp_score
            self.best_score = score
            self.best_checkpoint = checkpoint
            self.counter = 0
