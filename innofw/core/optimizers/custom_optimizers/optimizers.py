from torch.optim import SGD as Sgd, Adam


class SGD:
    """
        Class defines a wrapper of the torch optimizer to illustrate
         how to use custom optimizer implementations in the innofw

         Attributes
         ----------
         optimizer
            optimizer from torch framework
    """
    optimizer = Sgd


from innofw.core.optimizers import Optimizer


class ADAM(Optimizer):
    """
    Class defines a wrapper of the torch optimizer to illustrate
    how to use custom optimizer implementations in the innofw
    Attributes
    ----------
    optimizer : torch.optim
        optimizer from torch framework
    """

    def __init__(self, *args, **kwargs):
        super().__init__(optimizer=None)
        self.optim = Adam(*args, **kwargs)
