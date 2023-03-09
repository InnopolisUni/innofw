import functools
import math


# todo: refactor
def lr_cyclic(decay=0.2, decay_step=10, cyclic_decay=0.9, cyclic_len=25):
    def lr_step(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        cyclic_n = epoch // cyclic_len
        epoch -= cyclic_len * cyclic_n
        # cyclic_decay=0.99
        # decay=0.5
        return (decay * (cyclic_decay**cyclic_n)) ** (epoch // decay_step)

    def lr_exp(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        cyclic_n = epoch // cyclic_len
        epoch -= cyclic_len * cyclic_n
        return math.exp(-decay * epoch) * (cyclic_decay**cyclic_n)

    def lr_cos(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        """Returns alpha * cos((epoch - a) / (b - a) * pi) + beta
        a, b - left and right bounds of i-th cycle, i=0,1,...
        So lr = lr0 * lr_cos
        """
        n = 0
        k = 1
        cyclic_sum = cyclic_len
        while epoch >= cyclic_sum:
            k *= decay_step
            n += 1
            cyclic_sum += k * cyclic_len
        b = cyclic_sum
        a = b - k * cyclic_len
        alpha = 0.5 * (1 - decay)
        beta = 0.5 * (1 + decay)

        return (alpha * math.cos((epoch - a) / (b - a) * math.pi) + beta) * (
            cyclic_decay**n
        )

    def lr_poly(epoch, decay, decay_step, cyclic_decay, cyclic_len):
        cyclic_n = epoch // cyclic_len
        epoch -= cyclic_len * cyclic_n
        return (1 - epoch / (1.048 * cyclic_len)) ** 0.9 * (
            cyclic_decay**cyclic_n
        )

    return functools.partial(
        #         lr_exp,
        #         lr_poly,
        lr_cos,
        decay=decay,
        decay_step=decay_step,
        cyclic_decay=cyclic_decay,
        cyclic_len=cyclic_len,
    )
