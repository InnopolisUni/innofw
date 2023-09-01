def func():
    return lambda epoch: epoch // 30


def func2():
    return lambda epoch: 0.95**epoch


lambda1 = func
lambda2 = func2
