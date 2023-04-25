def dummy_func(epoch):
    print("here")
    print(1 / 0)
    return (epoch * 0.01) + 0.05


def func():
    return lambda epoch: epoch // 30


lambda1 = func
