import math
import random
import statistics

p = 0.25
q = 1 - p
mu = q / p
eps = .01
delta = .05
sqrt_DX = 12 ** .5
Phi = statistics.NormalDist(0, 1).cdf


def get_x_def(p: float):
    res = 0
    while True:
        x = random.random()
        if x <= p:
            return res
        res += 1


def get_x_float(p: float):
    q = 1 - p
    point = random.random()
    cur_p = p
    cur_border = cur_p
    cur_x = 0
    # print(f'{point=}')
    while True:
        if point <= cur_border:
            # print(f'{cur_x=}')
            return cur_x
        cur_x += 1
        cur_p *= q
        cur_border += cur_p
        # print(f'{cur_border=}')


def get_sn(p: float, n: int):
    s = 0
    for _ in range(n):
        # s += get_x_float(p)
        s += get_x_def(p)
    return s / n


def run_experiment(n: int):
    success = 0
    total = 100
    for i in range(total):
        s = get_sn(p, n)
        diff = math.fabs(s - mu)
        print(i, s, diff, sep='\t')
        if diff <= eps:
            success += 1

    print(n, success, success / total, 1 - delta, sep='\t')


def get_n_for_clt():
    n = 1
    while 2 * Phi(eps * (n ** .5) / sqrt_DX) < 2 - delta:
        n += 1
    return n


def main():
    n1 = 2_400_000
    n2 = get_n_for_clt()
    print(n2)
    run_experiment(n1)
    run_experiment(n2)


if __name__ == '__main__':
    main()
