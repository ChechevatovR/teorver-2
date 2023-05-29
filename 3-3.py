import math
import random
import typing
import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_continuous

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive', 'tab:cyan']


def pdf_unsafe(x: float) -> float:
    return 4 * math.log(x) ** 3 / x


def pdf(x: float) -> float:
    if x < 1 or x > math.e:
        return 0
    return pdf_unsafe(x)


def cdf_unsafe(x: float) -> float:
    return math.log(x) ** 4


def cdf(x: float) -> float:
    if x <= 1:
        return 0
    if x >= math.e:
        return 1
    return cdf_unsafe(x)


# Листинг 2
def cdf_inverse(y: float) -> float:
    return math.exp(y ** .25)


# Листинг 3
def rejection_sampling(xmin: float, xmax: float, pdf: typing.Callable[[float], float], pmax: float):
    while True:
        x = random.uniform(xmin, xmax)
        y = random.uniform(0, pmax)
        if y < pdf(x):
            return x


def getDistByRejection(pmax: float):
    return rejection_sampling(1, math.e, pdf_unsafe, pmax)


# Листинг 1
class Dist(rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(0, 1, math.e, xtol=xtol, seed=seed)

    def _pdf(self, x, *args):
        return pdf_unsafe(x)


def plot_distribution(dist: typing.Callable[[], float], title: str, color, n: int = 100_000):
    fig = plt.figure(figsize=(19, 6), layout='tight')
    ax = fig.subplots(1, 1)
    ax.hist([dist() for _ in range(n)], bins=30, density=True, rwidth=0.9, label=title, color=color)
    xs = np.linspace(1, 3, 100)
    ax.plot(xs, [pdf(x) for x in xs], label='y=pdf(x)', color='black', linewidth=5)
    ax.legend()
    fig.suptitle(title)
    plt.savefig(f'img/3. {title}.svg')
    fig.show()


# Plot
funs = [Dist(1e-3).rvs, lambda: getDistByRejection(1.5), lambda: cdf_inverse(random.uniform(0, 1))]
labels = ['rv_continuos', 'Rejection sampling', 'Inverse of F']
# for ind, (fun, label) in enumerate(zip(funs, labels)):
#     plot_distribution(fun, label, color=COLORS[ind])

# Performance
funs = [Dist(1e-3).rvs, Dist(1e-10).rvs, lambda: getDistByRejection(1.5), lambda: getDistByRejection(25), lambda: cdf_inverse(random.uniform(0, 1))]
labels = ['rv_continuos, xtol=1e-3', 'rv_continuos, xtol=1e-10', 'Rejection sampling, pmax=1.5', 'Rejection sampling, pmax=25', 'Inverse of F']
ns = [10, 100, 1_000, 10_000]

fig = plt.figure(figsize=(19, 6), layout='tight')
ax = fig.subplots(1, 1)
ax.set_yscale('log')
ax.set_ylabel('Время на точку, сек')
ax.set_xlabel('Количество точек')
ax.set_xticks(range(0, len(ns)), map(str, ns))
print('\t'.join(['Distribution'] + list(map(str, ns))))
for ind, (fun, label) in enumerate(zip(funs, labels)):
    times = []
    times_per = []
    for n in ns:
        time = timeit.timeit(fun, number=n)
        times.append(time)
        times_per.append(time / n)
    print(label, end='\t')
    print('\t'.join(map(lambda x: '%.3e' % x, times_per)))
    ax.plot(times_per, label=label)
ax.legend(loc='upper left')
fig.suptitle('Время работы алгоритмов')
plt.savefig(f'img/3. Times.svg')
fig.show()
