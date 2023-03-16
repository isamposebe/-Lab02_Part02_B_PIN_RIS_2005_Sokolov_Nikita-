import numpy as np
from scipy.optimize import minimize
# Построение линий уровня функции для определения интервала экстремума
import matplotlib.pyplot as plt
# Наискорейший градиентный спуск с использованием алгоритма золотого сечения
from scipy.optimize import minimize_scalar
def f(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2


def nelder_mead(f, x_start, step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
    Nelder Mead algorithm.
    '''
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    res_list = [prev_best]
    no_improv = 0
    res = [x_start]

    for i in range(dim):
        x = np.array(x_start, dtype=float)
        x[i] += step
        res.append(x)

    # simplex iter
    iters = 0
    while True:
        # order
        res.sort(key=f)
        best = f(res[0])

        # break after max_iter iterations
        if max_iter and iters >= max_iter:
            return res[0], best

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best)
        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0], best

        # centroid
        x0 = np.mean(res[:-1], axis=0)

        # reflection
        xr = x0 + alpha * (x0 - res[-1])
        rs = f(xr)

        if best <= rs < f(res[-2]):
            del res[-1]
            res.append(xr)
            continue

        # expansion
        if rs < best:
            xe = x0 + gamma * (xr - x0)
            re = f(xe)
            if re < rs:
                del res[-1]
                res.append(xe)
                continue
            else:
                del res[-1]
                res.append(xr)
                continue

        # contraction
        xc = x0 + rho * (res[-1] - x0)
        rc = f(xc)

        if rc < f(res[-1]):
            del res[-1]
            res.append(xc)
            continue

        # shrink
        x1 = res[0]
        new_res = []

        for xr in res:
            new_res.append(x1 + sigma * (xr - x1))

        res = new_res

        # increment iterations counter
        iters += 1

        # append result to list
        res_list.append(best)

    return res[0], best




x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y, 3])

plt.contour(X, Y, Z, levels=[i for i in range(0, 100, 10)])
plt.show()

# Задание начальных значений переменных
x_start = [0.5, 2.5, 4.5]

# Минимизация функции методом Нелдера-Мида
xmin, ymin = nelder_mead(f, x_start)

# Вывод результатов
print('Минимум функции:', ymin)
print('Точка минимума:', xmin)

def f1(x):
    return (x-1)**2

res = minimize_scalar(f1, method='golden')
print('Минимум функции f1:', res.x)