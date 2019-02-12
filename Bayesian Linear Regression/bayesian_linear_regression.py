# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

import numpy as np
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10 * X ** 2) + 0.1 * np.sin(100 * X)


def inverse(x):
    try:
        inverse_result = np.linalg.inv(x)
        return inverse_result
    except np.linalg.LinAlgError:
    # Not invertible. Skip this one.
        pass


# <-------------------------- LML and Grad LML functions -------------------------> #
def lml(alpha, beta, Phi, Y):
    N = float(len(Y))
    I = np.identity(int(N))
    phiT = np.transpose(Phi)
    group = alpha * np.dot(Phi, phiT) + beta * I
    return ((-N/2.)*np.log(2*np.pi) - (1./2)*np.log(np.linalg.det(group))
            - 1./2 *np.dot(np.transpose(Y), np.dot(inverse(group), Y)))[0][0]


def grad_lml(alpha, beta, Phi, Y):
    N = float(len(Y))
    I = np.identity(int(N))
    phiT = np.transpose(Phi)
    group = alpha * np.dot(Phi, phiT) + beta * I
    group = (1. / 2) * (group + np.transpose(group))
    YT = np.transpose(Y)
    phiphi = np.dot(Phi, phiT)
    A1 = np.trace(np.dot(inverse(group), phiphi))
    A2 = np.dot(YT, np.dot(inverse(group), np.dot(phiphi, np.dot(inverse(group), Y))))
    B1 = np.trace(inverse(group))
    B2 = np.dot(YT, np.dot(inverse(group), np.dot(inverse(group), Y)))
    result = -1./2 * np.array([A1 - A2, B1 - B2])
    return np.array([result[0][0][0], result[1][0][0]])


# <--------------- Code for questions b, c and d -------------------------> #


def polynomial(x, order):
    result = [1]
    if order > 0:
        result += [x**i for i in range(1, order + 1)]
    return result


def trigonometric(x, order):
    result = [1]
    if order > 0:
        for j in range(1, order + 1):
            result += [np.sin(2 * np.pi * j * x)]
            result += [np.cos(2 * np.pi * j * x)]
    return result


def gaussian(x, order):
    means = np.linspace(-0.5, 1.15, order)
    scale = 0.1
    result = [1]
    if order > 0:
        for mean in means:
            result += [np.exp(-(x - mean)**2 / (2 * scale ** 2))]
    return result


def phi(x, base, order):
    return np.array([base(x[i][0], order) for i in range(len(x))])


def grad_descent(function_to_minimize, grad_of_function, start_point_coordinates, step_size, number_of_steps):
    points = [start_point_coordinates]
    for i in range(number_of_steps):
        slope = grad_of_function(points[-1])
        next_point = points[-1] - step_size * slope
        current_height = function_to_minimize(points[-1])
        next_height = function_to_minimize(next_point)
        points = points + [next_point]
    return "final coordinates are :", points[-1], " the height is: ", function_to_minimize(points[-1]),\
           " the corresponding step size is: ", step_size, points


def grad_ascent_optim(function_to_minimize, grad_of_function, start_point_coordinates, first_step_size, number_of_steps, Phi, Y):
    points = [start_point_coordinates]
    step_size = first_step_size
    for step in range(number_of_steps):
        slope = grad_of_function(points[-1][0], points[-1][1], Phi, Y)
        next_point = points[-1] + step_size * slope
        current_height = function_to_minimize(points[-1][0], points[-1][1], Phi, Y)
        next_height = function_to_minimize(next_point[0], next_point[1], Phi, Y)
        if (next_height < current_height) or (next_point[1] < 0):
            step_size = step_size / 2
            step -= 1
        else:
            points = points + [next_point]
            step_size = step_size * 1.5
    print("height difference is ",
          function_to_minimize(points[-1][0], points[-1][1], Phi, Y) - function_to_minimize(points[-2][0],
                                                                                            points[-2][1], Phi, Y),
          " and alpha, beta are ", points[-1])
    return "final coordinates are :", points[-1], " the height is: ", current_height,\
           " the corresponding step size is: ", step_size, points


def grad_ascent(function_to_minimize, grad_of_function, start_point_coordinates, step_size, number_of_steps, Phi, Y):
    points = [start_point_coordinates]
    for i in range(number_of_steps):
        slope = grad_of_function(points[-1][0], points[-1][1], Phi, Y)
        next_point = points[-1] + step_size * slope
        current_height = function_to_minimize(points[-1][0], points[-1][1], Phi, Y)
        next_height = function_to_minimize(points[-1][0], points[-1][1], Phi, Y)
        points = points + [next_point]
    return "final coordinates are :", points[-1], " the height is: ", current_height,\
           " the corresponding step size is: ", step_size, points




def compare_orders(start_order, end_order):
    lml_values = np.zeros(((end_order - start_order + 1), 2))
    for order in range(start_order, end_order + 1):
        print(" Computing order: ", order)
        phi_matrix = phi(X, trigonometric, order)
        max_lml = grad_ascent_optim(lml, grad_lml, np.array([0.2, 0.2]), 0.00001, 500, phi_matrix, Y)
        lml_values[order] = np.array(max_lml[3])
    return lml_values


def omega_sample(phi, y):
    N = 10
    I = np.identity(N)
    phit = np.transpose(phi)
    sn = inverse(I + 10 * np.dot(phit, phi))
    mn = np.dot(sn, 10 * np.dot(phit, y))
    return np.random.multivariate_normal(np.reshape(mn, (1, 10))[0], sn, size=5)


def noise_free_function(x, omega):
    phi_star = phi(x, gaussian, 9)
    return np.dot(phi_star, omega)


def mean_noise_free_function(x):
    N = 10
    I = np.identity(N)
    phi_star = phi(x, gaussian, 9)
    Phi = phi(X, gaussian, 9)
    phit = np.transpose(Phi)
    sn = inverse(I + 10 * np.dot(phit, Phi))
    mn = np.dot(sn, 10 * np.dot(phit, Y))
    return np.dot(np.transpose(mn), np.transpose(phi_star))


def standard_deviation_noise_free_function(x):
    N = 10
    I = np.identity(N)
    phi_star = phi(x, gaussian, 9)
    Phi = phi(X, gaussian, 9)
    phit = np.transpose(Phi)
    sn = inverse(I + 10 * np.dot(phit, Phi))
    mn = np.dot(sn, 10 * np.dot(phit, Y))
    variance = np.dot(phi_star, np.dot(sn, np.transpose(phi_star)))
    sigma2 = np.diag(variance)
    sigma = np.sqrt(sigma2)
    return np.dot(np.transpose(mn), np.transpose(phi_star)) + 2 * sigma, np.dot(np.transpose(mn), np.transpose(phi_star)) - 2 * sigma


def plot_predictions():
    fig, ax = plt.subplots(1, 1, sharex=True)
    plt.scatter(X, Y, label="training data", s=20, marker="x", c='r')
    x = np.reshape(np.linspace(-0.5, 1.5, 200), (200, 1))
    phi_matrix = phi(X, gaussian, 9)
    samples = omega_sample(phi_matrix, Y)
    predictions = np.zeros((5, 200))
    for i in range(5):
        for j in range(200):
            predictions[i][j] = noise_free_function([x[j]], samples[i])
        plt.plot(x, predictions[i], label="sample")
    plt.plot(x, np.reshape(mean_noise_free_function(x), (200, 1)), c='b', label="mean")
    plt.plot(x, np.reshape(standard_deviation_noise_free_function(x)[0], (200, 1)), c='g', label="95% upper bound")
    plt.plot(x, np.reshape(standard_deviation_noise_free_function(x)[1], (200, 1)), c='g', label="95% lower bound")
    ax.fill_between(np.reshape(x, (1, 200))[0], standard_deviation_noise_free_function(x)[1][0],
                    standard_deviation_noise_free_function(x)[0][0], alpha=0.1)
    plt.legend()
    plt.show()

plot_predictions()
'''

# <-------------------------- Code for plots ---------------------------> #
'''
#N = compare_orders(0, 10)

# After 11 gradient ascent optim plot with customised starting points we have :
question_c_values = [[0.05953787, 0.5126303],[0.25259886, 0.17256039], [0.1713283, 0.09402557], [0.13241788, 0.0432884],
                     [0.10693271, 0.0230254], [0.08890008, 0.01532944], [0.07578422, 0.01264163],
                     [0.06591716, 0.01256351], [0.05831886, 0.01423679], [0.05257424, 0.01684657],
                     [0.05077136, 0.01131081]]

question_c_lml = [-27.80190178, -18.2780087, -14.15597575, -9.35550911, -6.92848522, -7.05850596, -9.06790061,
                  -12.2155798, -15.75875175, -19.10983338, -21.26188673]


#alphas = np.array([question_c_values[i][0] for i in range(len(question_c_values))])
#betas = np.array([question_c_values[i][1] for i in range(len(question_c_values))])

fig, ax = plt.subplots()

Order = np.arange(0, 11, 1)
#plt.plot(np.reshape(Order, (11, 1)), alphas,  c='r', label="alpha")
#plt.plot(np.reshape(Order, (11, 1)), betas,  c='b', label="beta")
plt.plot(np.reshape(Order, (11, 1)), question_c_lml,  c='g', label="lml max value")

ax.set_title('LML max value obtained with gradient ascent, trigo base and orders 0 to 10 inclusive')
plt.legend()
plt.show()

#print(lml(1, 1, phi_matrix, Y))

phi_matrix = phi(X, trigonometric, 10)

#print(grad_lml(1, 1, phi_matrix, Y))

# Working values
#alpha = np.arange(0.3, 0.5, 0.01)
#beta = np.arange(0.4, 0.5, 0.01)

alpha = np.arange(0.001, 0.6, 0.005)
beta = np.arange(0.001, 0.3, 0.005)

marginal_likelihood = np.array([[lml(a, b, phi_matrix, Y) for a in alpha] for b in beta])

max_lml = grad_ascent_optim(lml, grad_lml, np.array([0.2, 0.2]), 0.00001, 500, phi_matrix, Y)
# ('final coordinates are :', array([0.41675929, 0.44958621]), ' the height is: ', -27.608846990376925, ' the corresponding step size is: ', 0.01
# 400 steps : ('final coordinates are :', array([0.42443286, 0.44923678]), ' the height is: ', -27.608794641855013, ' the corresponding step size is: ', 0.01,
# 1000 steps : 'final coordinates are :', array([0.42455482, 0.44923134]), ' the height is: ', -27.608794629202777, ' the corresponding step size is: ', 0.01
x_coordinates = [max_lml[-1][i][0] for i in range(len(max_lml[-1]))]
y_coordinates = [max_lml[-1][j][1] for j in range(len(max_lml[-1]))]

A, B = np.meshgrid(alpha, beta)
fig, ax = plt.subplots()
CS = ax.contour(A, B, marginal_likelihood, 40)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Gradient ascent of lml with constant step size of 0.01, 1000 steps')

plt.scatter(x_coordinates, y_coordinates, s=20, marker="x", c='r')

plt.show()
