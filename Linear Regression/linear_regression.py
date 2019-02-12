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
    means = np.linspace(0, 1, order)
    scale = 0.1
    result = [1]
    if order > 0:
        for mean in means:
            result += [np.exp(-(x - mean)**2 / (2 * scale ** 2))]
    return result


def phi(x, base, order):
    return np.array([base(x[i][0], order) for i in range(len(x))])


def give_omega_mle(phi_matrix, y_observed):
    big_inverse = inverse(np.dot(np.transpose(phi_matrix), phi_matrix))
    return np.dot(big_inverse, np.dot(np.transpose(phi_matrix), y_observed))


def give_omega_map(phi_matrix, lambda_regul, order, y_observed):
    big_inverse = inverse(np.dot(np.transpose(phi_matrix), phi_matrix) + lambda_regul * np.identity(order + 1))
    return np.dot(big_inverse, np.dot(np.transpose(phi_matrix), y_observed))


def y_predicted(x, parameters, base, order):
    value = 0
    if base == polynomial:
        for i in range(order + 1):
            value = value + parameters[i][0] * base(x, order)[i]
        return value
    elif base == trigonometric:
        for i in range(2 * order + 1):
            value += parameters[i][0] * base(x, order)[i]
        return value
    elif base == gaussian:
        for i in range(order + 1):
            value = value + parameters[i][0] * base(x, order)[i]
        return value


def generate_y(x_training_set, y_training_set, base, order, interval_to_predict, number_of_points):
    phi_matrix = phi(x_training_set, base, order)
    parameters_mle = give_omega_mle(phi_matrix, y_training_set)
    x_predicted_set = np.reshape(np.linspace(interval_to_predict[0], interval_to_predict[1], number_of_points),
                                 (number_of_points, 1))
    y_predicted_set = [[y_predicted(x_predicted_set[i][0], parameters_mle, base, order)]
                       for i in range(len(x_predicted_set))]
    return x_predicted_set, y_predicted_set


def cross_validation(full_x_training_set, full_y_training_set, base, order, k_fold):
    length_full_training_set = len(full_x_training_set)
    length_validation_subset = int(length_full_training_set / k_fold)
    x_validation_sub_sets = [full_x_training_set[i * length_validation_subset: (i+1) * length_validation_subset] for i in range(k_fold - 1)]
    x_validation_sub_sets += [full_x_training_set[(k_fold - 1) * length_validation_subset:]]
    x_training_sub_sets = [np.concatenate((full_x_training_set[:length_validation_subset * i],
                                         full_x_training_set[(i + 1) * length_validation_subset:])) for i in range(k_fold)]
    y_validation_sub_sets = [full_y_training_set[i * length_validation_subset: (i + 1) * length_validation_subset] for i
                             in range(k_fold - 1)]
    y_validation_sub_sets += [full_y_training_set[(k_fold - 1) * length_validation_subset:]]
    y_training_sub_sets = [np.concatenate((full_y_training_set[:length_validation_subset * i],
                                           full_y_training_set[(i + 1) * length_validation_subset:])) for i in
                           range(k_fold)]
    test_prediction = [generate_y(x_training_sub_sets[i], y_training_sub_sets[i], base, order,
                                  [x_validation_sub_sets[i], x_validation_sub_sets[i]], length_validation_subset) for i in range(k_fold)]
    return test_prediction


def rmse(y_observed, y_predicted, number_of_points):
    differences = [y_observed[i][0] - y_predicted[i][0] for i in range(len(y_observed))]
    squared_differences = [elem ** 2 for elem in differences]
    sum_squares = np.sum(squared_differences)
    result = np.sqrt((1. / number_of_points) * sum_squares)
    return result


def compare_models(start_order, end_order, base, x_full_set, y_full_set):
    number_of_points = len(x_full_set)
    rmse_by_order = []
    for order in range(start_order, end_order + 1):
        cross = cross_validation(x_full_set, y_full_set, base, order, number_of_points)
        x_cross, y_cross = [cross[i][0] for i in range(number_of_points)], [cross[i][1] for i in range(number_of_points)]
        rmse_by_order += [rmse(y_full_set, y_cross, number_of_points)]
    return rmse_by_order


def sigma_squared_mle(x_full_set, y_full_set, base, start_order, end_order):
    number_of_points = len(x_full_set)
    sigma_squared_mle_by_order = []
    for order in range(start_order, end_order + 1):
        prediction = generate_y(x_full_set, y_full_set, base, order, [x_full_set[0], x_full_set[-1]], number_of_points)[1]
        sigma_squared_mle_by_order += [rmse(y_full_set, prediction, number_of_points)]
    return sigma_squared_mle_by_order


def generate_y_ridge(x_training_set, y_training_set, base, order, interval_to_predict, number_of_points):
    phi_matrix = phi(x_training_set, base, order)
    parameters_map = give_omega_map(phi_matrix, 5, order, y_training_set)
    x_predicted_set = np.reshape(np.linspace(interval_to_predict[0], interval_to_predict[1], number_of_points),
                                 (number_of_points, 1))
    y_predicted_set = [[y_predicted(x_predicted_set[i][0], parameters_map, base, order)]
                       for i in range(len(x_predicted_set))]
    return x_predicted_set, y_predicted_set


fig, ax = plt.subplots()
plt.scatter(X, Y, s=20, marker="x", c='b', label="training data")

## question 1/a)
#x_predicted_set_poly_0, y_predicted_set_poly_0 = generate_y(X, Y, polynomial, 0, [-0.3, 1.3], 200)
#x_predicted_set_poly_1, y_predicted_set_poly_1 = generate_y(X, Y, polynomial, 1, [-0.3, 1.3], 200)
#x_predicted_set_poly_2, y_predicted_set_poly_2 = generate_y(X, Y, polynomial, 2, [-0.3, 1.3], 200)
#x_predicted_set_poly_3, y_predicted_set_poly_3 = generate_y(X, Y, polynomial, 3, [-0.3, 1.3], 200)
#x_predicted_set_poly_11, y_predicted_set_poly_11 = generate_y(X, Y, polynomial, 11, [-0.3, 1.3], 200)
#plt.plot(x_predicted_set_poly_0, y_predicted_set_poly_0, label="order 0")
#plt.plot(x_predicted_set_poly_1, y_predicted_set_poly_1, label="order 1")
#plt.plot(x_predicted_set_poly_2, y_predicted_set_poly_2, label="order 2")
#lt.plot(x_predicted_set_poly_3, y_predicted_set_poly_3, label="order 3")
#plt.plot(x_predicted_set_poly_11, y_predicted_set_poly_11, label="order 11")

## question 1/b)
#x_predicted_set_trigo_1, y_predicted_set_trigo_1 = generate_y(X, Y, trigonometric, 1, [-1.2, 1.2], 200)
#x_predicted_set_trigo_11, y_predicted_set_trigo_11 = generate_y(X, Y, trigonometric, 11, [-1.2, 1.2], 200)
#plt.plot(x_predicted_set_trigo_1, y_predicted_set_trigo_1, label="order 1")
#plt.plot(x_predicted_set_trigo_11, y_predicted_set_trigo_11, label="order 11")


#plt.scatter(x_cross, y_cross, s=20, marker="x", c='r')
#plt.plot(x_predicted_set_poly, y_predicted_set_poly, c='r')
#plt.plot(x_predicted_set_trigo, y_predicted_set_trigo, c='g')

'''
## question 1/c)
comparison = compare_models(0, 10, trigonometric, X, Y)
sigma_mle_list = sigma_squared_mle(X, Y, trigonometric, 0, 10)
list_orders = [i for i in range(0, 11)]
plt.plot(list_orders, comparison, label="RMSE")
plt.plot(list_orders, sigma_mle_list, label="MLE of sigma squared")

# A régler selon les question
ax.set_xlim(-1.3, 1.3)

#ax.set_ylim(-1.3, 1.3)
plt.title("Linear regression of order 1 and 11 with trigonometric basis")
#plt.title("RMSE on testing data with leave-one-out cross validation and maximum likelihood sigma squared for 0 to 10 orders with trigonometric basis")
plt.legend()
plt.show()
'''

plt.title("Linear regression with regulariser lambda set at 5, base of 20 gaussians")
ax.set_ylim(-1.3, 1.3)
x_predicted_set_ridge_underfitting, y_predicted_set_ridge_underfitting = generate_y_ridge(X, Y, gaussian, 20, [-0.3, 1.3], 200)
plt.plot(x_predicted_set_ridge_underfitting, y_predicted_set_ridge_underfitting, label="under-fitting, lambda 5")
plt.legend()
plt.show()