# cython: boundscheck=False

#cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
#np.import_array()

cdef double *c_k = [1.0 / (2.0 * (2.0 - 2 ** (1. / 3))),
       (1.0 - 2 ** (1. / 3)) / (2.0 * (2.0 - 2 ** (1. / 3))),
       (1.0 - 2 ** (1. / 3)) / (2.0 * (2.0 - 2 ** (1. / 3))),
       1.0 / (2.0 * (2.0 - 2 ** (1. / 3)))]
cdef double *d_k = [1.0 / (2.0 - 2 ** (1. / 3)),
       -2 ** (1. / 3) / (2.0 - 2 ** (1. / 3)),
       1.0 / (2.0 - 2 ** (1. / 3)),
       0]

def simulate_steps(double t_start, const int nbr_steps, double[:] x, double[:] v, const double h, const int N, const double alpha):
    for _ in range(nbr_steps):
        for j in range(4):
            for i in range(N):
                x[i] = x[i] + c_k[j] * h * v[i]
            for i in range(N):
                v[i] = v[i] + d_k[j] * h * a(x, i, N, alpha)
        t_start += h
    return t_start, x, v


cdef double a(const double[:] x, const int i, const int N, const double alpha):
    if i == 0:
        return x[i + 1] - 2 * x[i] + alpha * ((x[i + 1] - x[i]) ** 2 - x[i] ** 2)
    elif i == N - 1:
        return x[i - 1] - 2 * x[i] + alpha * (x[i] ** 2 - (x[i] - x[i - 1]) ** 2)
    else:
        return x[i + 1] + x[i - 1] - 2 * x[i] + alpha * ((x[i + 1] - x[i]) ** 2 - (x[i] - x[i - 1]) ** 2)


def simulate_steps_damped(double t_start,
                            const int nbr_steps,
                            double[:] x,
                            double[:] v,
                            const double h,
                            const int N,
                            const double alpha,
                            const double gamma,
                            const int input_duration,
                            const double input_amplitude,
                            const double[:] input_seq,
                            int input_0_steps_done):
    cdef int cur_input_idx = 0
    cdef int cur_steps = 0
    while input_0_steps_done + nbr_steps >= input_duration:
        cur_steps = input_duration - input_0_steps_done
        # ----- BEGIN simulate_steps_damped_one_input -----
        #t_start, x, v = simulate_steps_damped_one_input(t_start, cur_steps, x, v, h, N, alpha, gamma, input_duration, input_amplitude, input_seq[cur_input_idx])
        for _ in range(cur_steps):
            for j in range(4):
                for i in range(N):
                    x[i] = x[i] + c_k[j] * h * v[i]
                for i in range(N):
                    v[i] = v[i] + d_k[j] * h * a_damped(x, v, i, N, alpha, gamma, input_seq[cur_input_idx], input_amplitude)
            t_start += h
        # ----- END simulate_steps_damped_one_input -----
        nbr_steps -= cur_steps
        cur_input_idx += 1
        input_0_steps_done = 0
    if nbr_steps > 0:
        # ----- BEGIN simulate_steps_damped_one_input -----
        #t_start, x, v = simulate_steps_damped_one_input(t_start, nbr_steps, x, v, h, N, alpha, gamma, input_duration, input_amplitude, input_seq[cur_input_idx])
        for _ in range(nbr_steps):
            for j in range(4):
                for i in range(N):
                    x[i] = x[i] + c_k[j] * h * v[i]
                for i in range(N):
                    v[i] = v[i] + d_k[j] * h * a_damped(x, v, i, N, alpha, gamma, input_seq[cur_input_idx], input_amplitude)
            t_start += h
        # ----- END simulate_steps_damped_one_input -----
        input_0_steps_done += nbr_steps
    return t_start, x, v, input_0_steps_done, cur_input_idx


def simulate_steps_damped_multi_input(double t_start,
                            const int nbr_steps,
                            double[:] x,
                            double[:] v,
                            const double h,
                            const int N,
                            const double alpha,
                            const double gamma,
                            const int input_duration,
                            const double input_amplitude,
                            const double[:, :] input_seq,
                            int input_0_steps_done):
    cdef int cur_input_idx = 0
    cdef int cur_steps = 0
    while input_0_steps_done + nbr_steps >= input_duration:
        cur_steps = input_duration - input_0_steps_done
        # ----- BEGIN simulate_steps_damped_one_input -----
        #t_start, x, v = simulate_steps_damped_one_input(t_start, cur_steps, x, v, h, N, alpha, gamma, input_duration, input_amplitude, input_seq[cur_input_idx])
        for _ in range(cur_steps):
            for j in range(4):
                for i in range(N):
                    x[i] = x[i] + c_k[j] * h * v[i]
                for i in range(N):
                    v[i] = v[i] + d_k[j] * h * a_damped(x, v, i, N, alpha, gamma, input_seq[cur_input_idx, i], input_amplitude)
            t_start += h
        # ----- END simulate_steps_damped_one_input -----
        nbr_steps -= cur_steps
        cur_input_idx += 1
        input_0_steps_done = 0
    if nbr_steps > 0:
        # ----- BEGIN simulate_steps_damped_one_input -----
        #t_start, x, v = simulate_steps_damped_one_input(t_start, nbr_steps, x, v, h, N, alpha, gamma, input_duration, input_amplitude, input_seq[cur_input_idx])
        for _ in range(nbr_steps):
            for j in range(4):
                for i in range(N):
                    x[i] = x[i] + c_k[j] * h * v[i]
                for i in range(N):
                    v[i] = v[i] + d_k[j] * h * a_damped(x, v, i, N, alpha, gamma, input_seq[cur_input_idx, i], input_amplitude)
            t_start += h
        # ----- END simulate_steps_damped_one_input -----
        input_0_steps_done += nbr_steps
    return t_start, x, v, input_0_steps_done, cur_input_idx



cdef simulate_steps_damped_one_input(double t_start,
                            const int nbr_steps,
                            double[:] x,
                            double[:] v,
                            const double h,
                            const int N,
                            const double alpha,
                            const double gamma,
                            const int input_duration,
                            const double input_amplitude,
                            int cur_input):
    for _ in range(nbr_steps):
        for j in range(4):
            for i in range(N):
                x[i] = x[i] + c_k[j] * h * v[i]
            for i in range(N):
                v[i] = v[i] + d_k[j] * h * a_damped(x, v, i, N, alpha, gamma, cur_input, input_amplitude)
        t_start += h
    return t_start, x, v


cdef double a_damped(const double[:] x, const double[:] v, const int i, const int N, const double alpha, const double gamma, const double cur_input, const double input_amplitude):
    return - gamma * v[i] + give_input(i, cur_input, input_amplitude) + a(x, i, N, alpha)


cdef double give_input(const int i, double input_state, const double input_amplitude):
    # if input_state == 0:
    #     input_state = -1
    return input_amplitude * input_state
