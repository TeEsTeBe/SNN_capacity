import numpy as np
import matplotlib.pyplot as plt


def to_alpha(squared_corr_coeff, var_ratio=1.):
    return 1 / (np.sqrt((1 / squared_corr_coeff - 1) / var_ratio) + 1)


def to_squared_corr(alpha, var_ratio=1.):
    return 1 / (1 + var_ratio * (1 - alpha) ** 2 / alpha ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    y_alpha = True
    n_steps = 1000000
    correlations = []
    alpha_list = np.arange(0., 1.01, 0.01)

    signal1 = np.random.normal(4, 1, size=n_steps)
    signal2 = np.random.normal(4, 2, size=n_steps)
    # signal1 = np.random.uniform(-1, 1, size=n_steps)
    # signal2 = np.random.uniform(-1, 1, size=n_steps)

    var_ratio = np.var(signal2) / np.var(signal1)
    for alpha in alpha_list:
        combined_signal = alpha * signal1 + (1 - np.abs(alpha)) * signal2
        print(np.var(combined_signal))
        corr = np.corrcoef(signal1, combined_signal)[1][0]
        correlations.append(corr)

    correlations = np.array(correlations)
    plt.figure(figsize=(5, 5))

    if y_alpha:
        plt.plot(correlations ** 2, alpha_list, label='measured')
        plt.plot(to_alpha(correlations ** 2, var_ratio=var_ratio), alpha_list, label='lin?')
        plt.plot(correlations ** 2, to_alpha(correlations ** 2, var_ratio=var_ratio), ':', color='k',
                 label='calculated')
        plt.xlabel('squared correlation')
        plt.ylabel(r'$\alpha$')
    else:
        plt.plot(alpha_list, correlations ** 2, label='measured')
        plt.plot(alpha_list, to_alpha(correlations ** 2, var_ratio=var_ratio), label='lin?')
        plt.plot(alpha_list, to_squared_corr(alpha_list, var_ratio=var_ratio), ':', color='k', label='calculated')
        plt.xlabel(r'$\alpha$')
        plt.ylabel('squared correlation')

    plt.title(fr'$\sigma^2(sig1) = {round(np.var(signal1), 2)}$, $\sigma^2(sig2) = {round(np.var(signal2), 2)}$')

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
