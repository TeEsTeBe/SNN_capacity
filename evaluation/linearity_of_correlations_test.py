import numpy as np
import matplotlib.pyplot as plt

from utils.general_utils import to_alpha, to_squared_corr


def main():
    y_alpha = False
    n_steps = 1000000
    correlations = []
    alpha_list = np.arange(0., 1.01, 0.01)

    signal1 = np.random.normal(4, 1, size=n_steps)
    signal2 = np.random.normal(-3, 0.6, size=n_steps)
    # signal1 = np.random.uniform(-1, 1, size=n_steps)
    # signal2 = np.random.uniform(-1, 1, size=n_steps)

    va = np.var(signal1)
    vb = np.var(signal2)
    var_ratio = vb / va
    std_ratio = np.std(signal2) / np.std(signal1)
    var_c_list = []
    var_sub_asdf_list = []
    new_calc = []
    for alpha in alpha_list:
        combined_signal = alpha * signal1 + (1 - np.abs(alpha)) * signal2
        norm_combined_sig = combined_signal / np.std(combined_signal)
        norm_sig1 = signal1/np.std(signal1)
        diff_ca = norm_combined_sig - norm_sig1
        # sub_asdf = norm_combined_sig - signal1/np.std(signal1)
        # sub_asdf = (combined_signal - signal1) / (np.std(combined_signal) * np.std(signal1))

        corr = np.corrcoef(signal1, combined_signal)[1][0]
        # corr = np.corrcoef(signal1, norm_combined_sig)[1][0]
        var_c_list.append(np.var(combined_signal))
        new_calc.append((1 - np.var(diff_ca)/2)**2)
        # new_calc.append((1 - np.var(combined_signal/np.std(combined_signal) - signal1/np.std(signal1))/2)**2)
        # print(f'var(c) = {np.var(combined_signal)}, corr = {corr}')
        var_c = np.var(combined_signal)
        var_sub_asdf_list.append(1-np.var(diff_ca)/2)

        # hier alpha rausbekommen

        # print(f'alpha={alpha}, asdf = {1-np.var(sub_asdf)/2}, vratio = {var_ratio}, vc = {var_c}, va = {va}, vb = {vb}')
        print(f'alpha={alpha}, asdf = {np.std(diff_ca)}, vratio = {var_ratio}, sratio = {std_ratio}, vc = {var_c}, va = {va*alpha}, vb = {vb*(1-alpha)}')
        correlations.append(corr)

    correlations = np.array(correlations)
    plt.figure(figsize=(5, 5))
    var_c_list = np.array(var_c_list)
    var_sub_asdf_list = np.array(var_sub_asdf_list)

    if y_alpha:
        plt.plot(correlations ** 2, alpha_list, label='measured')
        plt.plot(to_alpha(correlations ** 2, var_ratio=var_ratio), alpha_list, label='lin?')
        plt.plot(correlations ** 2, to_alpha(correlations ** 2, var_ratio=var_ratio), ':', color='k',
                 label='calculated')
        # plt.plot(alpha_list, var_sub_asdf_list*var_c_list, label='subasdf')
        plt.xlabel('squared correlation')
        plt.ylabel(r'$\alpha$')
    else:
        plt.plot(alpha_list, correlations**2, label='squared corr coeff')
        plt.plot(alpha_list, to_alpha(correlations ** 2, var_ratio=var_ratio), label='lin?')
        plt.plot(alpha_list, correlations, label='corr coeff')
        plt.plot(alpha_list, to_squared_corr(alpha_list, var_ratio=var_ratio), ':', color='k', label='calculated')
        plt.plot(alpha_list, np.sqrt(to_squared_corr(alpha_list, var_ratio=var_ratio)), ':', color='k')
        # plt.plot(alpha_list, var_c_list, label='var comb')
        # plt.plot(alpha_list, (var_sub_asdf_list)**2, label='subasdf')
        # plt.plot(alpha_list, new_calc, label='subasdf')
        plt.xlabel(r'$\alpha$')
        plt.ylabel('correlation')

    plt.title(fr'$\sigma^2(sig1) = {round(np.var(signal1), 2)}$, $\sigma^2(sig2) = {round(np.var(signal2), 2)}$')
    print(np.polyfit(alpha_list, var_c_list, deg=2))
    print(f'{np.var(signal1) + np.var(signal2)}, {-2*np.var(signal2)}, {np.var(signal2)}')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('/home/schultetobrinke/Downloads/deleteme/alpha_correlations_different_stds.pdf')
    plt.show()


if __name__ == "__main__":
    main()
