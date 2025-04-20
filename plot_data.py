# kingston
# ❯ python plot_data.py m3data/250405_ibm_kingston_custom14_depth2_folds4_qor0_rd250.pkl m3data/250408_ibm_kingston_path20_depth2_folds4_qor0_rd250.pkl m3data/250403_ibm_kingston_custom26_depth2_folds4_qor0_rd250.pkl m3data/250409_ibm_kingston_custom32_depth2_folds4_qor0_rd250.pkl m3data/250408_ibm_kingston_custom38c_depth2_folds4_qor0_rd250.pkl m3data/250407_ibm_kingston_custom44c_depth2_folds4_qor0_rd250.pkl m3data/250405_ibm_kingston_custom50c_depth2_folds4_qor0_rd250.pkl

# ❯ python plot_data.py data/250405_ibm_kingston_custom14_depth2_folds4_qor0_rd250.pkl data/250408_ibm_kingston_path20_depth2_folds4_qor0_rd250.pkl data/250403_ibm_kingston_custom26_depth2_folds4_qor0_rd250.pkl data/250409_ibm_kingston_custom32_depth2_folds4_qor0_rd250.pkl data/250408_ibm_kingston_custom38c_depth2_folds4_qor0_rd250.pkl data/250407_ibm_kingston_custom44c_depth2_folds4_qor0_rd250.pkl data/250405_ibm_kingston_custom50c_depth2_folds4_qor0_rd250.pkl


# fez
# ❯ python plot_data.py m3data/250309_ibm_fez_custom14_depth2_folds4_qor0_rd500.pkl m3data/250401_ibm_fez_path20_depth2_folds4_qor0_rd250.pkl m3data/241205_ibm_fez_custom26_depth2_folds4_qor0_rd500.pkl m3data/250302_ibm_fez_custom32_depth2_folds4_qor0_rd250.pkl m3data/250304_ibm_fez_custom38c_depth2_folds4_qor0_rd250.pkl m3data/250207_ibm_fez_custom44c_depth2_folds4_qor0_rd500.pkl m3data/250405_ibm_fez_custom50c_depth2_folds4_qor0_rd250.pkl

# ❯ python plot_data.py data/250309_ibm_fez_custom14_depth2_folds4_qor0_rd500.pkl data/250401_ibm_fez_path20_depth2_folds4_qor0_rd250.pkl data/241205_ibm_fez_custom26_depth2_folds4_qor0_rd500.pkl data/250302_ibm_fez_custom32_depth2_folds4_qor0_rd250.pkl data/250304_ibm_fez_custom38c_depth2_folds4_qor0_rd250.pkl data/250207_ibm_fez_custom44c_depth2_folds4_qor0_rd500.pkl data/250405_ibm_fez_custom50c_depth2_folds4_qor0_rd250.pkl


# sims
# python plot_data.py simdata/*5*custom14* simdata/*5*path20* simdata/*5*custom26*2.pkl simdata/*5*custom32_* simdata/*5*38c* simdata/*5*44c* simdata/*5*50c*depth2*
# python plot_data.py simdata/*3*custom14* simdata/*3*path20* simdata/*3*custom26*2.pkl simdata/*3*custom32_* simdata/*3*38c* simdata/*3*44c* simdata/*3*50c*depth2*
import os
import argparse
import pickle as pkl
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def sigma_squared(ler, nshots):
    mu_exp = ler
    valued_m1 = ler * nshots
    valued_1 = nshots - valued_m1
    return (valued_1 * (mu_exp - 1) ** 2 + valued_m1 * (mu_exp + 1) ** 2) / nshots**2


def mu_exp(logical_error_rates):
    return np.mean([ler for ler in logical_error_rates])


def sigma_exp(logical_error_rates):
    return np.std(logical_error_rates)


def sem(logical_error_rates, shot_counts):
    M = len(logical_error_rates)
    sigmas_squared = [
        sigma_squared(ler, n) for ler, n in zip(logical_error_rates, shot_counts)
    ]
    variance = sum(sigmas_squared) / M
    variance += sigma_exp(logical_error_rates) ** 2
    variance /= M
    return np.sqrt(variance)


def compute_confidence_interval(shot_counts, logical_error_rates, alpha):
    standard_error = sem(logical_error_rates, shot_counts)
    z_star = norm.ppf(1 - alpha / 2)
    return z_star * standard_error



def sigma_squared_psrate(psr, nshots):
    valued_1 = psr * nshots
    valued_0 = nshots - valued_1
    mu_exp = psr
    return (valued_1 * (mu_exp - 1) ** 2 + valued_0 * mu_exp ** 2) / nshots**2


def mu_exp_psrate(ps_rates):
    return np.mean(ps_rates)


def sigma_exp_psrate(ps_rates):
    return np.std(ps_rates)


def sem_psrate(ps_rates, total_shot_count):
    M = len(ps_rates)
    sigmas_squared = [
        sigma_squared_psrate(psr, total_shot_count) for psr in ps_rates 
    ]
    variance = sum(sigmas_squared) / M
    variance += sigma_exp_psrate(ps_rates) ** 2
    variance /= M
    return np.sqrt(variance)


def compute_confidence_interval_psrate(accepted_shots, num_shots, alpha):
    ps_rates = [acc/num_shots for acc in accepted_shots]
    standard_error = sem_psrate(ps_rates, num_shots)
    z_star = norm.ppf(1 - alpha / 2)
    return z_star * standard_error

# color pallette


# darker
ibmgreen = '#24a148'
ibmteal = '#08bdba'
ibmcyan = '#33b1ff'
ibmblue = '#0f62fe'
ibmpurple = '#8a3ffc' 
ibmmagenta = '#ee5396'
ibmred = '#a2191f'
max_rounds = 14
ps_threshold = 1e-5

def plot_data(datas, datas_noavg=None, ler_floors_list=None, num_shots_list=None, confidence=None, title=None, legend=None, save=False):
    max_checks = 0
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    colors = [ibmgreen, ibmteal, ibmcyan, ibmblue, ibmpurple, ibmmagenta, ibmred]
    for d, data in enumerate(datas):
        reps = len(data)
        # cut off data if ps rate goes below a threshold
        cutoff = max_rounds
        for r, (ps, _, _, _) in enumerate(data[0]):
            if ps < ps_threshold:
                cutoff = r
                break
        rounds = min(len(data[0]), max_rounds, cutoff)
        max_checks = max(max_checks, rounds-1)
        flat_data = [data[rep][round] for round in range(rounds) for rep in range(reps)]
        ps_data = [ps for ps, _, _, _ in flat_data]
        ler_data = [1 - 2 * ler for _, ler, _, _ in flat_data]
        count_data = [count for _, _, count, _ in flat_data]
        coverage_data = [1 - coverage for _, _, _, coverage in flat_data]

        ps_reshaped = np.array(ps_data).reshape(rounds, reps)
        ler_reshaped = np.array(ler_data).reshape(rounds, reps)
        count_reshaped = np.array(count_data).reshape(rounds, reps)
        coverage_reshaped = np.array(coverage_data).reshape(rounds, reps)

        ps_averages = np.mean(ps_reshaped, axis=1)
        ler_averages = np.mean(ler_reshaped, axis=1)
        count_averages = np.mean(count_reshaped, axis=1)
        coverage_averages = np.mean(coverage_reshaped, axis=1)

        if datas_noavg[d]:
            flat_data_noavg = [datas_noavg[d][rep][round] for round in range(rounds) for rep in range(reps)]
            accepted_shots_data = [acc for acc, _ in flat_data_noavg]
            lers_data = [[1 - 2 * ler for ler in lers] for _, lers in flat_data_noavg]
            confidence_ler_data, confidence_ps_data = [], []
            for accepted_shots, lers in zip(accepted_shots_data, lers_data):
                confidence_ler = compute_confidence_interval(accepted_shots, lers, alpha=confidence)
                confidence_ps = compute_confidence_interval_psrate(accepted_shots, num_shots_list[d], alpha=confidence)
                confidence_ler_data.append(confidence_ler)
                confidence_ps_data.append(confidence_ps)
            confidence_ler_reshaped = np.array(confidence_ler_data).reshape(rounds, reps)
            confidence_ps_reshaped = np.array(confidence_ps_data).reshape(rounds, reps)

        linestyle = '--' if 'sim' in legend[d] or 'right' in legend[d] else None
        markersize = 3 if 'sim' in legend[d] else None
        for i in range(reps):
            ax1.plot(range(rounds), ps_reshaped[:, i], color=colors[d], alpha=0.5, linestyle=linestyle, markersize=markersize)
            if datas_noavg[d]:
                ax1.fill_between(range(rounds), ps_reshaped[:, i] - confidence_ps_reshaped[:, i], ps_reshaped[:, i] + confidence_ps_reshaped[:, i], alpha=0.2, color=colors[d])
        ax1.plot(range(rounds), ps_averages, marker='o', color=colors[d], label=legend[d], linestyle=linestyle, markersize=markersize)
        ax1.set_xlabel('pairs of checks', fontsize=14)
        ax1.set_yscale('log')
        ax1.set_ylabel('Postselection Rate', fontsize=14)
        ax1.set_ylim(ps_threshold, 1.)
        ax1.set_xlim(0, max_checks)
        ax1.grid(True)
        for i in range(reps):
            ax2.plot(range(rounds), ler_reshaped[:, i], color=colors[d], alpha=0.5, linestyle=linestyle, markersize=markersize)
            if datas_noavg[d]:
                ax2.fill_between(range(rounds), ler_reshaped[:, i] - confidence_ler_reshaped[:, i], ler_reshaped[:, i] + confidence_ler_reshaped[:, i], alpha=0.2, color=colors[d])
                print(f'{ler_averages[0]} +/- {confidence_ler_reshaped[0]} -> {ler_averages[-1]} +/- {confidence_ler_reshaped[-1]}  ({ler_averages[-1] / ler_averages[0]} gain) [{count_averages[0]} -> {count_averages[-1]}]')
        ax2.plot(range(rounds), ler_averages, marker='o', color=colors[d], label=legend[d], linestyle=linestyle, markersize=markersize)
        ax2.set_xlabel('pairs of checks', fontsize=14)
        ax2.set_xlim(0, max_checks)
        ax2.set_ylabel('Estimated Fidelity', fontsize=14)
        ax2.set_ylim(0, 1)
        ler_floors = ler_floors_list[d]
        if ler_floors:
            for i in range(reps):
                ax2.axhline(ler_floors[i], color=colors[d], alpha=0.2)
            ax2.axhline(np.mean(ler_floors), linestyle='--', color=colors[d])
        ax2.grid(True)
        ax3_data = count_reshaped
        for i in range(reps):
            ax3.plot(range(rounds), ax3_data[:, i], color=colors[d], alpha=0.5, linestyle=linestyle, markersize=markersize)
        ax3.plot(range(rounds), count_averages, marker='o', color=colors[d], label=legend[d], linestyle=linestyle, markersize=markersize)
        ax3.set_ylim(0, 2800)
        ax3.set_xlim(0, max_checks)
        ax3.set_ylabel('Physical CZ Count', fontsize=14)
        ax3.set_xlabel('pairs of checks', fontsize=14)
        ax3.grid(True)
        ax1.legend(loc='lower left')
    plt.tight_layout()    
    if save:
        plt.savefig(f'./plots/data_{title}.pdf', bbox_inches='tight')
    else:
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+', type=str)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

rep_data_list = []
rep_data_noavg_list = []
ler_floors_list = []
num_shots_list = []
base_names = []
for filename in args.filenames:
    with open(filename, 'rb') as f:
        data = pkl.load(f)
        rep_data_list.append(data['rep_data'])
        if data.get('rep_data_noavg'):
            rep_data_noavg_list.append(data['rep_data_noavg'])
            num_shots_list.append(data['num_shots'])
        else:
            rep_data_noavg_list.append([])
            num_shots_list.append(1000)
        ler_floors_list.append(None)
        base_names.append(os.path.splitext(os.path.basename(filename))[0])

title = 'kingston_final'
base_names = ['14 logical qubits', '20 logical qubits', '26 logical qubits', '32 logical qubits', '38 logical qubits', '44 logical qubits', '50 logical qubits']

plot_data(rep_data_list, rep_data_noavg_list, ler_floors_list, num_shots_list, confidence=.32, title=title, legend=base_names, save=args.save)
