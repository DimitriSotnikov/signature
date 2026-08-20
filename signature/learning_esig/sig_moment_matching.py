import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Tuple, List
from scipy.optimize import minimize
from scipy.special import gamma
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from math import factorial

from simulation.diffusion import Diffusion
import matplotlib.pyplot as plt

from ..shuffle_product import shuffle_prod
from ..shuffle_table import get_shuffle_table
from ..tensor_sequence import TensorSequence
from ..expected_signature import expected_bm_fm_signature, expected_bm_signature
from ..path_signature import path_to_fm_signature, path_to_signature
from ..signature_of_signature import get_signature_of_linear_form
from ..words import number_of_words_up_to_trunc, index_to_word_vect, index_to_word
from ..factory import from_array, unit


class SigSignal:
    trunc: int
    t_grid: NDArray[float64]
    trunc_signature_moments: int
    eSig: None # by default extended Brownian motion
    optimizer: str

    shuffle_table = None
    l: TensorSequence = None
    words: List
    eSig: TensorSequence
    SigS: TensorSequence

    def __init__(
        self,
        trunc,
        T: float,
        trunc_signature_moments: int = 5,
        eSig: TensorSequence = None,
        optimizer="Powell",
    ):
        self.trunc = trunc
        self.trunc_signature_moments = trunc_signature_moments
        self.optimizer = optimizer

        # self.ta = TensorAlgebra(dim=2, trunc=self.trunc * self.trunc_signature_moments)
        self.shuffle_table = get_shuffle_table(table_trunc=self.trunc * self.trunc_signature_moments, dim=2)

        if eSig is None:
            self.eSig = expected_bm_signature(t=T, trunc=self.trunc * self.trunc_signature_moments)
        else:
            self.eSig = eSig

        n_moments = number_of_words_up_to_trunc(self.trunc_signature_moments, 2)
        words = index_to_word_vect(np.arange(n_moments), 2)
        self.words = words

    def x_to_ts(self, x):
        array = np.zeros(number_of_words_up_to_trunc(self.trunc, 2))
        array[1:] = x
        return from_array(trunc=self.trunc * self.trunc_signature_moments, array=array, dim=2)

    def fit(self, expected_signal_sig: TensorSequence):
        x0 = np.zeros(number_of_words_up_to_trunc(self.trunc, 2) - 1)

        def callback(x, f=None, context=None, accept=None, convergence=None):
            val = self.loss(x, expected_signal_sig)
            print(f"New iteration: \n x = {x}, \n val={val}. \n")

        res = minimize(lambda x: self.loss(x, expected_signal_sig), x0, method=self.optimizer, callback=callback)
        self.l = self.x_to_ts(res.x)

    def loss(self, x, expected_signal_sig):
        l = self.x_to_ts(x)
        signal_sig_coefs = get_signature_of_linear_form(ts=l, trunc_moments=self.trunc_signature_moments, shuffle_table=self.shuffle_table)

        loss = 0
        for word in self.words:
            if "2" in word:
                loss_word = np.sqrt(
                    np.mean(np.abs((signal_sig_coefs[word] @ self.eSig).real.squeeze() - expected_signal_sig[word].real.squeeze()) ** 2))
                loss += loss_word * factorial(len(word))

        return loss


class StatSigSignal:
    trunc: int
    lam: float
    t_grid: NDArray[float64]
    window_size: int
    max_stationary_moment: int
    trunc_signature_moments: int
    loss_weights: Tuple
    optimizer: str
    rng: np.random.Generator

    shuffle_table = None
    l: TensorSequence = None
    eSSig: TensorSequence
    SigS: TensorSequence

    def __init__(
        self,
        trunc,
        lam,
        t_grid,
        window_size=100,
        max_stationary_moment=7,
        trunc_signature_moments=5,
        optimizer="BFGS",
        loss_weights=(1, 5e-4),
        rng=None
    ):
        self.trunc = trunc
        self.lam = lam
        self.t_grid = t_grid
        self.window_size = window_size
        self.max_stationary_moment = max_stationary_moment
        self.trunc_signature_moments = trunc_signature_moments
        self.loss_weights = loss_weights
        self.optimizer = optimizer

        # self.ta = TensorAlgebra(dim=2, trunc=self.trunc * 2)
        self.shuffle_table = get_shuffle_table(self.trunc * 2, dim=2)

        EPS = 1e-8
        n_t_past = 1000
        t_past = np.linspace(np.log(2 * np.min(lam) * EPS**2) / 2 / np.min(lam), 0, n_t_past)
        t_grid_extended = np.concatenate([t_past[:-1] + t_grid[0], t_grid])

        if rng is None:
            self.rng = np.random.default_rng(seed=42)
        else:
            self.rng = rng

        diffusion = Diffusion(t_grid=t_grid_extended - t_grid_extended[0], size=1, rng=self.rng)

        brownian_motion = diffusion.brownian_motion()[0, 0, :]
        path = np.vstack([t_grid_extended, brownian_motion]).T
        self.SigS = path_to_fm_signature(path=path, trunc=self.trunc, t_grid=t_grid_extended - t_grid[0], lam=self.lam * np.ones(2))
        self.eSSig = expected_bm_fm_signature(trunc=self.trunc * self.trunc_signature_moments, lam=lam)

    def fit(self, signal: NDArray[float64]):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        normalized_signal = (signal - signal_mean) / signal_std

        signal_moments = (normalized_signal[:, None] ** np.arange(3, 7)).mean(axis=0)

        num_of_win = len(self.t_grid) - self.window_size
        path_signal = np.empty((self.window_size, 2, num_of_win))
        path_signal[:, 0, :] = np.tile(self.t_grid[:self.window_size][:, None], num_of_win)
        for idx_start in range(num_of_win):
            path_signal[:, 1, idx_start] = normalized_signal[idx_start:idx_start + self.window_size]

        SignalSig = path_to_signature(path=path_signal, trunc=self.trunc_signature_moments)
        e_signal_sig = SignalSig.array[:, -1, :].mean(axis=1).real

        rng = np.random.default_rng(seed=42)
        if self.trunc == 3:
            x_size = 8
        elif self.trunc == 2:
            x_size = 4
        else:
            raise NotImplementedError("Implement properly the number of linearly independent elements!")
        x0 = rng.normal(size=x_size) * 0.01

        def callback(x, f=None, context=None, accept=None, convergence=None):
            val = self.loss(x, e_signal_sig=e_signal_sig, signal_moments=signal_moments, verbose=True)
            print(f"New iteration: \n x = {x}, \n val={val}. \n")

        def calib_loss(x):
            return self.loss(x, e_signal_sig=e_signal_sig, signal_moments=signal_moments)

        res = minimize(calib_loss, x0, callback=callback, method=self.optimizer)

        print(res)

        self.l = self.normalize_ts(ts=self.ts_from_x(res.x), loc=signal_mean, scale=signal_std)

    def ts_from_x(self, x):
        mask = np.array([0, 2, 4, 6, 8, 10, 12, 14])
        number_of_elements = number_of_words_up_to_trunc(self.trunc, 2)
        array = np.zeros(number_of_elements)
        array[mask[mask < number_of_elements]] = x
        return from_array(trunc=self.trunc * self.trunc_signature_moments, array=array, dim=2)

    def normalize_ts(self, ts: TensorSequence, loc: float = 0, scale: float = 1):
        l_mean = (self.eSSig @ ts).squeeze().real
        l_second_moment = (self.eSSig @ shuffle_prod(ts, ts, self.shuffle_table)).squeeze().real
        l_std = np.sqrt(l_second_moment - l_mean ** 2)

        new_array = ts.array
        new_array.at[0].add(-l_mean)
        new_array = new_array / l_std * scale
        new_array.at[0].add(loc)
        return from_array(trunc=self.trunc, array=new_array, dim=2)

    def empirical_expected_signal_sig(self, signal: NDArray[float64]):
        num_of_win = len(self.t_grid) - self.window_size
        path_signal = np.empty((self.window_size, 2, num_of_win))
        path_signal[:, 0, :] = np.tile(self.t_grid[:self.window_size][:, None], num_of_win)
        for idx_start in range(num_of_win):
            path_signal[:, 1, idx_start] = signal[idx_start:idx_start + self.window_size]

        SignalSig = path_to_signature(path=path_signal, trunc=self.trunc_signature_moments)

        # possibly add a tensor normalization here

        return SignalSig.array[:, -1, :].mean(axis=1).squeeze().real

    def loss(self, x, e_signal_sig, signal_moments, verbose=False):
        l = self.ts_from_x(x)
        l = self.normalize_ts(l)

        signal_l = (l @ self.SigS).squeeze().real
        e_signal_sig_l = self.empirical_expected_signal_sig(signal_l)

        signal_l_moments = (signal_l[:, None] ** np.arange(3, self.max_stationary_moment)).mean(axis=0)

        weight_level = gamma(unit(trunc=self.trunc_signature_moments, dim=2).get_lengths_array() + 1)
        loss_esig = np.sqrt(np.mean(((e_signal_sig - e_signal_sig_l) * weight_level) ** 2)) * self.loss_weights[0]
        # print("loss:", ((e_signal_sig - e_signal_sig_l)) ** 2)
        # print("loss (weighted):", ((e_signal_sig - e_signal_sig_l) * weight_level) ** 2)
        loss_higher_moments = np.sqrt(np.mean((signal_l_moments - signal_moments) ** 2)) * self.loss_weights[1]

        if verbose:
            print(f"Esig: {loss_esig}, Stationary moments: {loss_higher_moments}")

        return loss_esig + loss_higher_moments

    def analysis(self, signal, window_size=1000):
        if self.l is not None:
            l_SigS = (self.l @ self.SigS).squeeze().real

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        ax[0][0].plot(self.t_grid[:window_size], signal[:window_size], label="Signal")
        if self.l is not None:
            ax[0][0].plot(self.t_grid[:window_size], l_SigS[:window_size], "--", label="<l*, W_t>")
        ax[0][0].set_title("Sample paths")
        ax[0][0].legend()

        plot_acf(signal, ax=ax[0][1], label="Signal")
        if self.l is not None:
            plot_acf(l_SigS, ax=ax[0][1], label="<l*, W_t>")
        ax[0][1].legend()
        ax[0][1].set_ylim([-1.05, 1.05])

        ax[1][0].hist(signal, bins=100, alpha=0.5, density=True, label="Signal")
        if self.l is not None:
            ax[1][0].hist(l_SigS, bins=100, alpha=0.5, density=True, label="<l*, W_t>")
        ax[1][0].set_title("Stationary distribution histogram")
        ax[1][0].legend()

        sm.qqplot(signal, ax=ax[1][1], label="signal")
        if self.l is not None:
            color_2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
            sm.qqplot(l_SigS, ax=ax[1][1], marker="v", markerfacecolor=color_2, markeredgecolor=color_2, label="fit")
        ax[1][1].set_title("Q-Q Plot")

        plt.show()
