from collections import defaultdict
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
import glob
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

# Metrics
from improved_diffusion.image_datasets import load_data


def compute_cross_spectrum(d1, d2, L, kmin=None, kmax=None, nk=64, dimensionless=True, eps_slop=1e-3):
    '''
    Compute the cross spectrum between two real-space 2D density slices: d1 and d2
    If d1 = d2 then this will be the (auto) power spectrum, otherwise cross spectrum
    d1, d1: real-space density fields
    L: Box size [usually Mpc/h units in cosmology]
    kmin, kmax: Minimum/maximum wavenumbers
    nk: Number of k bins
    dimesionless: Compute Delta^2(k) rather than P(k)
    eps_slop: Ensure that all modes get counted (maybe unnecessary)
    TODO: Correct for binning by sharpening
    '''

    def _k_FFT(ix, iy, m, L):
        '''
        Get the wavenumber associated with the element in the FFT array
        ix, iy: indices in 2D array
        m: mesh size for FFT
        L: Box size [units]
        '''
        kx = ix if ix<=m/2 else m-ix
        ky = iy if iy<=m/2 else m-iy
        kx *= 2.*np.pi/L
        ky *= 2.*np.pi/L
        return kx, ky, np.sqrt(kx**2+ky**2)

    # Calculations
    m = d1.shape[0] # Mesh size for array
    if kmin is None: kmin = 2.*np.pi/L # Box frequency
    if kmax is None: kmax = m*np.pi/L  # Nyquist frequency

    # Bins for wavenumbers k
    kbin = np.logspace(np.log10(kmin), np.log10(kmax), nk+1) # Note the +1 to have all bins
    kbin[0] *= (1.-eps_slop); kbin[-1] *= (1.+eps_slop) # Avoid slop

    # Fourier transforms (renormalise for mesh size)
    # TODO: Use FFT that knows input fields are real
    dk1 = np.fft.fft2(d1)/m**2
    dk2 = np.fft.fft2(d2)/m**2 if d1 is not d2 else dk1 # Avoid a second FFT if possible

    # Loop over Fourier arrays and accumulate power
    # TODO: These loops could be slow (rate limiting) in Python
    k = np.zeros(nk)
    power = np.zeros(nk)
    sigma = np.zeros(nk)
    nmodes = np.zeros(nk, dtype=int)
    for ix in range(m):
        for iy in range(m):
            _, _, kmod = _k_FFT(ix, iy, m, L)
            for i in range(nk):
                if kbin[i] <= kmod < kbin[i+1]:
                    k[i] += kmod
                    f = np.real(np.conj(dk1[ix, iy])*dk2[ix, iy])
                    power[i] += f
                    sigma[i] += f**2
                    nmodes[i] += 1
                    break

    # Averages over number of modes
    for i in range(nk):
        if nmodes[i]==0:
            k[i] = np.sqrt(kbin[i+1]*kbin[i])
            power[i] = 0.
            sigma[i] = 0.
        else:
            k[i] /= nmodes[i]
            power[i] /= nmodes[i]
            sigma[i] /= nmodes[i]
            if nmodes[i]==1:
                sigma[i] = 0.
            else:
                sigma[i] = np.sqrt(sigma[i]-power[i]**2)
                sigma[i] = sigma[i]*nmodes[i]/(nmodes[i]-1)
                sigma[i] = sigma[i]/np.sqrt(nmodes[i])

    # Create dimensionless spectra if desired
    if dimensionless:
        Dk = 2.*np.pi*(k*L/(2.*np.pi))**2
        power = power*Dk
        sigma = sigma*Dk

    return k, power, sigma, nmodes




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to evaluate.")
    args = parser.parse_args()

    # load model samples
    files = sorted(glob.glob(str(Path(args.eval_dir) / "samples" / "*.npy")))[:args.n_samples]
    samples = np.stack([np.load(f) for f in files])
    print(samples.shape, samples.min(), samples.max())
    # load dataset samples (getting arguments from the saved config)
    model_config_path = Path(args.eval_dir) / "model_config.json"
    assert model_config_path.exists(), f"Could not find model config at {model_config_path}"
    with open(model_config_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))
    data = next(load_data(
        data_path=model_args.data_path, batch_size=args.n_samples,
        image_channels=1, max_data_value=model_args.max_data_value
    ))[0].squeeze(dim=1).numpy()

    densities = np.exp(1 + samples) - 1  # densities in range [0, inf]
    data_densities = np.exp(1 + data) - 1

    fig, axes = plt.subplots(ncols=2)
    for ax in axes:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_xscale('log')
        ax.set_ylabel(r'$\Delta^2(k)$')
    axes[0].set_yscale('log')
    axes[0].set_title("Power spectrum")
    axes[1].set_title("Cross correlation")
    def get_power_spectra(densities):
        power = []
        cross = []
        for s, density in enumerate(densities):
            k, p, _, _ = compute_cross_spectrum(density, density, L=256.)
            power.append(p)
            _, c, _, _ = compute_cross_spectrum(density, densities[s-1], L=256.)
            cross.append(c)
        return k, np.array(power), np.array(cross)
    k, power, cross = get_power_spectra(densities)
    k, data_power, data_cross = get_power_spectra(data_densities)
    col_ddpm, col_data = 'b', 'r'
    axes[0].plot(k, power.mean(axis=0), label="DDPM", color=col_ddpm)
    axes[0].fill_between(k, power.mean(axis=0)-power.std(axis=0), power.mean(axis=0)+power.std(axis=0), alpha=0.2, color=col_ddpm)
    axes[1].plot(k, cross.mean(axis=0), label="DDPM", color=col_ddpm)
    axes[1].fill_between(k, cross.mean(axis=0)-cross.std(axis=0), cross.mean(axis=0)+cross.std(axis=0), alpha=0.2, color=col_ddpm)
    axes[0].plot(k, data_power.mean(axis=0), label="Data", color=col_data)
    axes[0].fill_between(k, data_power.mean(axis=0)-data_power.std(axis=0), data_power.mean(axis=0)+data_power.std(axis=0), alpha=0.2, color=col_data)
    axes[1].plot(k, data_cross.mean(axis=0), label="Data", color=col_data)
    axes[1].fill_between(k, data_cross.mean(axis=0)-data_cross.std(axis=0), data_cross.mean(axis=0)+data_cross.std(axis=0), alpha=0.2, color=col_data)
    axes[1].legend()
    fig.savefig(Path(args.eval_dir) / f"power-spectra-{args.n_samples}.pdf", bbox_inches='tight')

    # plot histogram of np.log(1+density) = np.log(2+overdensity)
    fig, ax = plt.subplots()
    ax.hist(1+samples.reshape(-1), bins=100, density=True, range=[-1, model_args.max_data_value+1], label="DDPM", alpha=0.5, color=col_ddpm)
    ax.hist(1+data.reshape(-1), bins=100, density=True, range=[-1, model_args.max_data_value+1], label="Data", alpha=0.5, color=col_data)
    ax.set_xlabel('log(1+density)')
    ax.set_ylabel('pdf')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(Path(args.eval_dir) / f"density-histogram-{args.n_samples}.pdf", bbox_inches='tight')
