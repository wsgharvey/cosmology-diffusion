from collections import defaultdict
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import pickle
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

# Metrics
from improved_diffusion import test_util


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

    files = sorted(glob.glob(str(Path(args.eval_dir) / "samples" / "*.npy")))[:args.n_samples]
    samples = np.stack([np.load(f) for f in files])
    print(samples.shape, samples.min(), samples.max())

    densities = np.exp(1 + samples) - 1  # densities in range [0, inf]

    fig, axes = plt.subplots(ncols=2)
    for ax in axes:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_xscale('log')
        ax.set_ylabel(r'$\Delta^2(k)$')
    axes[0].set_yscale('log')
    axes[0].set_title("Power spectrum")
    axes[1].set_title("Cross correlation")
    for s, density in enumerate(densities):
        # compute autocorrelation
        k, power, _, _ = compute_cross_spectrum(density, density, L=256.)
        axes[0].plot(k, power, alpha=0.5)
        # compute cross correlation
        k2, power2, _, _ = compute_cross_spectrum(density, densities[s-1], L=256.)
        axes[1].plot(k2, power2, alpha=0.5)
    fig.savefig(Path(args.eval_dir) / f"power-spectra-{args.n_samples}.pdf", bbox_inches='tight')

    mean_densities = np.stack([density.mean() for density in densities])
    dm = mean_densities.mean()
    ds = mean_densities.std()

    # plot histogram of np.log(1+density) = np.log(2+overdensity)
    fig, ax = plt.subplots()
    ax.hist(1+samples.reshape(-1), bins=100, density=True)
    ax.set_xlabel('log(1+density)')
    ax.set_ylabel('pdf')
    ax.set_yscale('log')
    fig.savefig(Path(args.eval_dir) / f"density-histogram-{args.n_samples}.pdf", bbox_inches='tight')

    # # Save all metrics as a pickle file (update it if it already exists)
    # with test_util.Protect(pickle_path): # avoids race conditions
    #     if pickle_path.exists():
    #         metrics_pkl = pickle.load(open(pickle_path, "rb"))
    #     else:
    #         metrics_pkl = {}
    #     for mode in args.modes:
    #         metrics_pkl[mode] = new_metrics[mode]
    #     pickle.dump(metrics_pkl, open(pickle_path, "wb"))
    # print(f"Saved metrics to {pickle_path}.")
