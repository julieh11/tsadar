from jax import numpy as jnp


def add_ATS_IRF(config, sas, lamAxisE, modlE, amps, TSins, lam):
    stddev_lam = config["other"]["PhysParams"]["widIRF"]["spect_FWHM_ele"] / 2.3548
    stddev_ang = config["other"]["PhysParams"]["widIRF"]["ang_FWHM_ele"] / 2.3548
    # Conceptual_origin so the convolution donsn't shift the signal
    origin_lam = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    origin_ang = (jnp.amax(sas["angAxis"]) + jnp.amin(sas["angAxis"])) / 2.0
    inst_func_lam = jnp.squeeze(
        (1.0 / (stddev_lam * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((lamAxisE - origin_lam) ** 2.0) / (2.0 * (stddev_lam) ** 2.0))
    )  # Gaussian
    inst_func_ang = jnp.squeeze(
        (1.0 / (stddev_ang * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((sas["angAxis"] - origin_ang) ** 2.0) / (2.0 * (stddev_ang) ** 2.0))
    )  # Gaussian
    ThryE = jnp.array([jnp.convolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    # ThryE = jnp.array([fftconvolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    ThryE = jnp.array([jnp.convolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])
    # ThryE = jnp.array([fftconvolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])

    ThryE = jnp.amax(modlE, axis=1, keepdims=True) / jnp.amax(ThryE, axis=1, keepdims=True) * ThryE

    if config["other"]["PhysParams"]["norm"] > 0:
        ThryE = jnp.where(
            lamAxisE < lam,
            TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
            TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
        )
    return lamAxisE, ThryE


def add_ion_IRF(config, lamAxisI, modlI, amps, TSins):
    stddevI = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ion"]
    originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
    inst_funcI = jnp.squeeze(
        (1.0 / (stddevI * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
    )  # Gaussian
    ThryI = jnp.convolve(modlI, inst_funcI, "same")
    ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
    ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

    if config["other"]["PhysParams"]["norm"] == 0:
        lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
        ThryI = TSins["amp3"]["val"] * amps * ThryI / jnp.amax(ThryI)
        #lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

    return lamAxisI, ThryI


def add_electron_IRF(config, lamAxisE, modlE, amps, TSins, lam):
    stddevE = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
    # Conceptual_origin so the convolution doesn't shift the signal
    originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    inst_funcE = jnp.squeeze(
        (1.0 / (stddevE * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
    )  # Gaussian
    ThryE = jnp.convolve(modlE, inst_funcE, "same")
    ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

    if config["other"]["PhysParams"]["norm"] > 0:
        ThryE = jnp.where(
            lamAxisE < lam,
            TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
            TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
        )

    ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
    if config["other"]["PhysParams"]["norm"] == 0:
        lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
        ThryE = amps * ThryE / jnp.amax(ThryE)
        ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)

    return lamAxisE, ThryE