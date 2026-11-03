def _check_config(config):

    # Telescope
    if "diameter" not in config["telescope"]:
        raise ValueError("diameter is mandatory")
    
    if "central_obscuration" not in config["telescope"]:
        config["telescope"]["central_obscuration"] = 0.0

    if "spider" not in config["telescope"]:
        config["telescope"]["spider"] = 0

    # Images
    if "n_pixel" not in config["images"]:
        raise ValueError("n_pixel is mandatory")
    
    if "pix_size" not in config["images"]:
        raise ValueError("pix_size is mandatory")
    
    if "apodization_border" not in config["images"]:
        config["images"]["apodization_border"] = 0
    
    if "remove_gradient_apodization" not in config["images"]:
        config["images"]["remove_gradient_apodization"] = False

    # Optimization
    if "gpu" not in config["optimization"]:
        config["optimization"]["gpu"] = -1

    if "transform" not in config["optimization"]:
        config["optimization"]["transform"] = "none"

    if "softplus_scale" not in config["optimization"]:
        config["optimization"]["softplus_scale"] = 1.0
    
    if "lr_obj" not in config["optimization"]:
        config["optimization"]["lr_obj"] = 0.01

    if "lr_modes" not in config["optimization"]:
        config["optimization"]["lr_modes"] = 0.01
    
    if "lr_prior" not in config["optimization"]:
        config["optimization"]["lr_prior"] = 0.02

    if "jitter" not in config["psf"]:
        config["psf"]["jitter"] = 'none'

    # YAML `jitter: False` / `jitter: ~` mean "no jitter"; anything else must be a known mode
    if config["psf"]["jitter"] is False or config["psf"]["jitter"] is None:
        config["psf"]["jitter"] = 'none'
    if config["psf"]["jitter"] not in ["none", "fast", "slow"]:
        raise ValueError(f"Invalid value for psf jitter. It is {config['psf']['jitter']} but should be none, fast or slow")

    if "loss_type" not in config["optimization"]:
        config["optimization"]["loss_type"] = "joint"

    if config["optimization"]["loss_type"] not in ["joint", "marginal"]:
        raise ValueError(f"Invalid value for loss_type. It is {config['optimization']['loss_type']} but should be joint or marginal")

    if "filter_loss" in config["optimization"]:
        if config["optimization"]["filter_loss"] not in ["wiener", "wiener_time", "wiener_time_contrast", "lowpass"]:
            raise ValueError(f"Invalid value for filter_loss. It is {config['optimization']['filter_loss']} but should be wiener, wiener_time, wiener_time_contrast or lowpass")

    if "regularization" not in config or config["regularization"] is None:
        config["regularization"] = {}

    if "enforce_positive_object" not in config["optimization"]:
        config["optimization"]["enforce_positive_object"] = False
        
    if "shift" not in config["psf"]:
        config["psf"]["shift"] = False

    if "remove_tt" not in config["psf"]:
        config["psf"]["remove_tt"] = False

    if config["optimization"]["transform"] not in ["softplus", "none"]:
        raise ValueError(f"Invalid value for transform. It is {config['optimization']['transform']} but should be softplus or none")
    
    if config["initialization"]["object"] not in ["contrast", "average"]:
        raise ValueError(f"Invalid value for initialization of the object. It is {config['initialization']['object']} but should be contrast or average")
    
    if config["annealing"]["type"] not in ["sigmoid", "linear", "none"]:
        raise ValueError(f"Invalid value for annealing type. It is {config['annealing']['type']} but should be sigmoid, linear or none")
        
    if config["psf"]["model"] not in ["zernike", "kl", "pca", "nmf", "vae"]:
        raise ValueError(f"Invalid value for psf model. It is {config['psf']['model']} but should be zernike, kl, pca, nmf or vae")
    
    for k, v in config.items():
        if "object" in k:
            if v["image_filter"] not in ["tophat", "scharmer"]:
                raise ValueError(f"Invalid value for image_filter. It is {v['image_filter']} but should be tophat or scharmer")
            if "s_u_joint" not in v:
                v["s_u_joint"] = 100.0

    if "psd" not in config:
        config["psd"] = {}
        config["psd"]["K"] = 100.0
        config["psd"]["v0"] = 0.1
        config["psd"]["p"] = 2.0
    # Retain priors block fallback if present for backward compatibility
    if "priors" not in config:
        config["priors"] = {}
        config["priors"]["K"] = {'mean': config["psd"]["K"], 'sigma': 1.0}
        config["priors"]["v0"] = {'mean': config["psd"]["v0"], 'sigma': 1.0}
        config["priors"]["p"] = {'mean': config["psd"]["p"], 'sigma': 1.0}
                                    
    return config