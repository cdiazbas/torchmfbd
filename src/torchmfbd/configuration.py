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

    if "remove_fixed_pattern" not in config["images"]:
        config["images"]["remove_fixed_pattern"] = True
    if not isinstance(config["images"]["remove_fixed_pattern"], bool):
        raise ValueError(f"Invalid value for images.remove_fixed_pattern. It is {config['images']['remove_fixed_pattern']} but should be true or false")

    if "follow_motion" not in config["images"]:
        config["images"]["follow_motion"] = 0
    fm = config["images"]["follow_motion"]
    if isinstance(fm, bool) or not isinstance(fm, int) or fm < 0:
        raise ValueError(f"Invalid value for images.follow_motion. It is {fm} but should be 0 (off) or the largest shift in pixels")
    if "follow_motion_refine" not in config["images"]:
        config["images"]["follow_motion_refine"] = False
    if not isinstance(config["images"]["follow_motion_refine"], bool):
        raise ValueError(f"Invalid value for images.follow_motion_refine. It is {config['images']['follow_motion_refine']} but should be true or false")

    # Optimization
    if "gpu" not in config["optimization"]:
        config["optimization"]["gpu"] = -1

    if "half_spectrum" not in config["optimization"]:
        config["optimization"]["half_spectrum"] = True
    if not isinstance(config["optimization"]["half_spectrum"], bool):
        raise ValueError(f"Invalid value for optimization.half_spectrum. It is {config['optimization']['half_spectrum']} but should be true or false")

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

    # The time-coupled filters solve for an explicit object; the marginal loss has no term for them
    if config["optimization"].get("filter_loss", "wiener") in ["wiener_time", "wiener_time_contrast"] and config["optimization"]["loss_type"] == "marginal":
        raise ValueError(f"filter_loss {config['optimization']['filter_loss']} is only implemented for loss_type joint, not marginal")

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
            if v["image_filter"] not in ["tophat", "scharmer", "garrote"]:
                raise ValueError(f"Invalid value for image_filter. It is {v['image_filter']} but should be tophat, scharmer or garrote")
            if "s_u_joint" not in v:
                v["s_u_joint"] = 100.0

    # psd: initial values of the PSD parameters, no hyperprior. priors: initial values (the means) and
    # Gaussian hyperpriors on K, v0, p. A configuration gives one of them; with neither, psd defaults.
    if "psd" in config and "priors" in config and not config["priors"].get("derived_from_psd", False):
        raise ValueError("Give either a psd block (initial values) or a priors block (initial values and hyperpriors), not both")
    if "psd" not in config and "priors" not in config:
        config["psd"] = {}
        config["psd"]["K"] = 100.0
        config["psd"]["v0"] = 0.1
        config["psd"]["p"] = 2.0
    if "priors" not in config:
        config["priors"] = {}
        config["priors"]["K"] = {'mean': config["psd"]["K"], 'sigma': 1.0}
        config["priors"]["v0"] = {'mean': config["psd"]["v0"], 'sigma': 1.0}
        config["priors"]["p"] = {'mean': config["psd"]["p"], 'sigma': 1.0}
        config["priors"]["derived_from_psd"] = True
                                    
    return config