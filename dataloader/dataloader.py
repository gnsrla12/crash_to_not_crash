def CreateDataLoader(opt, sess=None):
    dataloader = None
    if opt.dataset_mode == 'sample_per_vehicle':
        from dataloader.sample_per_vehicle import Dataset
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print()
    print("Creating dataloader [%s]" % (opt.dataset_mode))
    dataloader = Dataset()
    dataloader.initialize(opt, sess)
    print("Dataloader [%s] Created" % (dataloader.name()))
    
    return dataloader
