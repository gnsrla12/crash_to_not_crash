
def CreateModel(opt, sess, pos_weight=1):
    model = None
    if opt.model == 'per_vehicle':
        assert(opt.dataset_mode == 'sample_per_vehicle')
        from .per_vehicle_model import per_vehicle_model
        model = per_vehicle_model()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print()
    print("model [%s] was created" % (model.name()))
    model.initialize(opt, sess, pos_weight=pos_weight)

    return model
