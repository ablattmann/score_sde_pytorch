from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.batch_size = 10

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.dataset = 'Imagenet'
  data.centered = True
  data.image_size = 256
  data.random_crop=True
  data.num_classes = 1000

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 4, 4)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.class_conditional=True

  # optim
  optim = config.optim
  optim.lr = 2e-5

  return config
