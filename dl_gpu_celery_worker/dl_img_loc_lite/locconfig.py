import torch
from typing import Union

class LocConfig():
    def __init__(self,
            # Model and dataset
            dataset: Union[int, str] = 6, # one of 1,2,3,4,5,6, or strings listed below
            data_split: str = 'random', #one of 'random', 'gridK' (where K is integer), 'radiusK'. If dataset==6, 'driving', 'april','july','nov'
            arch: str = 'unet', # String to select model, see localization.py:DLLocalization:build_model()
            min_inputs = None, # String
            meter_scale = None,
            random_state: int = 0,
            one_tx: bool = True,
            device_multiplication: bool = False,
            category_multiplication: bool = True,
            remove_mobile: bool = False,
            use_alt_for_ds8_grid2: bool = True,
            # Training
            test_size: float = 0.2,
            tx_marker_value: float = 0.01,
            batch_size: int = 32,
            better_epoch_limit: int = 50,
            lr: float = 5e-4,
            device = None,
            make_val = True,
            # Data Augmentations
            apply_sensor_dropout: bool = False,
            min_dropout_inputs: int = None,
            apply_rss_noise: bool = False, # Scale RSS randomly
            power_limit: float = 0.3,
            apply_power_scaling: bool = False, # Scale RSS uniformly
            scale_limit: float = 0.3,
            adv_train: bool = False,
            include_elevation_map: bool = False,
            should_augment = False,
            augmentation = None,
    ):
        dataset_strings = {'utah44':1, 'outdoor44':2, 'hallways2tx':3, 'outdoor2tx':4, 'orbit5tx':5, 'utah_frs':6, 'antwerp_lora':7, 'utah_cbrs':8, 'bounded_SD':9}
        dataset_options = [1,2,3,4,5,6,7,8,9, 'utah44', 'outdoor44', 'hallways2tx', 'outdoor2tx', 'orbit5tx', 'utah_frs', 'antwerp_lora', 'utah_cbrs', 'bounded_SD']
        assert dataset in dataset_options
        if isinstance(dataset, str):
            dataset = dataset_strings[dataset]
        self.dataset_index = dataset
        self.set_default_options(min_inputs, meter_scale, min_dropout_inputs)
        self.data_split = data_split
        self.arch = arch
        self.random_state = random_state
        self.one_tx = one_tx 
        self.device_multiplication = device_multiplication
        self.category_multiplication = category_multiplication
        self.remove_mobile = remove_mobile
        self.use_alt_for_ds8_grid2 = use_alt_for_ds8_grid2

        self.test_size = test_size
        self.training_size = 1 - test_size
        self.tx_marker_value = tx_marker_value
        self.batch_size = batch_size
        self.better_epoch_limit = better_epoch_limit
        self.lr = lr
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.make_val = make_val

        self.sensor_dropout = apply_sensor_dropout
        self.apply_rss_noise = apply_rss_noise
        self.power_limit = power_limit
        self.apply_power_scaling = apply_power_scaling
        self.scale_limit = scale_limit
        self.adv_train = adv_train
        self.include_elevation_map = include_elevation_map
        self.should_augment = should_augment
        self.augmentation = augmentation
        self.partial_train = 0
        self.printable_keys = {
            'dataset_index': 'DS',
            'random_state': 'randstate',
            'test_size': 'test_size',
            'arch': 'arch',
            'data_split': 'split',
            'meter_scale': 'img_scale',
            'include_elevation_map': 'elev',
            'adv_train': 'AdvTr',
            'augmentation': 'Aug'
        }

    def __str__(self):
        if self.apply_rss_noise:
            self.printable_keys['power_limit'] = 'rand_pow'
        if self.apply_power_scaling:
            self.printable_keys['scale_limit'] = 'scale_pow'
        if self.device_multiplication:
            self.printable_keys['device_multiplication'] = 'DevMult'
        if self.category_multiplication:
            self.printable_keys['category_multiplication'] = 'CatMult'
        if self.sensor_dropout:
            self.printable_keys['min_dropout_inputs'] = 'dropout'
        if self.partial_train > 0:
            self.printable_keys['partial_train'] = 'partialTrain'
        members = [member for member in dir(self) if not member.startswith('__')]
        param_string = ''
        for member in members:
            attr = getattr(self, member)
            if member in self.printable_keys and self.printable_keys[member] is not None:
                param_string += '%s:%s__' % (self.printable_keys[member], attr )
        param_string = param_string[:-2]
        return param_string

    def set_default_options(self, min_inputs: int, meter_scale: int, min_dropout_inputs: int):
        ds = self.dataset_index
        if ds in [1,2,3,4,5]:
            self.min_sensors = 4 if min_inputs is None else min_inputs
            self.min_dropout_inputs = 4 if min_dropout_inputs is None else min_dropout_inputs
            self.meter_scale = 1 if meter_scale is None else meter_scale
        elif ds == 6:
            self.min_sensors = 5 if min_inputs is None else min_inputs
            self.min_dropout_inputs = 15 if min_dropout_inputs is None else min_dropout_inputs
            self.meter_scale = 25 if meter_scale is None else meter_scale
        elif ds == 7:
            self.min_sensors = 5 if min_inputs is None else min_inputs
            self.min_dropout_inputs = 5 if min_dropout_inputs is None else min_dropout_inputs
            self.meter_scale = 100 if meter_scale is None else meter_scale
        elif ds == 8:
            self.min_sensors = 5 if min_inputs is None else min_inputs
            self.min_dropout_inputs = 4 if min_dropout_inputs is None else min_dropout_inputs
            self.meter_scale = 25 if meter_scale is None else meter_scale
        elif ds == 9:
            self.min_sensors = 5 if min_inputs is None else min_inputs
            self.min_dropout_inputs = 4 if min_dropout_inputs is None else min_dropout_inputs
            self.meter_scale = 100 if meter_scale is None else meter_scale