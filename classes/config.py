import yaml

def merge_dictionaries_recursively(custom_dict, default_dict):
    ''' Update two config dictionaries recursively.
    Args:
    dict1 (dict): first dictionary to be updated
    dict2 (dict): second dictionary which entries should be preferred
    https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    '''
    if default_dict is None: return

    for key, value in default_dict.items():
        if key not in custom_dict:
            custom_dict[key] = dict()
        if isinstance(value, dict):
            merge_dictionaries_recursively(custom_dict[key], value)
        else:
            custom_dict[key] = value
    return custom_dict



class Config(object):  
    """ 
    https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    """

    def __init__(self, config_path, default_path=None):

        with open(config_path) as cf_file:
            cfg = yaml.safe_load( cf_file.read() )

        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = yaml.safe_load( def_cf_file.read() )
            
            cfg = merge_dictionaries_recursively(default_cfg, cfg)
        
        self._data = cfg


    def get(self, path=None, default=None):

        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)
            value = sub_dict.get(data_item, default)
            return value

        except (TypeError, AttributeError):
            return default