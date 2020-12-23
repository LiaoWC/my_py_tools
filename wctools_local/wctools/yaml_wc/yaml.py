import yaml

'''
How to install the python yaml package--"pyyaml"
1. pip install pyyaml
2. import yaml
'''


class YAML:
    def __init__(self, yaml_path=None):
        # Initialize
        self.__data = None
        # Load data depending on whether the param is given
        if yaml_path is not None:
            self.load(yaml_path=yaml_path)

    def load(self, yaml_path):
        # Load yaml data (as a dict)
        with open(yaml_path) as yaml_file:
            self.__data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Get the data dict
    def get_data(self):
        return self.__data

