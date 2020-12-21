from sklearn.ensemble import RandomForestClassifier
from utils import *


# All custom model must have methods:
#     - fit(...)
#     - predict(...)
# You can add custom features like what I do below by making new class which inherited some class.
# If your model doesn't have the required methods, create custom ones for it.
class MyRandomForestClf(RandomForestClassifier):
    def __init__(self, n_estimators_added_before_each_fit, **kwargs):
        self.n_estimators_added_before_each_fit = n_estimators_added_before_each_fit
        super(MyRandomForestClf, self).__init__(**kwargs)

    def fit(self, x, y, **kwargs):
        self.n_estimators += self.n_estimators_added_before_each_fit
        super(MyRandomForestClf, self).fit(x, y, **kwargs)

######################################################

# To use the following tutorial,
# please set the "attributes" array in configure yaml file contains string "hp"

# Make an new model
#
# Uncomment example:
# init_model = MyRandomForestClf(n_estimators_added_before_each_fit=1)


# Call this function, you'll get a dict of:
#     key -> value: str(attribute) -> MyModel
#
# Uncomment example:
# attr_models = read_config_and_train(model=init_model, img_dir='data/images')


# How to save model?
#     MyModel.save_model(filename)
#
# Note:
#     Not sure whether using this function to save model that is not from sklearn is appropriate.
#
# Uncomment the example:
# attr_models['hp'].save_model('filename_you_want')

#
# How to load model?
#     - Existing MyModel load other one:
#              MyModel.load_model(file_path='filename_you_want')
#    - Creating new MyModel object:
#              MyModel(model_path_to_load='model_name_you_want_to_load')
#
# Note:
#     Not sure whether using this function to save model that is not from sklearn is appropriate.
#
# Uncomment the example:
# # For existing MyModel:
# attr_models['hp'].load_model('filename_you_want') # It'll cover your old model
# # For creating MyModel:
# new_my_model = MyModel(model_path_to_load='filename_you_want')

# Predict an image
#     MyModel.predict_one_image(
#         filename= ???(str),
#         show_original_img= ???(bool),
#         show_preprocessed_img= ???(bool,
#     )
#
# Uncomment the example:
# pred = attr_models['hp'].predict_one_image(filename=filename,
#                                   show_original_img=True,
#                                   show_preprocessed_img=True)
# print(pred)

######################################################
