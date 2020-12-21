from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
import my_py_tools
from werkzeug.utils import secure_filename  # https://blog.csdn.net/weixin_44493841/article/details/107864437
from my_py_tools.yaml.yaml import YAML
from my_py_tools.plt.barh import plot_horizontal_bar
from my_py_tools.numpy.frequencies import get_frequencies
# https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/
from my_py_tools.image.size_manupilation import resize_with_zero_padding, random_crop
import seaborn as sn
import copy
import time
import cv2
from PIL import Image
import joblib
import werkzeug

# Docstring (numpy format)
# https://numpydoc.readthedocs.io/en/latest/format.html

plt.style.use('seaborn')


##############################################################################
# === Setting structures ===
class RandomCropSetting:
    def __init__(self,
                 # Specify a size the img will be resize before cropping
                 use: bool,
                 resize_width_height: tuple,
                 crop_width_height: tuple):
        self.use = use
        self.resize_width_height = resize_width_height
        self.crop_width_height = crop_width_height


class EdgesSetting:
    def __init__(self,
                 use: bool,
                 threshold1: int,
                 threshold2: int):
        self.use = use
        self.threshold1 = threshold1
        self.threshold2 = threshold2


class ValidationSetting:
    def __init__(self,
                 method: str,
                 k_fold_n_folds: int,
                 holdout_size: int
                 ):
        self.method = method  # 'none', 'k_fold', or 'holdout'
        self.k_fold_n_folds = k_fold_n_folds
        self.holdout_size = holdout_size


##############################################################################

def get_img_data_from_filename(filename: str,
                               target_width_height: tuple,
                               random_crop_setting: RandomCropSetting,
                               edges_setting: EdgesSetting,
                               pil_img_mode: str,
                               return_mode: str):
    """
    Given an image's file path, get its image data.
    The image is processed by PIL Image and numpy.
    :param filename: The path of the image.
    :param target_width_height:
    :param random_crop_setting:
    :param edges_setting:
    :param pil_img_mode:
    :param return_mode: "PIL_Image"'" or "ndarray"
    :return: If return_mod is "PIL_Image", return a PIL Image; if return_mod is "ndarray", return a numpy array.
    """
    # Open image
    img = None  # Variable "img" will be in PIL Image format.
    if pil_img_mode != 'RGBA':
        # If mode isn't RGBA, remove transparency.
        # Using cv2 to avoid PIL's warning message of turn RGBA into RGB
        img_arr = cv2.imread(filename=filename)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
        img = Image.fromarray(img_arr)
    else:
        img = Image.open(filename)
    # Convert PIL image mode
    img = img.convert(pil_img_mode)
    # Resize first
    img = resize_with_zero_padding(img=img,
                                   input_mode='PIL_Image',
                                   target_size=target_width_height,
                                   return_mode='PIL_Image')
    img = img.convert(mode=pil_img_mode)

    # Random crop
    if random_crop_setting.use is True:
        img = random_crop(img=img,
                          bbox_size=random_crop_setting.crop_width_height,
                          input_mode='PIL_Image',
                          min_size=random_crop_setting.crop_width_height,
                          all_possible=False,
                          return_mode='PIL_Image')

    # Edges
    if edges_setting.use is True:
        get_edges(pil_img=img,
                  threshold1=edges_setting.threshold1,
                  threshold2=edges_setting.threshold2)

    # Image color format
    img = img.convert(mode=pil_img_mode)

    # Done. Return.
    if return_mode == 'PIL_Image':
        return img
    elif return_mode == 'ndarray':
        return np.array(img)
    else:
        raise ValueError('Invalid return_mode.')


def get_images_data_from_filenames(filenames: np.ndarray,
                                   target_width_height: tuple,
                                   random_crop_setting: RandomCropSetting,
                                   edges_setting: EdgesSetting,
                                   pil_img_mode: str) -> np.ndarray:
    """

    :param filenames: list
    :param target_width_height:
    :param random_crop_setting:
    :param edges_setting:
    :param pil_img_mode:
    :return: If filenames is list, return a list; if filenames is numpy array, return a numpy array.
    """
    return np.array([get_img_data_from_filename(filename=filename,
                                                target_width_height=target_width_height,
                                                random_crop_setting=random_crop_setting,
                                                edges_setting=edges_setting,
                                                pil_img_mode=pil_img_mode,
                                                return_mode='ndarray') for filename in filenames])


#
def simply_train(model,
                 x_filenames,
                 x_test,
                 validation: bool,
                 y_train,
                 y_test,
                 n_epochs,
                 batch_size,
                 input_width_height: tuple,
                 pil_img_mode: str,
                 random_crop_setting: RandomCropSetting,
                 edges_setting: EdgesSetting
                 ):
    """
    :param model:
    :param x_filenames: Image names for making training input.
    :param x_test: Image numpy array for testing input.
    :param validation: If it's False, no need to set x_test and y_test.
    :param y_train:
    :param y_test:
    :param n_epochs:
    :param batch_size:
    :param input_width_height:
    :param pil_img_mode:
    :param random_crop_setting:
    :param edges_setting:
    :return: If validation is True, return last_accuracy_score and last_y_pred; if is False, there's no return
    """
    ################
    # Make variables
    ################
    x_train_size = x_filenames.size
    n_iterations = x_train_size // batch_size if x_train_size % batch_size == 0 \
        else x_train_size // batch_size + 1
    last_accuracy_score = None
    last_y_pred = None
    ########
    # Epochs
    ########
    for cur_epoch_num in range(1, n_epochs + 1):  # From 1 to n_epochs
        #########################
        # Record epoch start time
        #########################
        start_time = time.time()
        ######################
        # Initialize variables
        ######################
        accumulated_x = 0
        ############
        # Iterations
        ############
        for cur_iteration_num in range(1, n_iterations + 1):  # From 1 to n_iterations
            ########################
            # Print info dynamically
            ########################
            print('Epoch {} iteration [{}/{}]'.format(cur_epoch_num, cur_iteration_num, n_iterations),
                  end='' if cur_iteration_num == n_iterations else '\r')
            ################
            # Make variables
            ################
            # Calculate current iteration's actual batch size
            cur_iteration_batch_size = min(batch_size, x_train_size - accumulated_x)
            # For collecting batch data
            x_batch_data = []
            y_batch_data = []
            ####################
            # Collect batch data
            ####################
            for cur_x_number in range(1, cur_iteration_batch_size + 1):  # From 1 to cur_iteration_batch_size
                #####################################
                # Get image data from image file path
                #####################################
                # Get idx of x_filenames
                idx = accumulated_x + cur_x_number - 1
                # Get image data
                img = get_img_data_from_filename(filename=x_filenames[idx],
                                                 target_width_height=input_width_height,
                                                 random_crop_setting=random_crop_setting,
                                                 edges_setting=edges_setting,
                                                 pil_img_mode=pil_img_mode,
                                                 return_mode='PIL_Image')

                #####################################
                # Put batch data in the current batch
                #####################################
                x_batch_data.append(np.array(img).flatten())  # Option TODO: custom img arr functions setting
                y_batch_data.append(y_train[idx])
            ####################################
            # Make the batch data in numpy array
            ####################################
            x_batch_data = np.array(x_batch_data)
            y_batch_data = np.array(y_batch_data)
            ###############
            # Fit the model
            ###############
            model.fit(x_batch_data, y_batch_data)
            ###############################################
            # Renew number of x_filename has been processed
            ###############################################
            accumulated_x += cur_iteration_batch_size
        ############################################
        # Record end time and print the elapsed time
        ############################################
        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(" elapsed_time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), end='')
        ############
        # Validation
        ############
        if validation is True:
            y_pred = model.predict(np.array([x.flatten() for x in x_test]))
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            last_accuracy_score = accuracy
            last_y_pred = y_pred
            print(" val_acc: {:.3f}".format(last_accuracy_score))
    ################################################
    # Decide what to return depening on "validation"
    ################################################
    if validation is True:
        return last_accuracy_score, last_y_pred  # TODO: Not sure here.


#
def holdout_training(model,
                     x_filenames,
                     y_data,
                     n_epochs: int,
                     batch_size: int,
                     input_width_height: tuple,
                     pil_img_mode: str,
                     random_crop_setting: RandomCropSetting,
                     edges_setting: EdgesSetting,
                     test_size=0.2):
    """

    :param model:
    :param x_filenames:
    :param y_data:
    :param n_epochs:
    :param batch_size:
    :param input_width_height:
    :param pil_img_mode:
    :param random_crop_setting:
    :param edges_setting:
    :param test_size:
    :return: the model after being trained ,confusion_matrix (ndarray), and last_accuracy_score
    """
    #######
    # Print
    #######
    print("Holdout (test_size={})".format(test_size))
    ###########################
    # Make a deep copy of model
    ###########################
    model = copy.deepcopy(model)
    ##################
    # Train test split
    ##################
    x_filenames_train, x_filenames_test, y_train, y_test = train_test_split(x_filenames, y_data,
                                                                            test_size=test_size)
    #########################################
    # Get x_test data in correct input format
    #########################################
    x_filenames = np.array(x_filenames) if isinstance(x_filenames, list) else x_filenames
    x_test = get_images_data_from_filenames(filenames=x_filenames_test,
                                            target_width_height=input_width_height,
                                            random_crop_setting=random_crop_setting,
                                            edges_setting=edges_setting,
                                            pil_img_mode=pil_img_mode)
    #######
    # Train
    #######
    last_accuracy_score, last_y_pred = simply_train(model=model,
                                                    x_filenames=x_filenames_train,
                                                    x_test=x_test,
                                                    validation=True,
                                                    y_train=y_train,
                                                    y_test=y_test,
                                                    n_epochs=n_epochs,
                                                    batch_size=batch_size,
                                                    input_width_height=input_width_height,
                                                    pil_img_mode=pil_img_mode,
                                                    random_crop_setting=random_crop_setting,
                                                    edges_setting=edges_setting)

    print()
    return model, confusion_matrix(y_true=y_test, y_pred=last_y_pred, labels=np.arange(11)), last_accuracy_score


def k_fold_training(model,
                    x_filenames,
                    y_data,
                    n_epochs: int,
                    batch_size: int,
                    input_width_height: tuple,
                    pil_img_mode: str,
                    random_crop_setting: RandomCropSetting,
                    edges_setting: EdgesSetting,
                    formalize_cm=True,
                    only_return_best=True,
                    n_folds=3):
    """

    :param model:
    :param x_filenames:
    :param y_data:
    :param n_epochs:
    :param batch_size:
    :param input_width_height:
    :param pil_img_mode:
    :param random_crop_setting:
    :param edges_setting:
    :param formalize_cm:
    :param n_folds:
    :param only_return_best:
    :return: If "only_return_best" is True, return best_model, cm_after_added, avg_accuracy;
        if it's False, return k_models, cm_after_added, k_model_accuracies.
    """
    ###############
    # Make k models
    ###############
    k_models = [copy.deepcopy(model) for i in range(n_folds)]
    k_model_accuracies = []
    #############
    # Make K_Fold
    #############
    kf = KFold(n_splits=n_folds)
    ######################
    # Initialize variables
    ######################
    total_accuracy = 0
    total_confusion_matrices = []
    ################
    # Loop each fold
    ################
    for i, (train_idx, test_idx) in enumerate(kf.split(x_filenames)):
        #################
        # Print fold info
        #################
        print("^^^ Fold-{} ^^^".format(i + 1))
        ###############
        # Get fold data
        ###############
        x_filenames_train, x_filenames_test = x_filenames[train_idx], x_filenames[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        ###################################
        # Get x_filenames_test's image data
        ###################################
        x_test = get_images_data_from_filenames(filenames=x_filenames_test,
                                                target_width_height=input_width_height,
                                                random_crop_setting=random_crop_setting,
                                                edges_setting=edges_setting,
                                                pil_img_mode=pil_img_mode)
        #######
        # Train
        #######
        accuracy, y_last_pred = simply_train(model=k_models[i],
                                             x_filenames=x_filenames_train,
                                             x_test=x_test,
                                             validation=True,
                                             y_train=y_train,
                                             y_test=y_test,
                                             n_epochs=n_epochs,
                                             batch_size=batch_size,
                                             input_width_height=input_width_height,
                                             pil_img_mode=pil_img_mode,
                                             random_crop_setting=random_crop_setting,
                                             edges_setting=edges_setting)

        ########
        # Record
        ########
        k_model_accuracies.append(accuracy)
        total_accuracy += accuracy
        ##########################
        # Collect confusion matrix
        ##########################
        total_confusion_matrices.append(confusion_matrix(y_test, y_last_pred, labels=np.arange(11)))
        ####################
        # Print a empty line
        ####################
        print()
    ######################
    # Print accuracy score
    ######################
    avg_accuracy = total_accuracy / n_folds
    print("Avg accuracy score: {:.3f}\n".format(avg_accuracy))
    #######################
    # Add all cm and return
    #######################
    # Add all confusion matrices
    cm_after_added = total_confusion_matrices[0]
    for i in range(1, len(total_confusion_matrices)):
        cm_after_added = np.add(cm_after_added, total_confusion_matrices[i])
    # Determine whether to formalize
    if formalize_cm is True:
        cm_after_added = cm_after_added / (np.linalg.norm(cm_after_added))
    ########
    # Return
    ########
    if only_return_best is True:
        best_model = k_models[k_model_accuracies.index(min(k_model_accuracies))]
        return best_model, cm_after_added, avg_accuracy
    elif only_return_best is False:
        return k_models, cm_after_added, k_model_accuracies
    else:
        raise ValueError('Invalid value of parameter "only_return_best".')


def get_edges(pil_img: Image, threshold1, threshold2) -> Image:
    """
    :param pil_img: Input a PIL Image.
    :param threshold1: parameter for cv2.Canny
    :param threshold2: parameter for cv2.Canny
    :return: An edges-image in PIL Image format.
    """
    img = pil_img
    return Image.fromarray(
        cv2.Canny(image=np.array(img.convert('RGB')), threshold1=threshold1, threshold2=threshold2))


def get_config(config_filename):
    try:
        return YAML(config_filename).get_data()
    except ValueError(
            'Notes that if the config file is not found, you may have not make a new config.yaml'
            'from config.yaml.example yet.'):
        pass


def plot_and_save_confusion_matrix(cm,
                                   title: str,
                                   description: str,
                                   font_size: int,
                                   figsize: tuple,
                                   round_precision: str,
                                   save: bool,
                                   save_name: str,
                                   show: bool):
    plt.rcdefaults()
    font = {'size': font_size}
    matplotlib.rc('font', **font)
    plt.figure(figsize=figsize)
    plt.title(title)
    _ = sn.heatmap(pd.DataFrame(data=cm,
                                index=list(np.arange(11)),
                                columns=list(np.arange(11))),
                   annot=True,
                   cmap=plt.cm.Blues,
                   fmt='{}'.format(round_precision))
    plt.savefig(save_name)
    if show is True:
        plt.show()


# Model here must have method "fit"
def train_predicting_attr_score_by_filenames(model,
                                             n_epochs,
                                             batch_size,
                                             input_width_height: tuple,
                                             x_filenames,
                                             y_data,
                                             pil_img_mode: str,
                                             valid_setting: ValidationSetting,
                                             random_crop_setting: RandomCropSetting,
                                             edges_setting: EdgesSetting,
                                             formalize_cm: bool):
    """

    :param model:
    :param n_epochs:
    :param batch_size:
    :param input_width_height:
    :param x_filenames:
    :param y_data:
    :param pil_img_mode:
    :param valid_setting:
    :param random_crop_setting:
    :param edges_setting:
    :param formalize_cm:
    :return: model, cm, accuracy
    """
    #######
    # Print
    #######
    print('Epochs: {}'.format(n_epochs))
    print('Batch size: {}'.format(batch_size))
    print('Input size(w,h): ({},{})'.format(input_width_height[0], input_width_height[1]))
    print()
    ######################
    # Initialize variables
    ######################
    cm = None
    ######################################
    # Train depending on validation method
    ######################################
    if valid_setting.method == 'k_fold':
        # Notes: Here only get best model in all folds.
        model, cm, accuracy = k_fold_training(model=model,
                                              x_filenames=x_filenames,
                                              y_data=y_data,
                                              n_epochs=n_epochs,
                                              batch_size=batch_size,
                                              input_width_height=input_width_height,
                                              pil_img_mode=pil_img_mode,
                                              random_crop_setting=random_crop_setting,
                                              edges_setting=edges_setting,
                                              formalize_cm=formalize_cm,
                                              only_return_best=True,
                                              n_folds=valid_setting.k_fold_n_folds)
        return model, cm, accuracy
    elif valid_setting.method == 'holdout':
        model, cm, accuracy = holdout_training(model=model,
                                               x_filenames=x_filenames,
                                               y_data=y_data,
                                               n_epochs=n_epochs,
                                               batch_size=batch_size,
                                               input_width_height=input_width_height,
                                               pil_img_mode=pil_img_mode,
                                               random_crop_setting=random_crop_setting,
                                               edges_setting=edges_setting,
                                               test_size=valid_setting.holdout_size)
        return model, cm, accuracy
    elif valid_setting.method == 'none':
        model = simply_train(model=model,
                             x_filenames=x_filenames,
                             x_test=np.array([]),
                             validation=False,
                             y_train=y_data,
                             y_test=np.array([]),
                             n_epochs=n_epochs,
                             batch_size=batch_size,
                             input_width_height=input_width_height,
                             pil_img_mode=pil_img_mode,
                             random_crop_setting=random_crop_setting,
                             edges_setting=edges_setting)
    else:
        raise ValueError('Invalid value of validation method.')


##############################################################################
# Only for this project functions
##############################################################################

class MyModel:
    def __init__(self, model):
        self.model = model

    def set_model(self, model):
        self.model = model

    def save_sklearn_model(self, model, file_path):
        joblib.dump(self.model, filename=file_path)

    def load_sklearn_model(self, file_path):
        self.model = joblib.load(filename=file_path)


class RatingRecord:
    """
    """
    ALL_ATTRS = ['hp', 'pa', 'ma', 'sp', 'cr']

    def __init__(self, record_filename, img_dir):
        """
        Public members:
            - record_filename
            - df_all_record: A Pandas Dataframe
            - df_attrs: A dict of Pandas Dataframes (key is attribute; value is dataframe.)

        :param record_filename:
        """
        self.record_filename = record_filename
        self.img_dir = img_dir
        self.df_all_record = self.read_csv_and_add_img_dir_as_prefix()  # Pandas Dataframe
        self.df_attrs = {attr: self.df_all_record.filter(['name', 'image', attr]) for attr in RatingRecord.ALL_ATTRS}

    def read_csv_and_add_img_dir_as_prefix(self):
        # Read csv
        df = pd.read_csv(self.record_filename)
        # Add prefix to img filename to make completed filename
        for idx in df.index:
            df.at[idx, 'image'] = os.path.join(self.img_dir, df.at[idx, 'image'])
        return df

    def plot_how_many_images_each_rater_has_rated(self):
        """
        Plot how many images each rater has rated

        :return:
        """
        rater_to_num = self.df_all_record.groupby('name').size()
        plot_horizontal_bar(rater_to_num.values,
                            rater_to_num.index.values,
                            x_label='Number of images has rated',
                            title='How many images each rater has rated?',
                            show=False)

    def plot_rating_distribution_of_all_attrs(self):
        """
        See if there's anything outside the distribution range (0~10)

        :return:
        """
        print("Valid range: 0~10")
        for attr in RatingRecord.ATTRS:
            frequencies, unique, counts = get_frequencies(data=self.df_all_record[attr].values,
                                                          return_unique_and_counts=True)
            min_uniq, max_uniq = min(unique.tolist()), max(unique.tolist())
            print("Attribute {}: min={}, max={} => {}".format(attr, min_uniq, max_uniq,
                                                              "VALID" if min_uniq >= 0 and max_uniq <= 10 else "INVALID"))
            plt.figure(figsize=(10, 2))
            plt.bar([x[0] for x in frequencies], [x[1] for x in frequencies])
            plt.show()

    def plot_each_rater_rating_distribution_of_each_attribute(self, save_dir):
        """
        Plot each rater's rating distribution of each attribute

        :param save_dir:
        :return:
        """
        for attr in RatingRecord.ALL_ATTRS:
            # Group and collect as lists
            # https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby
            rater_to_attr_set = self.df_attrs[attr].groupby(['name'])[attr].apply(list)
            rater_num = len(rater_to_attr_set)  # Number of raters
            rater_hp_distributions = {}
            for i in range(rater_num):
                (unique_nums, counts) = np.unique(np.array(rater_to_attr_set.values[i]), return_counts=True)
                unique_nums, counts = unique_nums.tolist(), counts.tolist()
                frequencies = {unique_nums[i]: counts[i] for i in range(len(unique_nums))}
                distribution = []
                for j in range(0, 11):  # 0~10
                    distribution.append(0) if j not in frequencies else distribution.append(frequencies[j])
                rater_hp_distributions[rater_to_attr_set.index[i]] = distribution
            df = pd.DataFrame(data=rater_hp_distributions)
            ax = df.plot.bar(stacked=True, figsize=(15, 10), fontsize=20)
            ax.legend(prop={'size': 20})  # Set legend size
            title = "Each rater's rating distribution of {}".format(attr)
            ax.set_title(label=title, fontdict={'fontsize': 30})  # Set title
            plt.savefig(os.path.join(save_dir, secure_filename(title)))


def read_config_and_train(model, img_dir):
    """
    Model here must have method "fit"

    :param model:
    :param img_dir:
    :return: modes (dict of (attr -> model)
    """
    ################
    # Read configure
    ################
    config = get_config('config.yaml')
    ####################
    # Variables
    ####################
    n_epochs = config['train']['n_epochs']
    batch_size = config['train']['batch_size']
    input_width_height = (config['input']['input_size']['width'], config['input']['input_size']['height'])
    attributes = config['attributes']
    #######
    # Print
    #######
    print('Epochs: {}'.format(n_epochs))
    print('Batch size: {}'.format(batch_size))
    print('Input size(w,h): {}'.format(input_width_height))
    print()
    ####################
    # Load rating record
    ####################
    record = RatingRecord(record_filename=config['path']['data']['rating_record'], img_dir=img_dir)
    ############################
    # Record models of all attrs
    ############################
    models = {}
    ############################
    # Loop in all specified attr
    ############################
    for attr in attributes:
        #######
        # Print
        #######
        print("===== Attribute {} =====\n".format(attr))
        ##########################
        # Determine use whose data
        ##########################
        use_whose_data = config['train']['rater']
        df_rater = record.df_attrs[attr] if use_whose_data == 'all' else \
            record.df_attrs[attr][record.df_attrs[attr]['name'] == use_whose_data]
        ############
        # Drop zero
        ############
        df_rater = df_rater[df_rater[attr] != 0]
        #############################
        # Make x_filenames and y_data
        #############################
        x_filenames = np.array([name for name in df_rater['image'].values])
        y_data = np.array([v for v in df_rater[attr].values])
        ###############
        # Make settings
        ###############
        random_crop_setting = RandomCropSetting(
            use=True if config['train']['random_crop']['use_random_crop'] == 'true' else False,
            resize_width_height=(
                config['train']['random_crop']['resize']['width'],
                config['train']['random_crop']['resize']['width']),
            crop_width_height=input_width_height)
        edges_setting = EdgesSetting(use=config['input']['edges']['use_edges'] == 'true',
                                     threshold1=config['input']['edges']['threshold1'],
                                     threshold2=config['input']['edges']['threshold2'])
        valid_setting = ValidationSetting(method=config['train']['validation'],
                                          k_fold_n_folds=config['train']['K_fold_number'],
                                          holdout_size=config['train']['holdout_validation_size'])
        #######
        # Train
        #######
        trained_model, cm, accuracy = train_predicting_attr_score_by_filenames(model=copy.deepcopy(model),
                                                                               n_epochs=n_epochs,
                                                                               batch_size=batch_size,
                                                                               input_width_height=input_width_height,
                                                                               x_filenames=x_filenames,
                                                                               y_data=y_data,
                                                                               pil_img_mode=config['input'][
                                                                                   'pil_img_mode'],
                                                                               valid_setting=valid_setting,
                                                                               random_crop_setting=random_crop_setting,
                                                                               edges_setting=edges_setting,
                                                                               formalize_cm=True if
                                                                               config['confusion_matrix'][
                                                                                   'normalize'] == 'true' else False)

        models[attr] = trained_model
        #######################
        # Plot confusion matrix
        #######################
        if valid_setting.method != 'none':
            #####################
            # Make cm description
            #####################
            cm_description = 'Attribute="{}", epochs={}, batch_size={}, size=(w,h)=({},{}), color="{}"'.format(
                attr, n_epochs, batch_size, input_width_height[0], input_width_height[1],
                config['input']['pil_img_mode'])
            random_crop_description = 'False' if random_crop_setting.use is False \
                else 'True, min_size(w,h)=({},{})'.format(random_crop_setting.resize_width_height[0],
                                                          random_crop_setting.resize_width_height[1])
            cm_description = cm_description + ', {}'.format(random_crop_description)
            #########
            # Plot cm
            #########
            plot_and_save_confusion_matrix(cm=cm,
                                           title=cm_description,
                                           description=cm_description,
                                           font_size=40,
                                           figsize=(50, 35),
                                           round_precision=config['confusion_matrix']['round_precision'],
                                           save=True if config['confusion_matrix']['save'] == 'true' else False,
                                           save_name='statistics/{}'.format(cm_description),
                                           show=True if config['confusion_matrix']['show'] == 'true' else False)
    return models


#
# class MyModelTool:
#     #
#     CONFIG_PATH = "config.yaml"
#     ATTRS = ['hp', 'pa', 'ma', 'sp', 'cr']
#     '''
#     Warning: All image will be turned into RGB in default.
#     '''
#
#     #
#     def __init__(self):
#         # Load configure
#         self.config = YAML(MyModelTool.CONFIG_PATH).get_data()
#         # Get values from configure
#         # Input config
#         self.input_width_heigth = (
#             self.config['input']['input_size']['width'], self.config['input']['input_size']['height'])
#         self.input_color_format = self.config['input']['color']  # Image color format
#         self.pil_img_mode = self.config['input']['pil_img_mode']
#         self.use_edges = self.config['input']['edges']['use_edges']
#         # === Read "train" config ===
#         # Random crop settings
#         self.use_random_crop = self.config['train']['random_crop']['use_random_crop']
#         self.random_crop_width, self.random_crop_height = (self.config['train']['random_crop']['resize']['width'],
#                                                            self.config['train']['random_crop']['resize']['height'])
#         #
#         self.epochs = self.config['train']['epoch']
#         # batch_size = config['train']['batch_size'] if config['train']['batch_size'] != 'all' else
#         self.batch_size = self.config['train']['batch_size']
#
#         self.validation = self.config['train']['validation']
#         self.k_fold_number = self.config['train']['K_fold_number']
#         self.holdout_size = self.config['train']['holdout_validation_size']
#         self.use_whose_data = self.config['train']['rater']
#         # # Mode config
#         # self.model_type = self.config['model']['type']
#         # self.model = None  # Initially None
#
#         # # Load rating record
#         # self.df_record = pd.read_csv(self.config['path']['data']['rating_record'])
#
#         # # Make each attribute's dataframe (name, image, attribute)
#         # self.df_attrs = {}
#         # for attr in MyModelTool.ATTRS:
#         #     self.df_attrs[attr] = self.df_record.filter(['name', 'image', attr], axis=1)
#
#     #
#     def get_file_path(self, file_name):
#         return os.path.join(self.config['path']['data']['all_images'], file_name)
#
#     # # Get PIL color mode
#     # def get_pil_color_mode(self):
#     #     if self.input_color_format == 'RGB':
#     #         return 'RGB'
#     #     elif self.input_color_format == 'Grayscale':
#     #         return 'LA'
#     #     else:
#     #         raise ValueError('Invalid input color format.')
#     #
#     # # Get X_test_data
#     # def get_x_test_data(self, x_test_filenames):
#     #     x_test_data = []
#     #     for x_test_file in x_test_filenames:
#     #         img = resize_with_zero_padding(img=self.get_file_path(file_name=x_test_file),
#     #                                        input_mode='file_path',
#     #                                        target_size=self.input_size,
#     #                                        return_mode='PIL_Image').convert(
#     #             self.pil_color_mode)
#     #         if self.use_edges == 'true':
#     #             img = self.get_edges(pil_img=img)
#     #         x_test_data.append(np.array(img).flatten())
#     #     x_test_data = np.array(x_test_data)
#     #     return x_test_data
#
#
# # # Model
# # def get_model(self):
# #     if self.model_type == 'random_forest_clf':
# #         setting = self.config['model']['random_forest_clf']
# #         return RandomForestClassifier(
# #             n_estimators=setting['n_estimators'],
# #             criterion=setting['criterion'],
# #             max_depth=setting['max_depth'],
# #             min_samples_split=setting['min_samples_split'],
# #             min_samples_leaf=setting['min_samples_leaf'],
# #             # max_leaf_nodes=setting['max_leaf_nodes'],
# #             # max_features=setting['max_features'],
# #             bootstrap=setting['bootstrap'],
# #             n_jobs=setting['n_jobs'],
# #             warm_start=setting['warm_start'],
# #             # random_state=setting['random_state'],
# #             verbose=setting['verbose']
# #         )
# #     else:
# #         raise ValueError('Invalid model type.')


clf = RandomForestClassifier(verbose=0, n_jobs=-1, n_estimators=100)

my_model = MyModel(model=clf)

read_config_and_train(model=my_model.model, img_dir='data/images')
print("Done!!!!!!!!!!!!!")
# a = MyModelTool()
# print(a.config)
# print(a.config['model']['random_forest_clf']['warm_start'])
# a.train(model=a.get_model())
# a.train(model=clf)

# TODO: Notice "all" keyword in config. batch_size = x_train_size if self.batch_size == 'all' else self.batch_size
# TODO: x_filenames & y input shuffle
# TODO: M0ake full image file path before call functions.

#########
# Option:
#########
# TODO: Make img array custom processing (e.g. not only use flatten())
