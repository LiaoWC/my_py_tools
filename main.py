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

plt.style.use('seaborn')


###########################################
# === Setting structures ===
class RandomCropSetting:
    def __init__(self,
                 # Specify a size the img will be resize j
                 resize_width_height,
                 use=False,
                 crop_width_height=None
                 ):
        self.use = use
        self.resize_width_height = resize_width_height
        self.crop_width_height = crop_width_height


###########################################


def save_sklearn_model(model, file_path):
    joblib.dump(model, filename=file_path)


def load_sklearn_model(file_path):
    joblib.load(filename=file_path)


class MyModelTool:
    #
    CONFIG_PATH = "config.yaml"
    ATTRS = ['hp', 'pa', 'ma', 'sp', 'cr']
    '''
    Warning: All image will be turned into RGB in default.
    '''

    #
    def __init__(self):
        # Load configure
        self.config = YAML(MyModelTool.CONFIG_PATH).get_data()
        # Get values from configure
        # Input config
        self.input_size = (self.config['input']['input_size']['width'], self.config['input']['input_size']['height'])
        self.input_color_format = self.config['input']['color']  # Image color format
        self.pil_color_mode = self.get_pil_color_mode()
        self.use_edges = self.config['input']['edges']['use_edges']
        # === Read "train" config ===
        # Random crop settings
        self.use_random_crop = self.config['train']['random_crop']['use_random_crop']
        self.random_crop_width, self.random_crop_height = (self.config['train']['random_crop']['resize']['width'],
                                                           self.config['train']['random_crop']['resize']['height'])
        #
        self.epochs = self.config['train']['epoch']
        # TODO: batch_size='all'
        # batch_size = config['train']['batch_size'] if config['train']['batch_size'] != 'all' else
        self.batch_size = self.config['train']['batch_size']

        self.validation = self.config['train']['validation']
        self.k_fold_number = self.config['train']['K_fold_number']
        self.holdout_size = self.config['train']['holdout_validation_size']
        self.use_whose_data = self.config['train']['rater']
        # TODO: Model
        # # Mode config
        # self.model_type = self.config['model']['type']
        # self.model = None  # Initially None

        # Load rating record
        self.df_record = pd.read_csv(self.config['path']['data']['rating_record'])

        # Make each attribute's dataframe (name, image, attribute)
        self.df_attrs = {}
        for attr in MyModelTool.ATTRS:
            self.df_attrs[attr] = self.df_record.filter(['name', 'image', attr], axis=1)

            # Plot how many images each rater has rated

    def plot_how_many_images_each_rater_has_rated(self):
        rater_to_num = self.df_record.groupby('name').size()
        plot_horizontal_bar(rater_to_num.values,
                            rater_to_num.index.values,
                            x_label='Number of images has rated',
                            title='How many images each rater has rated?',
                            show=False
                            )

    # See if there's anything outside the distribution range (0~10)
    def rating_distribution_of_all_attrs(self):
        print("Valid range: 0~10")
        for attr in MyModelTool.ATTRS:
            frequencies, unique, counts = get_frequencies(data=self.df_record[attr].values,
                                                          return_unique_and_counts=True)
            min_uniq, max_uniq = min(unique.tolist()), max(unique.tolist())
            print("Attribute {}: min={}, max={} => {}".format(attr, min_uniq, max_uniq,
                                                              "VALID" if min_uniq >= 0 and max_uniq <= 10 else "INVALID"))
            plt.figure(figsize=(10, 2))
            plt.bar([x[0] for x in frequencies], [x[1] for x in frequencies])
            plt.show()

    # Plot each rater's rating distribution of each attribute
    def plot_each_rater_rating_distribution_of_each_attribute(self):
        for attr in self.ATTRS:
            # Group and collect as lists
            # https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby
            rater_to_attrSet = self.df_attrs[attr].groupby(['name'])[attr].apply(list)
            rater_num = len(rater_to_attrSet)  # Number of raters
            rater_hp_distributions = {}
            for i in range(rater_num):
                (unique_nums, counts) = np.unique(np.array(rater_to_attrSet.values[i]), return_counts=True)
                unique_nums, counts = unique_nums.tolist(), counts.tolist()
                frequencies = {unique_nums[i]: counts[i] for i in range(len(unique_nums))}
                distribution = []
                for j in range(0, 11):  # 0~10
                    distribution.append(0) if j not in frequencies else distribution.append(frequencies[j])
                rater_hp_distributions[rater_to_attrSet.index[i]] = distribution
            df = pd.DataFrame(data=rater_hp_distributions)
            ax = df.plot.bar(stacked=True, figsize=(15, 10), fontsize=20)
            ax.legend(prop={'size': 20})  # Set legend size
            title = "Each rater's rating distribution of {}".format(attr)
            ax.set_title(label=title, fontdict={'fontsize': 30})  # Set title
            plt.savefig(os.path.join(self.config['path']['statistics'], secure_filename(title)))

    #
    def get_file_path(self, file_name):
        return os.path.join(self.config['path']['data']['all_images'], file_name)

    # Get PIL color mode
    def get_pil_color_mode(self):
        if self.input_color_format == 'RGB':
            return 'RGB'
        elif self.input_color_format == 'Grayscale':
            return 'LA'
        else:
            raise ValueError('Invalid input color format.')

    # Input and output are both PIL_Image
    def get_edges(self, pil_img):
        img = pil_img
        return Image.fromarray(
            cv2.Canny(image=np.array(img.convert('RGB')), threshold1=100, threshold2=50))

    # Parameters:
    #     - filename: the img's path
    #     - return_mode: 'PIL_Image', 'array'
    #     - random_crop_setting: dict
    #           "use_random_crop": bool (required)
    #           "resize_"
    #     -
    def get_img_data_from_filename(self,
                                   filename,
                                   return_mode='PIL_Image',
                                   random_crop_setting={''}
                                   ):
        img = None
        if self.use_random_crop == 'true':
            img = random_crop(img=file_path,
                              bbox_size=self.input_size,
                              input_mode='file_path',
                              min_size=self.random_crop_resize,
                              all_possible=False,
                              return_mode='PIL_Image')

        else:
            img = resize_with_zero_padding(img=file_path,
                                           input_mode='file_path',
                                           target_size=self.input_size,
                                           return_mode='PIL_Image')
        # Image color format
        img = img.convert(self.pil_color_mode)
        # Edges
        if self.use_edges == 'true':
            img = self.get_edges(pil_img=img)
        # Put batch data in the current batch
        x_batch_data.append(np.array(img).flatten())

    #
    def single_train(self, model, x_filenames, x_test, y_train, y_test):
        # X_train = x_train_filename
        # X_test_data = x_test_data
        # y_train = y_train_data
        # y_test = ytest_da
        x_train_size = x_filenames.size
        batch_size = x_train_size if self.batch_size == 'all' else self.batch_size
        iterations = x_train_size // batch_size if x_train_size % batch_size == 0 \
            else x_train_size // batch_size + 1

        # Epochs
        last_accuracy_score = None
        last_y_pred = None
        for cur_epoch_num in range(1, self.epochs + 1):
            # Record epoch start time
            start_time = time.time()

            # print('Epoch {}'.format(cur_epoch_num))
            accumulated_x = 0
            # Iteration
            for cur_iteration_num in range(1, iterations + 1):
                print('Epoch {} iteration [{}/{}]'.format(cur_epoch_num, cur_iteration_num, iterations),
                      end='' if cur_iteration_num == iterations else '\r')
                cur_iteration_x_num = min(batch_size, x_train_size - accumulated_x)
                x_batch_data = []
                y_batch_data = []
                for cur_x_number in range(1, cur_iteration_x_num + 1):
                    # Get img data
                    idx = accumulated_x + cur_x_number - 1
                    file_path = os.path.join(self.config['path']['data']['all_images'],
                                             x_filenames[idx])
                    img = None
                    if self.use_random_crop == 'true':
                        img = random_crop(img=file_path,
                                          bbox_size=self.input_size,
                                          input_mode='file_path',
                                          min_size=self.random_crop_resize,
                                          all_possible=False,
                                          return_mode='PIL_Image')

                    else:
                        img = resize_with_zero_padding(img=file_path,
                                                       input_mode='file_path',
                                                       target_size=self.input_size,
                                                       return_mode='PIL_Image')
                    # Image color format
                    img = img.convert(self.pil_color_mode)
                    # Edges
                    if self.use_edges == 'true':
                        img = self.get_edges(pil_img=img)
                    # Put batch data in the current batch
                    x_batch_data.append(np.array(img).flatten())
                    y_batch_data.append(y_train[idx])
                x_batch_data = np.array(x_batch_data)
                y_batch_data = np.array(y_batch_data)
                # Train
                model.fit(x_batch_data, y_batch_data)
                #
                accumulated_x += cur_iteration_x_num
            # Validation
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            last_accuracy_score = accuracy
            last_y_pred = y_pred
            print(" val_acc: {:.3f}".format(last_accuracy_score), end="")

            # Record end time and calculate elapsed time
            end_time = time.time()
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(" elapsed_time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        return last_accuracy_score, last_y_pred

    # Get X_test_data
    def get_x_test_data(self, x_test_filenames):
        x_test_data = []
        for x_test_file in x_test_filenames:
            img = resize_with_zero_padding(img=self.get_file_path(file_name=x_test_file),
                                           input_mode='file_path',
                                           target_size=self.input_size,
                                           return_mode='PIL_Image').convert(
                self.pil_color_mode)
            if self.use_edges == 'true':
                img = self.get_edges(pil_img=img)
            x_test_data.append(np.array(img).flatten())
        x_test_data = np.array(x_test_data)
        return x_test_data

    # "models": list of models
    def k_fold_training(self, model, x_filenames, y_data):
        k_models = [copy.deepcopy(model) for i in range(self.k_fold_number)]
        k_model_accuracies = []

        # K-fold
        # TODO: K <= 1
        # folds = []
        # if K > 1:
        #     pass
        # else:
        #     folds.append([0, np.arange()])
        kf = KFold(n_splits=self.k_fold_number)
        total_accuracy = 0
        total_confusion_matrices = []
        for i, (train_idx, test_idx) in enumerate(kf.split(x_filenames)):
            # TODO K<=1, train_test_split
            # [i, train_idx, test_idx] = fold

            print("^^^ Fold-{} ^^^".format(i + 1))

            # Get fold
            x_train, x_test = x_filenames[train_idx], x_filenames[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            # Get X_test_data
            x_test_data = self.get_x_test_data(x_test_filenames=x_test)

            # Train
            accuracy, y_pred = self.single_train(k_models[i], x_train, x_test_data, y_train, y_test)

            # Record
            k_model_accuracies.append(accuracy)
            total_accuracy += accuracy
            total_confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=np.arange(11)))
            print()

        print("Avg accuracy score: {:.3f}\n".format(total_accuracy / self.k_fold_number))

        # Add all confusion matrices
        cm_after_added = total_confusion_matrices[0]
        for i in range(1, len(total_confusion_matrices)):
            cm_after_added = np.add(cm_after_added, total_confusion_matrices[i])

        # Determine whether to formalize
        if self.config['confusion_matrix']['normalize'] == 'true':
            cm_after_added = cm_after_added / (np.linalg.norm(cm_after_added))

        #
        return cm_after_added, k_models, k_model_accuracies

    #
    def holdout_training(self, model, x_filenames, y_data, test_size=0.2):
        print("Holdout (test_size={})".format(test_size))
        x_filenames_train, x_filenames_test, y_train, y_test = train_test_split(x_filenames, y_data,
                                                                                test_size=test_size)
        last_accuracy_score, last_y_pred = self.single_train(model=model, x_filenames=x_filenames_train,
                                                             x_test=self.get_x_test_data(
                                                                 x_test_filenames=x_filenames_test),
                                                             y_train=y_train, y_test=y_test)
        print()
        return confusion_matrix(y_true=y_test, y_pred=last_y_pred, labels=np.arange(11)), last_accuracy_score

    # Model here must have method "fit"
    def train(self, model):
        #
        print('Epochs: {}'.format(self.epochs))
        print('Batch size: {}'.format(self.batch_size))
        # print('Total iterations: {}'.format(iterations))
        print('Input size(w,h): ({},{})'.format(self.input_size[0], self.input_size[1]))
        print()

        for attr in self.config['attributes']:
            print("===== Attribute {} =====\n".format(attr))

            # Determin use whose data
            df_rater = None
            if self.use_whose_data == 'all':
                df_rater = self.df_attrs[attr]
            else:
                df_rater = self.df_attrs[attr][self.df_attrs[attr]['name'] == self.use_whose_data]

            # Drop zero
            df_rater = df_rater[df_rater[attr] != 0]

            # Make x_filenames and y_data
            x_filenames = [name for name in df_rater['image'].values]
            y_data = [v for v in df_rater[attr].values]
            x_filenames = np.array(x_filenames)
            y_data = np.array(y_data)

            #
            cm = None
            if self.validation == 'K_fold':
                cm, k_models, k_model_accuracies = self.k_fold_training(model=model, x_filenames=x_filenames,
                                                                        y_data=y_data)
                model = k_models[k_model_accuracies.index(min(k_model_accuracies))]
            elif self.validation == 'holdout':
                cm, _ = self.holdout_training(model=model, x_filenames=x_filenames, y_data=y_data,
                                              test_size=self.holdout_size)
            # elif self.validation == 'none':
            #     x_data = self.get_x_test_data(x_test_filenames=x_filenames)
            #     model.fit(x_data, y_data)
            else:
                raise ValueError('Invalid validation configure.')

            # Plot confusion matrix
            if self.validation != 'none':
                plt.rcdefaults()
                font = {'size': 40}
                matplotlib.rc('font', **font)
                plt.figure(figsize=(50, 35))
                cm_description = 'Attribute="{}", epochs={}, batch_size={}, size=(w,h)=({},{}), color="{}"'.format(
                    attr, self.epochs, self.batch_size, self.input_size[0], self.input_size[1], self.input_color_format)
                random_crop_description = 'False' if self.use_random_crop is False \
                    else 'True, min_size(w,h)=({},{})'.format(self.random_crop_resize[0], self.random_crop_resize[1])
                cm_description = cm_description + ', {}'.format(random_crop_description)
                plt.title(cm_description)
                _ = sn.heatmap(pd.DataFrame(data=cm,
                                            index=list(np.arange(11)),
                                            columns=list(np.arange(11))),
                               annot=True,
                               cmap=plt.cm.Blues,
                               fmt='{}'.format(self.config['confusion_matrix']['round_precision'])
                               )
                plt.savefig('statistics/{}'.format(cm_description))
                # plt.show()

    # Model
    def get_model(self):
        if self.model_type == 'random_forest_clf':
            setting = self.config['model']['random_forest_clf']
            return RandomForestClassifier(
                n_estimators=setting['n_estimators'],
                criterion=setting['criterion'],
                max_depth=setting['max_depth'],
                min_samples_split=setting['min_samples_split'],
                min_samples_leaf=setting['min_samples_leaf'],
                # max_leaf_nodes=setting['max_leaf_nodes'],
                # max_features=setting['max_features'],
                bootstrap=setting['bootstrap'],
                n_jobs=setting['n_jobs'],
                warm_start=setting['warm_start'],
                # random_state=setting['random_state'],
                verbose=setting['verbose']
            )
        else:
            raise ValueError('Invalid model type.')


clf = RandomForestClassifier(verbose=0, n_jobs=-1, n_estimators=100)
a = MyModelTool()
# print(a.config)
# print(a.config['model']['random_forest_clf']['warm_start'])
# a.train(model=a.get_model())
a.train(model=clf)
