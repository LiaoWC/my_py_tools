import matplotlib.pyplot as plt
import numpy as np


# === Plot horizontal bar ===


# Input: array like
def plot_horizontal_bar(y_values, y_indices, x_label=None, title=None, show=True):
    fig, ax = plt.subplots()
    y_ticks = np.arange(len(y_indices))
    ax.barh(y=y_ticks, width=y_values, align='center')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_indices)
    ax.invert_yaxis()
    if x_label:
        ax.set_xlabel(x_label)
    if title:
        ax.set_title(title)
    if show:
        plt.show()


# Practicing:
'''
# Plot how many images each rater has rated
def plot_how_many_images_each_rater_has_rated(df_record):
    # rater_to_num = df_attrs['hp'].groupby('name').size()
    rater_to_num = df_record.groupby('name').size()
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y=np.arange(len(rater_to_num.index.values)), width=rater_to_num.values, align='center')
    ax.set_yticks(np.arange(len(rater_to_num.index.values)))
    ax.set_yticklabels(rater_to_num.index.values)
    ax.invert_yaxis()
    ax.set_xlabel('Number of images has rated')
    ax.set_title('How many images each rater has rated?')
    plt.show()

plot_how_many_images_each_rater_has_rated(df_record)
'''
