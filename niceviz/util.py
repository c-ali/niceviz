import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import seaborn as sns


class ScalarDataWrapper:
    '''Wrapper for 2d plots'''

    def __init__(self, list_x, list_y, y_label, filename, smooth_sigma=2):
        self.x = np.array(list_x)
        self.y = np.array(list_y)
        if smooth_sigma > 0:
            self.y = ndimage.gaussian_filter1d(self.y, smooth_sigma)
        self.y_label = y_label
        self.filename = filename

    def plot_show(self):
        self.plot()
        plt.show()

    def plot(self, x_interval=(0, -1),xticklabels = None):
        '''Takes (x,y) tuple of cleaned scalar visdom data and plots it.

                Parameters:
                    x_interval (int, int): limit the x-interval of the plot
                    xticklabels (str[]): optional array containing x tick labels
        '''

        plt.plot(xticklabels[x_interval[0]:x_interval[1]] if xticklabels
                 else self.x[x_interval[0]:x_interval[1]], self.y[x_interval[0]:x_interval[1]])
        plt.xlabel("step")
        plt.ylabel(self.y_label)
        plt.title(self.filename)
        plt.show()

    def size(self):
        return len(self.y)


class Scalar2dDataWrapper:
    '''Wrapper for 3d plots'''

    def __init__(self, list_x, list_y, list_z, cbar_title, filename):
        self.x = np.array(list_x)
        self.y = np.array(list_y)
        self.z = np.array(list_z)
        self.cbar_title = cbar_title
        self.filename = filename

    def plot_show(self):
        self.plot()
        plt.show()

    def plot(self, x_every=10, y_every=2, x_interval=(0, -1), xticklabels=None, xlabel=None, ylabel=None, cbar_title=None, title=None, log=False):
        '''Takes (x,y,z) tuple of cleaned scalar visdom data and plots it in a heatmap.

                Parameters:
                    x_every (int): show only every *-th tick on x-axis
                    y_every (int): show only every *-th tick on y-axis
                    x_interval (int, int): limit the x-interval of the plot
                    xticklabels (str[]): optional array containing x tick labels
                    xlabel (str): alternate x-axis label. default is step
                    ylabel (str): alternate y-axis lable. default is the name of the data
                    log (bool): view data in log-space.
        '''

        yticklabels= [""] * len(self.y)
        yticklabels[1::y_every] = self.y[1::y_every]
        xticklabels_ = [""] * len(self.x)
        xticklabels_[1::x_every] = xticklabels[1::x_every] if xticklabels is not None else self.x[1::x_every]
        xticklabels_ = xticklabels_[x_interval[0]:x_interval[1]]

        if log:
            z = np.log(self.z)
        else:
            z = self.z

        ax = sns.heatmap(np.flipud(z[:, x_interval[0]:x_interval[1]]),
                         xticklabels=xticklabels_, yticklabels=np.flipud(yticklabels))
        ax.collections[0].colorbar.set_label(log*"log " + cbar_title if cbar_title else self.cbar_title )
        plt.xlabel(xlabel if xlabel else "step")
        plt.ylabel(ylabel)
        plt.title(title if title is not None else self.filename)
        plt.show()

    def size(self):
        return len(self.y)


class JsonWrapper:
    def __init__(self, path):
        self.filename = ""
        self.data = {}

        self.__read_json(path)
        self.__extract_scalars()
        self.__clean_data()
        pass

    def __getitem__(self, item):
        return self.data[item]

    def __read_json(self, path):
        '''Reads a visdom json file and returns dict of all values and the filename'''
        if os.path.isfile(path) and path.endswith("json"):
            self.filename = os.path.splitext(os.path.basename(path))[0]
            print("Sucessfully read file: " + self.filename)
            with open(path) as json_file:
                self.data = json.load(json_file)["jsons"]

        else:
            raise ValueError("File not valid")

    def __extract_scalars(self):
        '''Extracts all scalar value plots of a visdom-json'''
        new_dict = {}
        # only keep scalars
        for key, value in self.data.items():
            # remove "scalar_" in name
            if "scalar2d" in key:
                new_dict.update({key[9:]: self.data[key]})
            elif "scalar" in key:
                new_dict.update({key[7:]: self.data[key]})
        self.data = new_dict

    def __clean_data(self):
        '''Removes all unnecessary entries in the dict.
        Returned dict contains name as key and a tuple of lists x[]/y[] as value'''
        for key, value in self.data.items():
            if "z" in self.data[key]["content"]["data"][0]:
                self.data[key] = Scalar2dDataWrapper(self.data[key]["content"]["data"][0]["x"],
                                                     self.data[key]["content"]["data"][0]["y"],
                                                     self.data[key]["content"]["data"][0]["z"], key, self.filename)
            else:
                self.data[key] = ScalarDataWrapper(self.data[key]["content"]["data"][0]["x"],
                                                   self.data[key]["content"]["data"][0]["y"], key, self.filename)


def read_and_preprocess(folder_path):
    '''Reads and preprocesses all jsons in a folder indicated by folder_path.
    Returns a dict where keys are filenames and values are cleaned data-dicts.'''
    d_out = {}

    for filename in os.listdir(folder_path):
        filename_noext = os.path.splitext(filename)[0]
        full_path = os.path.join(folder_path, filename)
        try:
            data = JsonWrapper(full_path)
            d_out.update({filename_noext: data})
        except:
            "File " + full_path + " not valid json."

    return d_out


if __name__ == "__main__":
    # Example usage

    # Read and preprocess
    l = read_and_preprocess("/home/chris/workspace/loss-landscapes/logs_linearize/")

    # Use plot function for a single plot
    depths = [6*i+2 for i in range(3,21,3)]
    depths = [56]
    for depth in depths:
        #plt.plot(l["postrain_lr_0.005_rw_0.003_depth_"+str(depth)][" Proportion of disabled ReLUs per Layer"].z[:,-1], linestyle=' ', marker='x')
        plt.plot(l["postrain_lr_0.005_rw_0.003_depth_"+str(depth)]["Histogram of path lengths"].z[:,::10])

    plt.legend(depths)
    plt.show()
    #l["postrain_lr_0.005_rw_0.003_depth_56"]["Histogram of path lengths"].plot(ylabel="layer", log=False, xlabel="epoch", y_every=2)
    breakpoint()
