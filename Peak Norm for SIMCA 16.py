from filterabc import FilterABC
import numpy as np
import math


# Default editor can be changed in File | Options | Python Editor Command
# For example notepad++ to get better syntax highlight

# see FilterABC for more optional functions

class NNormalization(FilterABC):

    def __init__(self):
        # The name of the transform.
        self.name = r'N2 Normalization'

        # prefix to set before transformed variable names if generates_new_variables  is False
        self.prefix = r'N2Pypi'

        # False if the transform functions returns the same variables and the same number of variables as the input.r
        # the variable names will be kept plus the prefix in the new datasetr
        # True if the transform functions can changes the number of variables, get_variable_name will be calledr
        self.generates_new_variables = False

    """
    Required
    The function that performs the actual transform.

    Return value is the filtered data
    data_collection:  the input, an object of type umetrics.simca.DataCollection
    options:          one of the values returned by the "setup" or "settings" functions. If both are implemented it will be the values from "setup"
                      see the functions setup and settings.
    predicting: False when creating the dataset for the first time and True when creating predictions for a model.
    The function should return a matrix with transformed data for example a list of lists [][] or a UmPyMat with the same size as the input matrix (the number of columns can be different if you return true in generates_new_variables)
    """

    def transform(self, data_collection, options, predicting):
        data = data_collection.get_data()
        var = data_collection.get_var_names()

        var = [int(i) for i in var]
        var = np.array(var)

        N2point1 = 2320
        N2point2 = 2340
        bgWidth2 = 20

        filtered_data = [0] * len(data)

        for count, row in enumerate(data):

            idx1 = int((np.abs(var - N2point1)).argmin())
            idx2 = int((np.abs(var - N2point2)).argmin())

            bg1 = np.mean(row[idx1 - bgWidth2:idx1])
            bg2 = np.mean(row[idx2:idx2 + bgWidth2])

            y2 = np.insert(row, idx1, bg1)
            y2 = np.delete(y2, (idx1 + 1), axis=0)

            y2 = np.insert(y2, idx2, bg2)
            y2 = np.delete(y2, (idx2 + 1), axis=0)

            interpolant = np.interp([var[idx1], var[idx2]], var, y2)

            coefficients = np.polyfit([var[idx1], var[idx2]], interpolant, 1)

            N2area = np.sum(row[int(idx1):int(idx2)] - (var[idx1:idx2] * coefficients[0] + coefficients[1]))

            filtered_data[count] = [x / N2area for x in row]

        return filtered_data

    """
    Optional
       setup is called when the filter is created so the script can setup custom options,
       the options are then saved in the project and passed as arguments to the transform function both for predictions and for workset.
       the return values must either be 'bytes' objects or be possible to pickle.
       data_collection:  the input, an object of type umetrics.simca.DataCollection
       xml_settings : the string returned by settings and then modified by the filter wizard in SIMCA, see settings(...)
       ex;
          meanrow1 = float(sum(data[0])) / max(len(data[0])]
          return meanrow1
    """

    """
    def setup(self, data_collection, xml_settings = ""):
        return xml_settings

    """

    """
    Optional
    if generates_new_variables is True, SIMCA will call this function to enable setting a custom name for each column.
    options are the same values as returned by the setup function.
    column is the column index 0, 1, .....
    Return the new name.
    """

    """
    def get_variable_name(self, options, column) :
        return 'Filtered_' + str(column)
    """

    """
    Optional
    Return a html formatted string with information of the performed transform
    options, custom options returned by the "setup" or "settings" function, the setup function is called when the filter is created so the script can setup options, then the options is used in the transform function both for predictions and for workset.
    """

    """
    def summary(self,options) :
        # for example return '<h3>This is my custom filter</h3><br/><p>Here I can format my summary using HTML or leave empty to instead show the script.</p>'
        return ""
    """
