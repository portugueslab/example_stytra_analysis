import json
from pathlib import Path
import pandas as pd
import numpy as np


class Experiment(dict):
    """

    Parameters
    ----------
    path :


    Returns
    -------

    """

    log_mapping = dict(
        stimulus_param_log=["dynamic_log", "stimulus_log", "stimulus_param_log"],
        estimator_log=["estimator_log"],
        behavior_log=["tracking_log", "log", "behavior_log"],
    )

    def __init__(self, path, session_id=None):
        # Prepare path:
        inpath = Path(path)

        if inpath.suffix == ".json":
            self.path = inpath.parent
            session_id = inpath.name.split("_")[0]

        else:
            self.path = Path(path)

            if session_id is None:
                meta_files = list(self.path.glob("*metadata.json"))

                # Load metadata:
                if len(meta_files) == 0:
                    raise FileNotFoundError("No metadata file in specified path!")
                elif len(meta_files) > 1:
                    raise FileNotFoundError(
                        "Multiple metadata files in specified path!"
                    )
                else:
                    session_id = str(meta_files[0].name).split("_")[0]

        self.session_id = session_id
        metadata_file = self.path / (session_id + "_metadata.json")

        source_metadata = json.load(open(metadata_file))

        # Temporary workaround:
        try:
            source_metadata["behavior"] = source_metadata.pop("tracking")
        except KeyError:
            pass

        super().__init__(**source_metadata)

        self._stimulus_param_log = None
        self._behavior_log = None
        self._estimator_log = None

    def _get_log(self, log_name):
        uname = "_" + log_name

        if getattr(self, uname) is None:
            for possible_name in self.log_mapping[log_name]:
                try:
                    logname = next(
                        self.path.glob(self.session_id + "_" + possible_name + ".*")
                    ).name
                    setattr(self, uname, self._load_log(logname))
                    break
                except StopIteration:
                    pass
            else:
                raise ValueError(log_name + " does not exist")

        return getattr(self, uname)

    @property
    def stimulus_param_log(self):
        return self._get_log("stimulus_param_log")

    @property
    def estimator_log(self):
        return self._get_log("estimator_log")

    @property
    def behavior_log(self):
        return self._get_log("behavior_log")

    def _load_log(self, data_name):
        """

        Parameters
        ----------
        data_name :


        Returns
        -------

        """

        file = self.path / data_name
        if file.suffix == ".csv":
            return pd.read_csv(str(file), delimiter=";").drop("Unnamed: 0", axis=1)
        elif file.suffix == ".h5" or file.suffix == ".hdf5":
            return pd.read_hdf(file)
        elif file.suffix == ".feather":
            return pd.read_feather(file)
        elif file.suffix == ".json":
            return pd.read_json(file)
        else:
            raise ValueError(
                str(data_name) + " format is not supported, trying to load " + str(file)
            )

    def stimulus_starts_ends(self):
        starts = np.array([stim["t_start"] for stim in self["stimulus"]["log"]])
        ends = np.array([stim["t_stop"] for stim in self["stimulus"]["log"]])
        return starts, ends

    @staticmethod
    def resample(df_in, resample_sec=0.005):
        """

        Parameters
        ----------
        df_in :
        resample_sec :


        Returns
        -------

        """
        df = df_in.copy()
        t_index = pd.to_timedelta(
            (df["t"].as_matrix() * 10e5).astype(np.uint64), unit="us"
        )
        df.set_index(t_index - t_index[0], inplace=True)
        df = df.resample("{}ms".format(int(resample_sec * 1000))).mean()
        df.index = df.index.total_seconds()
        return df.interpolate().drop("t", axis=1)
    

def calcium_kernel():
    """

    :param indicator: 6f, s or m or 5G
    :param rise: if the rise time of the calcium indicator is taken into account

    :return: kernel function
    """    
    return lambda x: np.exp(-x /  (1.5 / np.log(2)))


def convolve_regressors(regressor, kernel):
    """ Convolves the regressor with a kernel function

    :param regressor: the regressor, or regressor matrix
    :param kernel:
    :return: the convolved kernel
    """
    return np.convolve(regressor, kernel)[0:len(regressor)]


def pearson_regressors(traces, regressors):
    """ Gives the pearson correlation coefficient

    :param traces: the regressors, with time in rows
    :param regressors: the regressors, with time in rows
    :return: the pearson correlation coefficient
    """
    # two versions, depending whether there is one or multiple regressors
    X = traces
    Y = regressors
    numerator = np.dot(X.T, Y) - X.shape[0] * np.mean(X, 0) * np.mean(Y)
    denominator = (X.shape[0] - 1) * np.std(X, 0) * np.std(Y)
    result = numerator / denominator

    return result

def x_y_stackbinning(stack, factor):
    """ Downsample ND stack along last 2 dims (x, y) of a factor.
    No padding implemented for borders. Run 1.3 times faster
    than block_reduce. Could be parallelized.

    :param stack: input, ND array
    :param fact: downsampling factor
    :return: downsampled stack
    """

    # Trim stack before downsampling:
    dims = stack.shape
    trm_dims = dims[:-2] + tuple([(s // factor) * factor for s in dims[-2:]])
    trimmed = stack[[slice(0, s) for s in trm_dims]]

    # reshape and then mean along last 2 dims (may be done more elegantly...)
    binned = trimmed.reshape(trm_dims[:-1] + (int(trm_dims[-1] / factor), factor)).mean(-1)
    binned = (
        binned.swapaxes(-1, -2)
        .reshape(
            trm_dims[:-2]
            + (int(trm_dims[-1] / factor),)
            + (int(trm_dims[-2] / factor), factor)
        )
        .mean(-1)
        .swapaxes(-1, -2)
    )

    return binned.astype(stack.dtype)
