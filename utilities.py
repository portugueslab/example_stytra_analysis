import json
from pathlib import Path
import pandas as pd
import numpy as np
from numba import jit


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
    
    
# Functions for bout analysis:
def _fish_renames(i_fish, n_segments):
    return dict(
        {
            "f{:d}_x".format(i_fish): "x",
            "f{:d}_vx".format(i_fish): "vx",
            "f{:d}_y".format(i_fish): "y",
            "f{:d}_vy".format(i_fish): "vy",
            "f{:d}_theta".format(i_fish): "theta",
            "f{:d}_vtheta".format(i_fish): "vtheta",
        },
        **{
            "f{:d}_theta_{:02d}".format(i_fish, i): "theta_{:02d}".format(i)
            for i in range(n_segments)
        }
    )


def _fish_column_names(i_fish, n_segments):
    return [
        "f{:d}_x".format(i_fish),
        "f{:d}_vx".format(i_fish),
        "f{:d}_y".format(i_fish),
        "f{:d}_vy".format(i_fish),
        "f{:d}_theta".format(i_fish),
        "f{:d}_vtheta".format(i_fish),
    ] + ["f{:d}_theta_{:02d}".format(i_fish, i) for i in range(n_segments)]

def _rename_fish(df, i_fish, n_segments):
    return df.filter(["t"] + _fish_column_names(i_fish, n_segments)).rename(
        columns=_fish_renames(i_fish, n_segments)
    )

def _extract_bout(df, s, e, n_segments, i_fish=0, scale=1.0):
    bout = _rename_fish(df.iloc[s:e], i_fish, n_segments)
    # scale to physical coordinates
    dt = (bout.t.values[-1] - bout.t.values[0]) / bout.shape[0]
    bout.iloc[:, 1:5] *= scale
    bout.iloc[:, 2:7:2] /= dt
    return bout

def extract_bouts(
    metadata,
    max_interpolate=2,
    window_size=7,
    recalculate_vel=False,
    scale=None,
    filter_nan=True,
    **kwargs
):
    """ Splits a dataframe with fish tracking into bouts

    :param metadata_file: the path of the metadata file
    :param max_interpolate: number of points to interpolate if surrounded by NaNs in trackign
    :param max_frames: the maximum numbers of frames to process, useful for debugging
    :param threshold: velocity threshold
    :param min_duration: minimal number of frames for a bout
    :param pad_before: number of frames that gets added before
    :param pad_after: number of frames added after
    :return: list of single bout dataframes
    """

    df = metadata.behavior_log

    scale = scale or get_scale_mm(metadata)

    n_fish = get_n_fish(df)
    n_segments = get_n_segments(df)
    dfint = df.interpolate("linear", limit=max_interpolate, limit_area="inside")
    bouts = []
    continuous = []
    for i_fish in range(n_fish):
        if recalculate_vel:
            for thing in ["x", "y", "theta"]:
                dfint["f{}_v{}".format(i_fish, thing)] = np.r_[
                    np.diff(dfint["f{}_{}".format(i_fish, thing)]), 0
                ]

        vel = dfint["f{}_vx".format(i_fish)] ** 2 + dfint["f{}_vy".format(i_fish)] ** 2
        vel = vel.rolling(window=window_size, min_periods=1).median()
        bout_locations, continuity = extract_segments_above_thresh(vel.values, **kwargs)
        all_bouts_fish = [
            _extract_bout(dfint, s, e, n_segments, i_fish, scale)
            for s, e in bout_locations
        ]
        bouts.extend(all_bouts_fish)
        continuous.extend(continuity)

    return bouts, np.array(continuous)

@jit(nopython=True)
def extract_segments_above_thresh(
    vel, threshold=0.1, min_duration=20, pad_before=12, pad_after=25, skip_nan=True
):
    """ Useful for extracing bouts from velocity or vigor

    :param vel:
    :param threshold:
    :param min_duration:
    :param pad_before:
    :param pad_after:
    :return:
    """
    bouts = []
    in_bout = False
    start = 0
    connected = []
    continuity = False
    i = pad_before + 1
    bout_ended = pad_before
    while i < vel.shape[0] - pad_after:
        if np.isnan(vel[i]):
            continuity = False
            if in_bout and skip_nan:
                in_bout = False

        elif i > bout_ended and vel[i - 1] < threshold < vel[i] and not in_bout:
            in_bout = True
            start = i - pad_before

        elif vel[i - 1] > threshold > vel[i] and in_bout:
            in_bout = False
            if i - start > min_duration:
                bouts.append((start, i + pad_after))
                bout_ended = i + pad_after
                if continuity:
                    connected.append(True)
                else:
                    connected.append(False)
            continuity = True

        i += 1

    return bouts, connected

def get_scale_mm(metadata):
    cal_params = metadata["stimulus"]["calibration_params"]
    proj_mat = np.array(cal_params["cam_to_proj"])
    return np.linalg.norm(np.array([1.0, 0.0]) @ proj_mat[:, :2]) * cal_params["mm_px"]

def get_n_segments(df, prefix=True):
    if prefix:

        def _tail_part(s):
            ps = s.split("_")
            if len(ps) == 3:
                return ps[2]
            else:
                return 0

    else:

        def _tail_part(s):
            ps = s.split("_")
            if len(ps) == 2:
                return ps[1]
            else:
                return 0

    tpfn = np.vectorize(_tail_part, otypes=[int])
    return np.max(tpfn(df.columns.values)) + 1


def get_n_fish(df):
    def _fish_part(s):
        ps = s.split("_")
        if len(ps) == 3:
            return ps[0][1:]
        else:
            return 0

    tpfn = np.vectorize(_fish_part, otypes=[int])
    return np.max(tpfn(df.columns.values)) + 1

def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi

def angle_mean(angles, axis=1):
    """Correct calculation of a mean of an array of angles
    """
    return np.arctan2(np.sum(np.sin(angles), axis), np.sum(np.cos(angles), axis))

def rot_mat(theta):
    """The rotation matrix for an angle theta """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

@jit(nopython=True)
def smooth_tail_angles(tail_angles):
    """Smooths out the tau jumps in tail angles, so that the angle between
    tail segments is smoothly changing

    Parameters
    ----------
    tail_angles :
        return:

    Returns
    -------

    """

    tau = 2 * np.pi

    for i in range(1, tail_angles.shape[0]):
        previous = tail_angles[i - 1]
        dist = np.abs(previous - tail_angles[i])
        if np.abs(previous - (tail_angles[i] + tau)) < dist:
            tail_angles[i] += tau
        elif np.abs(previous - (tail_angles[i] - tau)) < dist:
            tail_angles[i] -= tau

    return tail_angles

def normalise_bout(bout):
    dir_init = angle_mean(bout.theta.iloc[0:2], axis=0)
    coord = bout[["x", "y", "theta"]].values
    coord[:, :2] = (coord[:, :2] - coord[:1, :2]) @ rot_mat(dir_init + np.pi)
    coord[:, 2] -= dir_init
    coord[:, 2] = reduce_to_pi(coord[:, 2])
    return coord
    

# Functions for imaging analysis

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
