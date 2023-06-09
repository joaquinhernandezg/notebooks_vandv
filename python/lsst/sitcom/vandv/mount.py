import asyncio
import logging
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from astropy.time import Time
from lsst.sitcom.vandv.efd import create_efd_client

__all__ = [
    "move_mount_in_elevation_steps",
    "slew_identifier",
    "motion_state_slew_identifier"
    
]

log = logging.getLogger(__name__)

async def move_mount_in_elevation_steps(
    mount, target_el, azimuth=0, step_size=0.25, time_sleep=1
):
    """Move the mount from the current elevation angle to the target 
    elevation angle in steps to avoid any issues whe M1M3 and/or M2 are 
    running with the LUT using the mount instead of the inclinometer.

    This function will actually calculate the number of steps using the ceiling
    in order to make sure that we move carefully.

    Parameters
    ----------
    mtmount : Remote
        Mount CSC remote.
    target_el : float
        Target elevation angle in degrees
    azimuth : float
        Azimuth angle in degres (default)
    step_size : float
        Step elevation size in degrees (default: 0.25)
    time_sleep : float
        Sleep time between movements (default: 1)

    Returns
    -------
    azimuth : float
        Current azimuth
    elevation : float
        Current elevation
    """
    current_el = mount.tel_elevation.get().actualPosition
    n_steps = int(np.ceil(np.abs(current_el - target_el) / step_size))

    for el in np.linspace(current_el, target_el, n_steps):
        print(f"Moving elevation to {el:.2f} deg")
        await mount.cmd_moveToTarget.set_start(azimuth=azimuth, elevation=el)
        time.sleep(time_sleep)

    if current_el == target_el:
        el = target_el

    return azimuth, el

def get_starts_command_track_target(
    command_track_target_times, command_track_target_positions, shift=0.1
):
    """takes times and position (azimuth or elevation) 'lsst.sal.MTMount.command_trackTarget'
    and uses them to identify timestamps where the telesope has jumped more than 0.1 deg
    Parameters
    ----------
    command_track_target_times : float
        timestamps from command_track_target data_frame
    command_track_target_positions : float
        positions from command_track_target data_frame, should be azimuth or elevation
    shift : float
        shift threshold in degrees to identify slew starts
    
    Returns
    -------
    slew_start_times : float
        identified slew start times
    """
    lags = command_track_target_positions[1:] - command_track_target_positions[:-1]
    slew_indexes = np.where(abs(lags) > 0.1)[0] + 1  # lags is 1 shorter than positions
    slew_start_times = command_track_target_times[slewIndexes]
    return slew_start_times


def get_slews_edge_detection_telemetry(
    times, velocities, kernel_size=100, height=0.005, vel_thresh=0.05
):
    """
    given timestamps and velocity telemetry from mtMount azmuith or elevation.
    First, we smooth the velocity measurements and convert them to speed.
    Then, an edge detection kernel convolved with the speed data, `starts` are 
    identified by maxima and `stops` by minima values of the convolved data.
    
    Parameters
    ----------
    times : float
        timestamps from mtMount dataframe
    velocities: float
        actualVelocity measurements
    kernel size: int
        size of smoothing and edge detection kernel
    height: float
        minimum height of edge detection peaks (if spurious slews are 
        identified this should be raised)
    vel_thresh: float
        the minimum max velocity of a slew to flag
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    
    
    """
    smooth_kernel = np.ones(kernel_size) / kernel_size
    smoothed_speed = abs(np.convolve(velocities, smooth_kernel, mode="same"))

    edge_kernel = (
        np.concatenate(
            [1 * np.ones(int(kernel_size / 2)), -1 * np.ones(int(kernel_size / 2))]
        )
        / kernel_size
    )
    edge = np.convolve(smoothed_speed, edge_kernel, mode="same")

    starts = times[find_peaks(edge, height=height)[0]]
    stops = times[find_peaks(-1 * edge, height=height)[0]]
    maxv = []
    if (len(starts) ==0) | len(stops) == 0:
        print("No slews identified")
        return [], []
    starts, stops = get_slew_pairs(starts, stops)   
    
    for i, st in enumerate(starts):
        sel_vel = (times >= starts[i]) & (times <= stops[i])
        maxv.append(np.max(np.abs(smoothed_speed[sel_vel])))
    sel_slew = np.array(maxv) > vel_thresh

    starts = starts[sel_slew]
    stops = stops[sel_slew]

    for i, st in enumerate(starts):
        # adjust times to correspond with where the smoothed velocity has reached 0
        sel_starts = (times < st) & (smoothed_speed < 0.01)
        starts[i] = times[sel_starts][np.argmin(np.abs(times[sel_starts] - starts[i]))]

        sel_stops = (times > stops[i]) & (smoothed_speed < 0.01)
        stops[i] = times[sel_stops][np.argmin(np.abs(times[sel_stops] - stops[i]))]

    return starts, stops


def get_slews_command_track_target_and_telemetry(
    command_track_target_data_frame,
    mt_mount_data_frame,
    drive="azmiuth",
    kernel_size=100,
    height=0.005,
    vel_thresh=0.05,
):
    """
    given command track target data identify slew `starts` as large shifts in 
    the azimuth. Then, identify `stops` from the timestamps and velocity 
    telemetry from mtMount azmuith or elevation.
    
    Parameters
    ----------
    command_track_target_data_frame : data_frame
        data_frame from efd
    mt_mount_data_frame: data_frame
        data_frame from efd should be for "azmuith" or "elevation"
    drive: string
        search for slews using "azmuith" or "elevation" drives
    kernel size: int
        size of smoothing and edge detection kernel
    height: float
        minimum height of edge detection peaks (if spurious slews are 
        identified this should be raised)
    vel_thresh: float
        the minimum max velocity of a slew to flag
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    
    """
    
    if drive not in ["azimuth", "elevation"]:
        print("drive must be azimuth elevation")
        return [], []
    command_track_target_positions = command_track_target_data_frame[drive]
    
    # convert command_track_target_times from tai to utc
    command_track_target_times = Time(
        command_track_target_times, format="unix_tai"
    ).unix

    # get starts from command_track_target
    starts = get_starts_command_track_target(
        command_track_target_times, command_track_target_position
    )

    # get stops from telemetry
    mt_times = Time(mt_mount_data_frame["timestamp"], format="unix_tai").unix
    mt_velocities = mt_mount_data_frame["actualVelocity"]
    
    _, stops = get_slews_edge_detection_telemetry(
        mt_times,
        mt_velocities,
        kernel_size=kernel_size,
        height=height,
        vel_thresh=vel_thresh,
    )

    # make sure starts and stops are paired correctly
    starts, stops = get_slew_pairs(starts, stops)

    return starts, stops


def get_slew_from_mtmount(mt_mount_data_frame):
    """
    Givien a dataframe of mtMount telemetry return identified slews.
    Parameters
    ----------
    mt_mount_data_frame: data_frame
        data_frame from efd should contain "timestamp" and "actualVelocity"
        columns
    
    Returns
    -------
    starts : float
        slew start times that match the returned stop times
    stops : float
        slew stop times that match the returned start times
    """

    mt_times = Time(mt_mount_data_frame["timestamp"], format="unix_tai").unix
    mt_velocities = mt_mount_data_frame["actualVelocity"]
    return get_slews_edge_detection_telemetry(mt_times, mt_velocities)


class slew_identifier():
    def __init__(self, method="motion_state"):
        self.method=method
    def get_slew_pairs(self, starts, stops, return_unmatched=False):
        """
        Given vectors of start times and stop times take the longer vector
        and iterate over it. If that is `starts`, for each start time select all stop
        times that are > than the start time and < than the next start time.
        If multiple `stops` are detected select the minimum one. Also,
        the unmatched start and stop times can be returned with `return_unmatched`.
        Parameters
        ----------
        starts : float
            slew start times
        stops : float
            slew stop times
        return_unmatched : bool
            Whether to return stops or starts that cannot be associated
        
        Returns
        -------
        starts : float
            slew start times that match the returned stop times
        stops : float
            slew stop times that match the returned start times
        """
        new_starts = []
        new_stops = []
        unmatched_stops = []
        unmatched_starts = []
        
        if (len(starts) ==0) | len(stops) == 0:
            print("No slews identified")
            return [], []

        if len(stops) <= len(starts):
            for i in range(len(starts)):
                if i == len(starts) - 1:
                    stop_sel = stops > starts[i]
                else:
                    stop_sel = (stops > starts[i]) & (stops < starts[i + 1])
                if stop_sel.sum() == 1:
                    new_stops.append(stops[stop_sel][0])
                    new_starts.append(starts[i])
                if stop_sel.sum() > 1:
                    new_stops.append(np.min(stops[stop_sel]))
                    new_starts.append(starts[i])
                    for j in stops[stop_sel]:
                        if j != np.min(stops[stop_sel]):
                            unmatched_stops.append(j)
                if stop_sel.sum() == 0:
                    unmatched_starts.append(starts[i])

        if len(stops) > len(starts):
            for i in range(len(stops)):
                if i == 0:
                    start_sel = (starts < stops[0]) & (starts > 0)
                else:
                    start_sel = (starts < stops[i]) & (starts > stops[i - 1])
                if start_sel.sum() == 1:
                    new_stops.append(stops[i])
                    new_starts.append(starts[start_sel][0])
                if start_sel.sum() > 1:
                    new_stops.append(stops[i])
                    new_starts.append(np.max(starts[start_sel]))
                    for j in starts[start_sel]:
                        if j != np.max(starts[start_sel]):
                            unmatched_starts.append(j)
                if start_sel.sum() == 0:
                    unmatched_stops.append(stops[i])
        if (len(unmatched_starts) > 1) | (len(unmatched_stops) > 1):
            print("unmatched stops or starts found")
        if return_unmatched:
            return (
                np.array(new_starts),
                np.array(new_stops),
                unmatched_starts,
                unmatched_stops,
            )
        else:
            return np.array(new_starts), np.array(new_stops)

    def combine_az_el(self, ):
        # take identified slew starts and stops in az and el and combine to create master list of slew starts and stops
        return "a"
    
class motion_state_slew_identifier(slew_identifier):
    
    def __init__(self, 
                 start_time, 
                 stop_time,
                 axes = "both", 
                 az_axis_motion_state = None, 
                 az_in_position = None, 
                 el_in_position = None, 
                 el_axis_motion_state = None):
        # datastream start and stop times in iso string format
        self.start_time = start_time
        self.stop_time = stop_time
        self.axes = axes
        self.efd_dict={}
        self.efd_dict["az_axis_motion_state"] = az_axis_motion_state
        self.efd_dict["az_in_position"] = az_in_position
        self.efd_dict["el_axis_motion_state"] = el_axis_motion_state
        self.efd_dict["el_in_position"] = el_in_position
        
        self.merged_timestamp_dict={}
        self.slew_starts={}
        self.slew_stops={}

    async def get_data(self):
        # if data is not passed query new data
        if self.axes in ["azimuth", "az"]:
            if self._check_if_query_needed(self.axes):
                await self._query_efd(self.axes)
        elif self.axes in ["elevation", "el"]:
            if self._check_if_query_needed(self.axes):
                await self._query_efd(self.axes)
        elif self.axes == "both":
            if self._check_if_query_needed(self.axes):
                print("here")
                await self._query_efd(self.axes)
        else:
            raise ValueError(f'axes must be one of: "both", "azimuth", "elevation", "az", "el" instead of {self.axes}')
        
        # check that there is data to identify slews with
        for key in self.efd_dict.keys():
            if self.efd_dict[key] is not None:
                if self.efd_dict[key].empty:
                    raise ValueError(f'{key} has no data, make sure there is data for the specified range')
        
    
    def add_SlewState(self, axis_motion_state):
        """
        From conversation at:https://confluence.lsstcorp.org/display/LTS/23.06.05+Slew+Command+Sequences
        
        state= 0,1 : SlewState="stop"
        state= 4,5 : SlewState="Tracking"
        state= 2   : SlewState="movingPointToPoint"
        state= 3   : what even is jogging
        
        """
        if "snd_timestamp_utc" not in axis_motion_state.columns:
            axis_motion_state["snd_timestamp_utc"]=self.convert_sndStamp_to_utc(
                axis_motion_state["private_sndStamp"]
                )
        
        axis_motion_state["SlewState"]="notSet"
        
        sel_stop=(axis_motion_state["state"]==0)  | (axis_motion_state["state"]==1)
        axis_motion_state.loc[sel_stop,"SlewState"]="Stopped"
        
        sel_tracking=(axis_motion_state["state"]==4)  | (axis_motion_state["state"]==5)
        axis_motion_state.loc[sel_tracking,"SlewState"]="Tracking"
        
        sel_point_to_point=(axis_motion_state["state"]==2)
        axis_motion_state.loc[sel_point_to_point,"SlewState"]="movingPointToPoint"
        
        # check to make sure all SlewStates are set
        
        return axis_motion_state
    
    def run(self,):
        
        
        
        #ready dataframes
        if self.axes in ["azimuth", "az", "both"]:
            self.efd_dict["az_axis_motion_state"] = self.add_SlewState(self.efd_dict["az_axis_motion_state"])
            if "snd_timestamp_utc" not in self.efd_dict["az_in_position"].columns:
                self.efd_dict["az_in_position"]["snd_timestamp_utc"] = self.convert_sndStamp_to_utc(
                    self.efd_dict["az_in_position"]["private_sndStamp"]
                    )
        if self.axes in ["elevation", "el", "both"]:
            self.efd_dict["el_axis_motion_state"] = self.add_SlewState(self.efd_dict["el_axis_motion_state"])
            if "snd_timestamp_utc" not in self.efd_dict["az_in_position"].columns:
                self.efd_dict["el_in_position"]["snd_timestamp_utc"] = self.convert_sndStamp_to_utc(
                    self.efd_dict["el_in_position"]["private_sndStamp"]
                    )
        
        # identify slew starts and stops for each axis
        if self.axes in ["azimuth", "az", "both"]:
            self.get_slews(slew_id_axis="az")
        if self.axes in ["elevation", "el", "both"]:
            self.get_slews(slew_id_axis="el")
        
        
    def convert_sndStamp_to_utc(self, private_sndStamp):
        """given private_sndStamp data return values in utc"""
        return Time(private_sndStamp, format="unix_tai").unix
    
    def get_slews(self, slew_id_axis):
        if slew_id_axis not in ['az', 'el']:
            raise ValueError(f"slew_id_axis must be 'az' or 'el' not: {slew_id_axis}")
        slew_starts=[]
        slew_stops=[]
        current_axis_motion_state="NA"
        previous_axis_motion_state="NA"
        previous_in_position_state="NA"
        current_in_position_state="NA"
        previous_update="NA"
        update="NA"
        
        # Merge dataframes to get all timestamps
        self.merged_timestamp_dict[slew_id_axis]=pd.merge(
            self.efd_dict[f"{slew_id_axis}_axis_motion_state"].loc[:,["snd_timestamp_utc","SlewState"]],
            self.efd_dict[f"{slew_id_axis}_in_position"].loc[:,["snd_timestamp_utc","InPosition"]],
            on="snd_timestamp_utc", 
            how="outer", 
            sort=True, 
            indicator=True)
        
        for i,time in enumerate(self.merged_frame["snd_timestamp_utc"]):
             # check if current state is tracking and previous slew state was stop or did not exist
            previous_update=update
            
            if merged_frame["_merge"][i] == "left_only":
                update="SlewState"
                previous_axis_motion_state=current_axis_motion_state
                current_axis_motion_state=merged_frame["SlewState"][i]
            elif merged_frame["_merge"][i] == "right_only":
                update="InPosition"
                previous_in_position_state=current_in_position_state
                current_in_position_state=merged_frame["InPosition"][i]

            else:
                update="both"
                previous_axis_motion_state=current_axis_motion_state
                current_axis_motion_state=merged_frame["SlewState"][i]
                previous_in_position_state=current_in_position_state
                current_in_position_state=merged_frame["InPosition"][i]
                print("ahhhh")
                #not sure about thisreturn
                continue
            if (update=="SlewState"):
                if ((previous_axis_motion_state=="Stopped") | (previous_axis_motion_state=="NA")) & (current_axis_motion_state=="Tracking"): 
                    # switch from stop to tracking is a slew start
                    self.slew_starts[slew_id_axis].append(time)
                if ((previous_axis_motion_state=="Stopped") & (current_axis_motion_state=="movingPointToPoint")):
                    self.slew_starts[slew_id_axis].append(time)
            if (update=="InPosition"):
                if (current_axis_motion_state=="Tracking"):
                    if (previous_in_position_state==True) & (current_in_position_state==False):
                        # currently Tracking and switch from inPositon==True to False
                        self.slew_starts[slew_id_axis].append(time)
                    elif (previous_in_position_state==False) & (current_in_position_state==True):
                        # currently Tracking and switch from inPositon==False to True
                        self.slew_stops[slew_id_axis].append(time)
                    elif (previous_update=="SlewState") & (previous_in_position_state=="NA") & (current_in_position_state==True):
                        # catch the first slew stop in the dataframe"
                        self.slew_stops[slew_id_axis].append(time)
                    else: 
                        print(f"no idea how we ended up here:{i,previous_axis_motion_state,current_axis_motion_state,previous_in_position_state, current_in_position_state}")
                if (current_axis_motion_state=="Stopped") & (previous_axis_motion_state=="movingPointToPoint"):
                    if (previous_in_position_state==False) & (current_in_position_state==True):
                        # currently stopped after movePointToPoint and switch from inPositon==False to True
                        self.slew_stops[slew_id_axis].append(time)
            
            
    def _check_if_query_needed(self, axes):
        check_list = []
        if axes in ["azimuth", "az", "both"]:
            check_list.append(self.efd_dict["az_axis_motion_state"] is None)
            check_list.append(self.efd_dict["az_in_position"] is None)
        elif axes in ["elevation", "el", "both"]:
            check_list.append(self.efd_dict["el_axis_motion_state"] is None)
            check_list.append(self.efd_dict["el_in_position"] is None)

        if not any(item == False for item in check_list):
            # no data is passed need to do query
            return True
        elif not any(item == True for item in check_list):
            # all data is passed no need for query
            return False
        else:
            raise ValueError("Must pass all dataframes or no dataframes")
        
    # async   def _query_efd(self, axes):
    #             efd_names_dict={
    #                 "az_pos":"lsst.sal.MTMount.logevent_azimuthInPosition",
    #                 "el_pos":"lsst.sal.MTMount.logevent_elevationInPosition",
    #                 "az_motion_state":"lsst.sal.MTMount.logevent_azimuthMotionState",
    #                 "el_motion_state":"lsst.sal.MTMount.logevent_elevationMotionState",
    #             }
    #             client = create_efd_client()
    #             if axes in ["azimuth", "az", "both"]:
    #                 print("here")
    #                 import pdb; pdb.set_trace()
    #                 self.az_axis_motion_state= await client.select_time_series("lsst.sal.MTMount.logevent_azimuthMotionState", 
    #                                                                         ["private_sndStamp", "state"], 
    #                                                                         self.start_time, 
    #                                                                         self.stop_time)
    #                 self.az_in_position= await client.select_time_series("lsst.sal.MTMount.logevent_azimuthInPosition", 
    #                                                                         ["private_sndStamp", "InPosition"], 
    #                                                                         self.start_time, 
    #                                                                         self.stop_time)
    #             if axes in ["elevation", "el", "both"]:
    #                 self.el_axis_motion_state= await client.select_time_series("lsst.sal.MTMount.logevent_elevationMotionState", 
    #                                                                         ["private_sndStamp", "state"], 
    #                                                                         self.start_time, 
    #                                                                         self.stop_time)
    #                 self.el_in_position= await client.select_time_series("lsst.sal.MTMount.logevent_elevationInPosition", 
    #                                                                         ["private_sndStamp", "InPosition"], 
    #                                                                         self.start_time, 
    #                                                                         self.stop_time)
    
    
    
    
