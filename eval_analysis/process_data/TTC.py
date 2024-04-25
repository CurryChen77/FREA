#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：TTC.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/2
@Source  : This file is mainly modified from https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision
"""
########################################################################################################
#
# Use function TTC(samples, 'dataframe') or TTC(samples, 'values') to compute two-dimensional Time-To-Collision.
#
# The first input is a pandas dataframe of vehicle pair samples, which should include the following columns.
# -----------------------------------------------------------------------------------
# x_i      :  x coordinate of the ego vehicle (usually assumed to be centroid)      |
# y_i      :  y coordinate of the ego vehicle (usually assumed to be centroid)      |
# vx_i     :  x coordinate of the velocity of the ego vehicle                       |
# vy_i     :  y coordinate of the velocity of the ego vehicle                       |
# hx_i     :  x coordinate of the heading direction of the ego vehicle              |
# hy_i     :  y coordinate of the heading direction of the ego vehicle              |
# length_i :  length of the ego vehicle                                             |
# width_i  :  width of the ego vehicle                                              |
# x_j      :  x coordinate of another vehicle (usually assumed to be centroid)      |
# y_j      :  y coordinate of another vehicle (usually assumed to be centroid)      |
# vx_j     :  x coordinate of the velocity of another vehicle                       |
# vy_j     :  y coordinate of the velocity of another vehicle                       |
# hx_j     :  x coordinate of the heading direction of another vehicle              |
# hy_j     :  y coordinate of the heading direction of another vehicle              |
# length_j :  length of another vehicle                                             |
# width_j  :  width of another vehicle                                              |
# ------------------------------------------------------------------------------------
# The second input allows outputing a dataframe with inputed samples plus a new column named 'TTC', or mere TTC values.
#
# If TTC==np.inf, the ego vehicle and another vehicle will never collide if they keep current speed.
# A negative TTC means the bounding boxes of the ego vehicle and another vehicle are overlapping.
# This is due to approximating the space occupied by a vehicle with a rectangular.
# In other words, TTC<0 in this computation means the collision between the two vehicles almost (or although seldom, already) occurred.
#
# *** Note that mere TTC computation can give an extreme small positive value even when the vehivles are overlapping a bit.
#     In order to improve the accuracy, please use function CurrentD(samples, 'dataframe') or CurrentD(samples, 'values') to further
#     exclude overlapping vehicles.
#
########################## Copyright (c) 2022 Yiru Jiao <y.jiao-1@tudelft.nl> ###########################

# Import
import numpy as np
import warnings

import pandas as pd


# Functions
def line(point0, point1):
    x0, y0 = point0
    x1, y1 = point1
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def intersect(line0, line1):
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    D = a0 * b1 - a1 * b0  # D==0 then two lines overlap
    D[D == 0] = np.nan
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    #    x[np.isnan(x)] = np.inf
    #   y[np.isnan(y)] = np.inf
    return np.array([x, y])


def ison(line_start, line_end, point):
    crossproduct = (point[1] - line_start[1]) * (line_end[0] - line_start[0]) - (point[0] - line_start[0]) * (line_end[1] - line_start[1])
    dotproduct = (point[0] - line_start[0]) * (line_end[0] - line_start[0]) + (point[1] - line_start[1]) * (line_end[1] - line_start[1])
    squaredlength = (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2

    return (np.absolute(crossproduct) <= 1e5) & (dotproduct >= 0) & (dotproduct <= squaredlength)


def dist_p2l(point, line_start, line_end):
    return np.absolute((line_end[0] - line_start[0]) * (line_start[1] - point[1]) - (line_start[0] - point[0]) * (line_end[1] - line_start[1])) / np.sqrt(
        (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2)


def getpoints(samples):
    ## vehicle i
    heading_i = samples[['hx_i', 'hy_i']].values
    perp_heading_i = np.array([-heading_i[:, 1], heading_i[:, 0]]).T
    heading_scale_i = np.tile(np.sqrt(heading_i[:, 0] ** 2 + heading_i[:, 1] ** 2), (2, 1)).T
    length_i = np.tile(samples.length_i.values, (2, 1)).T
    width_i = np.tile(samples.width_i.values, (2, 1)).T

    point_up = samples[['x_i', 'y_i']].values + heading_i / heading_scale_i * length_i / 2
    point_down = samples[['x_i', 'y_i']].values - heading_i / heading_scale_i * length_i / 2
    point_i1 = (point_up + perp_heading_i / heading_scale_i * width_i / 2).T
    point_i2 = (point_up - perp_heading_i / heading_scale_i * width_i / 2).T
    point_i3 = (point_down + perp_heading_i / heading_scale_i * width_i / 2).T
    point_i4 = (point_down - perp_heading_i / heading_scale_i * width_i / 2).T

    ## vehicle j
    heading_j = samples[['hx_j', 'hy_j']].values
    perp_heading_j = np.array([-heading_j[:, 1], heading_j[:, 0]]).T
    heading_scale_j = np.tile(np.sqrt(heading_j[:, 0] ** 2 + heading_j[:, 1] ** 2), (2, 1)).T
    length_j = np.tile(samples.length_j.values, (2, 1)).T
    width_j = np.tile(samples.width_j.values, (2, 1)).T

    point_up = samples[['x_j', 'y_j']].values + heading_j / heading_scale_j * length_j / 2
    point_down = samples[['x_j', 'y_j']].values - heading_j / heading_scale_j * length_j / 2
    point_j1 = (point_up + perp_heading_j / heading_scale_j * width_j / 2).T
    point_j2 = (point_up - perp_heading_j / heading_scale_j * width_j / 2).T
    point_j3 = (point_down + perp_heading_j / heading_scale_j * width_j / 2).T
    point_j4 = (point_down - perp_heading_j / heading_scale_j * width_j / 2).T

    return (point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4)


def CurrentD(samples, toreturn='dataframe'):
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)

        dist_mat = []
        count_i = 0
        for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2], [point_i2, point_i3, point_i1, point_i4]):
            count_j = 0
            for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2], [point_j2, point_j3, point_j1, point_j4]):
                if count_i < 2 and count_j < 2:
                    # Distance from point to point
                    dist_mat.append(np.sqrt((point_i_start[0] - point_j_start[0]) ** 2 + (point_i_start[1] - point_j_start[1]) ** 2))
                    dist_mat.append(np.sqrt((point_i_start[0] - point_j_end[0]) ** 2 + (point_i_start[1] - point_j_end[1]) ** 2))
                    dist_mat.append(np.sqrt((point_i_end[0] - point_j_start[0]) ** 2 + (point_i_end[1] - point_j_start[1]) ** 2))
                    dist_mat.append(np.sqrt((point_i_end[0] - point_j_end[0]) ** 2 + (point_i_end[1] - point_j_end[1]) ** 2))

                # Distance from point to edge
                ist = intersect(line(point_i_start, point_i_start + np.array([-(point_j_start - point_j_end)[1], (point_j_start - point_j_end)[0]])),
                                line(point_j_start, point_j_end))
                ist[:, ~ison(point_j_start, point_j_end, ist)] = np.nan
                dist_mat.append(np.sqrt((ist[0] - point_i_start[0]) ** 2 + (ist[1] - point_i_start[1]) ** 2))

                # Overlapped bounding boxes
                ist = intersect(line(point_i_start, point_i_end), line(point_j_start, point_j_end))
                dist = np.ones(len(samples)) * np.nan
                dist[ison(point_i_start, point_i_end, ist) & ison(point_j_start, point_j_end, ist)] = 0
                dist[np.isnan(ist[0]) & (ison(point_i_start, point_i_end, point_j_start) | ison(point_i_start, point_i_end, point_j_end))] = 0
                dist_mat.append(dist)
                count_j += 1
            count_i += 1

        cdist = np.nanmin(np.array(dist_mat), axis=0)

        if toreturn == 'dataframe':
            samples['CurrentD'] = cdist
            return samples
        elif toreturn == 'values':
            return cdist


def TTC_ij(samples):
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
    direct_v = (samples[['vx_i', 'vy_i']].values - samples[['vx_j', 'vy_j']].values).T

    dist_mat = []
    leaving_mat = []
    for point_line_start in [point_i1, point_i2, point_i3, point_i4]:
        for edge_start, edge_end in zip([point_j1, point_j3, point_j1, point_j2], [point_j2, point_j4, point_j3, point_j4]):
            point_line_end = point_line_start + direct_v
            ### intersection point
            ist = intersect(line(point_line_start, point_line_end), line(edge_start, edge_end))
            ist[:, ~ison(edge_start, edge_end, ist)] = np.nan
            ### distance from point to intersection point
            dist_ist = np.sqrt((ist[0] - point_line_start[0]) ** 2 + (ist[1] - point_line_start[1]) ** 2)
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            leaving = direct_v[0] * (ist[0] - point_line_start[0]) + direct_v[1] * (ist[1] - point_line_start[1])
            leaving[leaving >= 0] = 10
            leaving[leaving < 0] = 1
            leaving_mat.append(leaving)

    dist2overlap = np.array(dist_mat).min(axis=0)
    TTC = dist2overlap / np.sqrt((samples.vx_i - samples.vx_j) ** 2 + (samples.vy_i - samples.vy_j) ** 2)
    leaving = np.nansum(np.array(leaving_mat), axis=0)
    TTC[leaving < 10] = np.inf
    TTC[(leaving > 10) & (leaving % 10 != 0)] = -1

    return TTC


# Computation

def TTC(samples, toreturn='dataframe'):
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        ttc_ij = TTC_ij(samples)
        keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        ttc_ji = TTC_ij(samples.rename(columns=rename_dict))

        if toreturn == 'dataframe':
            samples['TTC'] = np.minimum(ttc_ij, ttc_ji)
            return samples
        elif toreturn == 'values':
            return np.minimum(ttc_ij, ttc_ji)


# Efficiency evaluation
def efficiency(samples, iterations):
    import time
    ts = []
    for _ in range(iterations):
        t = time.time()
        _ = TTC(samples, 'values')
        ts.append(time.time() - t)
    return sum(ts) / iterations


def change_column_name(df):
    df['x'] = df['loc'].apply(lambda x: x[0])
    df['y'] = df['loc'].apply(lambda x: x[1])
    df['vx'] = df['vel'].apply(lambda x: x[0])
    df['vy'] = df['vel'].apply(lambda x: x[1])
    df['hx'] = np.cos(df['yaw'])
    df['hy'] = np.sin(df['yaw'])
    df['length'] = df['extent'].apply(lambda x: x[0])
    df['width'] = df['extent'].apply(lambda x: x[1])
    df.drop(['loc', 'vel', 'yaw', 'extent'], axis=1, inplace=True)
    return df


def get_ego_BV_overlap_traj(trajectory):
    ego_df = change_column_name(pd.DataFrame(trajectory['ego']))
    matched_dataframes = []
    for key in trajectory:
        if key != 'ego':
            vehicle_df = change_column_name(pd.DataFrame(trajectory[key]))
            merged_df = pd.merge(ego_df, vehicle_df, on='time', suffixes=('_i', '_j'))
            if not merged_df.empty:
                matched_dataframes.append(merged_df)

    return matched_dataframes


def get_trajectory_ttc(trajectory):
    ttc_list = []
    # convert overall trajectory to ego <-> BV overlap trajectory
    trajectory_dataframe_list = get_ego_BV_overlap_traj(trajectory)
    # process ttc for each overlap trajectory
    for dataframe in trajectory_dataframe_list:
        ttc_list.extend(TTC(dataframe, 'values'))

    return ttc_list
