import random
import numpy as np

def sort_ids_by_atime_asc(id_list, atime_list):
    """
    Sort a list of client ids according to their performance, in an descending ordered
    :param id_list: a list of client ids to sort
    :param perf_list: full list of all clients' arrival time
    :return: sorted id_list
    """
    # make use of a map
    cp_map = {}  # client-perf-map
    for id in id_list:
        cp_map[id] = atime_list[id]  # build the map with corresponding perf
    # sort by perf
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1])  # a sorted list of tuples
    sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]  # extract the ids into a list
    return sorted_id_list

id_list=[2,2,5]
atime_list=[0.0 for _ in range(10)]
sort_ids=sort_ids_by_atime_asc(id_list,atime_list)
