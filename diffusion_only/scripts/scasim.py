"""
Script that implements the scanpath similarity metric ScaSim by 
Von der Malsburg, Titus, and Shravan Vasishth. 
"What is the scanpath signature of syntactic reanalysis?." 
Journal of Memory and Language 65.2 (2011): 109-127.
"""

from __future__ import annotations
from math import pi, sin, cos, acos
import numpy as np
from typing import List, Tuple, Optional, Any, Union


# only need 0, 2 of s/t due to word index instead of x/y location
def scasim(
    s: List[Tuple[int, int, Union[int, float]]],
    t: List[Tuple[int, int, Union[int, float]]],
    modulator: Optional[float] = 0.83,
    normalize: Optional[str] = None,  # fixations, durations, None
) -> float:
    """
    Calculate the similarity between two scanpaths s and t.
    :param s: scanpath s, consisting of fixation locations (word indices) and fixation durations
    :param t: scanpath t, consisting of fixation locations (word indices) and fixation durations
    :param modulator: modulator specifies how spatial distances between fixations are assessed.  When set to 0, any spatial divergence of two
        compared scanpaths is penalized independently of its degree.  When set to 1, the scanpaths are compared only with respect to their
        temporal patterns.  The default value approximates the sensitivity to spatial distance found in the human visual system.
    :param normalize: if 'fixations', the similarity score is normalized by the number of fixations in the two scanpaths.  If 'durations',
        the similarity score is normalized by the sum of fixation durations in the two scanpaths.  If None, no normalization is applied.

    :return: similarity between scanpaths s and t
    """
    m, n = len(s), len(t)
    d = [list(map(lambda i: 0, range(n + 1))) for _ in range(m + 1)]

    # sum of fixation durations of the two scanpaths
    s_fixdur_sum = sum([fix[2] for fix in s])
    t_fixdur_sum = sum([fix[2] for fix in t])
    # number of fixations in the two scanpaths
    s_nfix = len(s)
    t_nfix = len(t)

    acc = 0
    # sequence alignment
    # loop over fixations in scanpath s:
    for fix_i in range(1, m + 1):
        acc += s[fix_i - 1][2]
        d[fix_i][0] = acc

    # loop over fixations in scanpath t:
    acc = 0
    for fix_j in range(1, n + 1):
        acc += t[fix_j - 1][2]
        d[0][fix_j] = acc


    # Compute similarity:
    for fix_i in range(n):
        for fix_j in range(m):
            # calculating angle between fixation targets:
            slon = s[fix_j][0] / (180 / pi)  # longitude (x-axis)
            tlon = t[fix_i][0] / (180 / pi)
            slat = s[fix_j][1] / (180 / pi)  # latitude (y-axis)
            tlat = t[fix_i][1] / (180 / pi)

            angle = acos(sin(slat) * sin(tlat) + cos(slat) * cos(tlat) * cos(slon - tlon)) * (180 / pi)

            # approximation of cortical magnification:
            mixer = modulator ** angle

            # cost for substitution:
            cost = (
                abs(t[fix_i][2] - s[fix_j][2]) * mixer +
                (t[fix_i][2] + s[fix_j][2]) * (1.0 - mixer)
            )

            # select optimal edit operation
            ops = (
                d[fix_j][fix_i + 1] + s[fix_j][2],
                d[fix_j + 1][fix_i] + t[fix_i][2],
                d[fix_j][fix_i] + cost,
            )

            #mi = which_min(*ops)
            mi = np.argmin(ops)

            d[fix_j + 1][fix_i + 1] = ops[mi]
        
    result = d[-1][-1]
    if normalize == 'fixations':
        result /= (s_nfix + t_nfix)
    elif normalize == 'durations':
        result /= (s_fixdur_sum + t_fixdur_sum)

    return result


def main():

    predicted_sp_ids = [[0, 1, 2, 3, 4, 5, 6, 8, 10, 10, 10, 11], [0, 1, 1, 2, 4, 5, 7, 4, 3, 7, 1, 8], [0, 1, 2, 4, 4, 5, 7, 7, 8, 9]]
    original_sp_ids = [[0, 1, 2, 4, 2, 3, 5, 6, 8, 9, 10, 4, 11], [0, 1, 2, 4, 3, 8], [0, 1, 6, 8, 9]]
    predicted_fix_durs = [[69, 52, 374, 374, 374, 374, 256, 423, 423, 423, 423, 188], [69, 52, 374, 384, 374, 374, 423, 423, 423, 423, 52, 423], [69, 52, 374, 502, 502, 374, 423, 423, 374, 374]]
    original_fix_durs = [[0, 208, 232, 197, 314, 151, 219, 308, 195, 280, 260, 102], [0, 192, 182, 297, 134], [0, 195, 130, 101]]

    # remove last element in each sublist of list for predicted_sp_ids, original_sp_ids, and predicted_fix_durs
    # these are the pad tokens and they are not contained in original_fix_durs
    predicted_sp_ids = [sublist[:-1] for sublist in predicted_sp_ids]
    original_sp_ids = [sublist[:-1] for sublist in original_sp_ids]
    predicted_fix_durs = [sublist[:-1] for sublist in predicted_fix_durs]

    # create dummy y values for original_sp_ids and predicted_sp_ids
    dummy_y_original_sp_ids = [[1] * len(sublist) for sublist in original_sp_ids]
    dummy_y_predicted_sp_ids = [[1] * len(sublist) for sublist in predicted_sp_ids]

    # zip together the predicted_sp_ids and predicted_fix_durs lists as list of list of tuples
    predicted_sp = list(map(lambda x, y, z: list(zip(x, y, z)), predicted_sp_ids, dummy_y_predicted_sp_ids, predicted_fix_durs))
    # zip together the original_sp_ids and original_fix_durs lists as list of list of tuples
    original_sp = list(map(lambda x, y, z: list(zip(x, y, z)), original_sp_ids, dummy_y_original_sp_ids, original_fix_durs))

    s1 = predicted_sp[0]
    t1 = original_sp[0]
    sim1 = scasim(s=s1, t=t1)
    
    s2 = predicted_sp[1]
    t2 = original_sp[1]
    sim2 = scasim(s=s2, t=t2)

    s3 = predicted_sp[2]
    t3 = original_sp[2]
    sim3 = scasim(s=s3, t=t3)

    # normalize by fixations
    sim10 = scasim(s=s1, t=t1, normalize='fixations')
    sim11 = scasim(s=s2, t=t2, normalize='fixations')
    sim12 = scasim(s=s3, t=t3, normalize='fixations')

    # normalize by durations
    sim13 = scasim(s=s1, t=t1, normalize='durations')
    sim14 = scasim(s=s2, t=t2, normalize='durations')
    sim15 = scasim(s=s3, t=t3, normalize='durations')

    print('normalize by fixations')
    print('sim10:', sim10)
    print('sim11:', sim11)
    print('sim12:', sim12)
    print('normalize by durations')
    print('sim13:', sim13)
    print('sim14:', sim14)
    print('sim15:', sim15)



if __name__ == '__main__':
    raise SystemExit(main())
