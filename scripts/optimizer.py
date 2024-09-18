# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
import numpy as np
import pandas as pd
from scipy.special import logsumexp, erfinv
from matplotlib import pyplot as plt
from warnings import warn



# --- scan-based approach ---

def get_optimum_from_scan(scan_results: pd.DataFrame, field, required_wls=None, return_errors=False):
    result_parts = []
    search_better_around = {}

    for it, ((channel_width, channel_length), data) in enumerate(scan_results.reset_index().groupby(['channel_width', 'channel_length'])):

        smoothed_data = data[field].rolling(3, center=True).mean()
        optimal_interval_idx = smoothed_data.argmax()
        if not 1 < optimal_interval_idx < len(data[field]) - 2:
            if not 0 < optimal_interval_idx < len(data[field]) - 1:
                print(data)
                raise ValueError(f"Problems with finding maximum for {channel_width = }, {channel_length = }: found {optimal_interval_idx=}")
            is_on_lower_edge = optimal_interval_idx <= 1
            warn(f"Maximum found at {data['interval'].iloc[optimal_interval_idx]}, which is the {'lower' if is_on_lower_edge else 'upper'} edge of the scan range for {channel_width = }, {channel_length = }")
            search_better_around.update({(channel_width, channel_length): 1 - 2 * is_on_lower_edge})
        optimal_interval = data['interval'].iloc[optimal_interval_idx] 
        max_value = data[field].iloc[optimal_interval_idx]

        result_part = {
            'channel_width': channel_width,
            'channel_length': channel_length,
            'optimal_interval': optimal_interval,
            'max_value': max_value,
        }
        result_parts.append(result_part)

    result = pd.DataFrame(
        result_parts, 
        columns=['channel_width', 'channel_length', 'optimal_interval', 'max_value'],
        ).astype({
            'channel_width': int,
            'channel_length': int,
            'optimal_interval': float,
            'max_value': float,
            }).set_index(['channel_width', 'channel_length'])
    if required_wls:
        result = result.reindex(required_wls)
    if return_errors:
        return result, search_better_around
    return result


# --- search-based approach ---


def find_maximum(fn, x0, xmin=None, xmax=None, prob_tol=1e-2, x_tol=1, expansion_ratio=1.6, budget=250, x0_confidence=20, DEBUG=False, *args, **kwargs):


    if DEBUG:
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1, 1, figsize=(5,10))
        fig3, ax3 = plt.subplots(1, 1)
        fig4, ax4 = plt.subplots(1, 1)

    logprob_tol = np.log(prob_tol)


    count = 0


    def add(x, idx):
        val = fn(x, *args, **kwargs)
        nonlocal xx
        nonlocal vals
        xx = np.insert(xx, idx, x)
        vals = np.insert(vals, idx, val)
        nonlocal count
        if DEBUG:
            print('add', x)
            ax1.annotate(str(count), (x, val))
            ax1.scatter(x, val)
        count +=1 
    
    def extend_left():
        if DEBUG:
            print('extend_left')
        add(xx[0] - expansion_ratio * (xx[1] - xx[0]), 0)

    def extend_right():
        if DEBUG:
            print('extend_right')
        add(xx[-1] + expansion_ratio * (xx[-1] - xx[-2]), len(xx))

    def drop_left():
        nonlocal xx
        nonlocal vals
        if DEBUG:
            print('drop_left', xx[0])
        xx = np.delete(xx, 0)
        vals = np.delete(vals, 0)

    def drop_right():
        nonlocal xx
        nonlocal vals
        if DEBUG:
            print('drop_right', xx[-1])
        xx = np.delete(xx, -1)
        vals = np.delete(vals, -1)


    xstep_start = (xmax-xmin) / x0_confidence

    xx = np.array([])
    vals = np.array([])

    add(x0, 0)
    add(x0 - xstep_start, 0)
    add(x0 + xstep_start, 2)

    for t in range(budget):

        if DEBUG:
            print('     ', t)
            ax2.scatter(xx, -t * np.ones_like(xx), s=1)
            ax3.scatter(t, len(xx), color='k')
            ax3.scatter(t, xx[-1] - xx[0], color='red')
            ax4.scatter(t, xx[len(xx)//2], color='red')

        vals_diff = np.diff(vals)
        vals_diff_sign = 1 * (vals_diff > 0) - (vals_diff < 0)

        s = ''
        signmap = lambda sign: (">" if sign < 0 else "<" if sign > 0 else "=")
        for x, sign in zip(np.exp(xx), vals_diff_sign):
            s += f" {x:.0f} " + signmap(sign)
        s += f" {np.exp(xx[-1]):.0f}"
        print(s)


        index = np.arange(len(vals_diff_sign))

        logprob_ith_greatest = np.array([(vals_diff_sign * (1 * (index < i) - (index >= i))).sum() for i in range(len(vals))]) * np.log(2)
        
        logprob_left_not_enough = logprob_ith_greatest[0]  - logsumexp(logprob_ith_greatest)
        logprob_right_not_enough = logprob_ith_greatest[-1]  - logsumexp(logprob_ith_greatest)

        # print(xx, vals)
        # print(vals_diff_sign)
        # print(logprob_ith_greatest)

        if logprob_left_not_enough > logprob_tol or logprob_right_not_enough > logprob_tol:
            if DEBUG:
                print('logprobs', logprob_left_not_enough, logprob_right_not_enough, logprob_tol)
            if logprob_left_not_enough > logprob_right_not_enough:
                extend_left()
            else: 
                extend_right()

        else:
            if xx[-1] - xx[0] <= x_tol:
                print('CONVERGED!')
                print(f'required {count} evaluations' )
                print(xx)
                break

            logprob_left_dropping_one_not_enough = logprob_ith_greatest[1]  - logsumexp(logprob_ith_greatest[1:])
            logprob_right_dropping_one_not_enough = logprob_ith_greatest[-2]  - logsumexp(logprob_ith_greatest[:-1])

            if logprob_left_dropping_one_not_enough <= logprob_tol or logprob_right_dropping_one_not_enough <= logprob_tol:
                # print('logprobs', logprob_left_not_enough, logprob_right_not_enough, logprob_tol)
                # print('logprobs_dropping_one', logprob_left_dropping_one_not_enough, logprob_right_dropping_one_not_enough, logprob_tol)

                if logprob_left_dropping_one_not_enough <= logprob_right_dropping_one_not_enough:
                    drop_left()
                else:
                    drop_right()
        
            else:

                if DEBUG:
                    print('interval_length', xx[-1] - xx[0], x_tol)
                center = (xx[-1] + xx[0]) / 2
                ranks = np.arange(1,len(xx))
                interval_penalty = np.log(np.diff(xx))  - erfinv(2 * ranks/len(xx) -1)**2 /.25 #(2/len(xx) * (np.arange(len(xx)-1)**2 + np.arange(len(xx)-1, 0, -1)**2))
                if DEBUG:
                    print(interval_penalty)
                interval_to_divide = interval_penalty.argmax()
                add((xx[interval_to_divide] + xx[interval_to_divide+1])/2, interval_to_divide+1)

    if DEBUG:
        print(xx, vals)
    return xx[len(xx)//2], vals[len(xx)//2]
    


