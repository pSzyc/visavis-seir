import numpy as np
import csv
from analyze_tracking import generate_dataset
import click
from multiprocessing import Pool

def get_results(df):
    df_results = df.loc[:, ['ext', 'chaos']]
    df_results['Position'] = df_results[['ext', 'chaos']].min(axis=1)
    df_results['Event'] = df_results.idxmin(axis=1)
    mean_steps = df_results['Position'].mean()
    events = df_results['Event'].value_counts().reindex(['ext', 'chaos']).fillna(0).astype(int)
    return events, mean_steps

def l_value(n_event, mean_steps, n, lenght, events):
    l = n_event / (n_event * mean_steps + (n - n_event) * lenght)
    l_chaos = events['chaos'] / n_event * l
    l_ext = events['ext'] / n_event * l
    return l, l_chaos, l_ext

def get_propensities(lenght, n_sim, width):
    df = generate_dataset([], n_sim, channel_width=width, channel_length=lenght)

    # Extinction
    df['ext'] = np.nan
    ext_mask = (df['fate'] == 'failure')
    df.loc[ext_mask, 'ext'] = df.loc[ext_mask, 'track_end_position']
    # Choas
    df['chaos'] = df['significant_split_position']

    events, mean_steps = get_results(df)
    n_events = events.sum()
    l, l_chaos, l_ext = l_value(n_events, mean_steps, n_sim, lenght, events)
    return width, lenght, l, l_chaos, l_ext

def to_csv(results, result_file):
    with open(result_file,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['width', 'lenght', 'l', 'l_chaos', 'l_ext'])
        for row in results:
            csv_out.writerow(row)

@click.command()
@click.argument('n_sim', type=int)
@click.argument('w_start', type=int)
@click.argument('w_end', type=int)
@click.argument('results_file', type=str, default='lambda.csv')
@click.argument('lenght', type=int, default=300)
@click.argument('n_workers', type=int, default=5)
def simulate(n_sim, w_start, w_end, results_file, lenght, n_workers):
    results = []

    with Pool(n_workers) as pool:
            results = pool.starmap(get_propensities, [(lenght, n_sim, w)  for w in range(w_start, w_end)])  
    to_csv(results, results_file)

if __name__ == '__main__':
    simulate()