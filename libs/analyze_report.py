""" Analyse Global Summary
"""
import json
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

def analyze_summary(fname, metric = 'J_AND_F'):
    METRIC_TXT = {'J': 'J',
                  'F': 'F',
                  'J_AND_F': 'J&F',}

    with open(fname, 'r') as fp:
        summary = json.load(fp)

    print('AUC: \t{:.3f}'.format(summary['auc']))
    th = summary['metric_at_threshold']['threshold']
    met = summary['metric_at_threshold'][metric]
    print('{}@{}: \t{:.3f}'.format(METRIC_TXT[metric], th, met))

    time = summary['curve']['time']
    metric_res = summary['curve'][metric]
    iteration = list(range(len(time)))

    fig = plt.figure(figsize=(6, 8))
    fig.suptitle('[AUC/t: {:.3f}]         [{}@{}: {:.3f}]'.format(summary['auc'],METRIC_TXT[metric],th, met), fontsize=16)
    ax1 = fig.add_subplot(211)
    ax1.plot(time, metric_res)
    ax1.plot(time, metric_res,'b.')
    # ax1.set_title('[AUC/t: {:.3f}]         [J@{}: {:.3f}]'.format(summary['auc'],th, jac) )
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, max(time)])
    ax1.set_xlabel('Accumulated Time (s)')
    ax1.set_ylabel(r'$\mathcal{' + METRIC_TXT[metric] + '}$')
    ax1.axvline(th, c='r')
    ax1.yaxis.grid(True)


    ax2 = fig.add_subplot(212)
    ax2.plot(iteration, metric_res)
    ax2.plot(iteration, metric_res,'b.')
    ax2.set_ylim([0, 1])
    ax2.set_xlim([0, len(time)-1])
    ax2.set_xlabel('Interactions (n)')
    ax2.set_ylabel(r'$\mathcal{' + METRIC_TXT[metric] + '}$')
    ax2.yaxis.grid(True)


    save_dir = os.path.split(fname)[0]+'/summary_graph_{:.3f}.png'.format(metric_res[-1])
    plt.savefig(save_dir)
