#!/usr/bin/env python3

import argparse
import fmqa
import math
import os
import numpy as np
import random
import re
import sys
import tempfile
import time

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def hour_min_sec():
    tm = time.localtime()
    return (tm.tm_hour, tm.tm_min, tm.tm_sec)

# use this instead of print to log on stderr
def log(*args, **kwargs):
    hms = hour_min_sec()
    print('%02d:%02d:%02d ' % hms, file=sys.stderr, end='')
    return print(*args, **kwargs, file=sys.stderr)

# transform a bitstring into a list of 0/1
def zero_ones_from_bitstring(bs):
    res = []
    for x in bs:
        i = int(x)
        assert(i == 0 or i == 1)
        res.append(i)
    return res

# line format is supposed to be:
# ^<depvar:float>,<bitstring:binary_string>\n'
def parse_line(line):
    split = line.split(',')
    depvar = float(split[0])
    bitstring = split[1]
    bits = zero_ones_from_bitstring(bitstring)
    return (bits, depvar)

def lines_of_file(fn):
    with open(fn) as input:
        return list(map(str.strip, input.readlines()))

# no shuffling of the lines here
def train_test_split(train_portion, lines):
    n = len(lines)
    train_n = math.ceil(train_portion * n)
    test_n = n - train_n
    log('train/test: %d/%d' % (train_n, test_n))
    train = lines[0:train_n]
    test = lines[train_n:]
    assert(len(train) + len(test) == n)
    return (train, test)

# train a FMQA model
def train_regr(X_train, y_train):
    xs = np.array(X_train)
    ys = np.array(y_train)
    model = fmqa.FMBQM.from_data(xs, ys)
    # this would be if we need to update a trained model
    # with new data points: model.train(xs, ys)
    return model

def test_regr(trained_model, X_test):
    xs = np.array(X_test)
    return trained_model.predict(xs)

def compute_r2_rmse(actual, preds):
    r2 = r2_score(actual, preds)
    rmse = mean_squared_error(actual, preds, squared=False)
    return (r2, rmse)

def unzip(l):
    xs = []
    ys = []
    for x, y in l:
        xs.append(x)
        ys.append(y)
    return (xs, ys)

# (120, 1) -> [120, 60, 30, 15, 7, 3, 1]
def exponential_scan_down(start, end):
    res = []
    n = start
    while n > end:
        res.append(n)
        n = int(n / 2)
    res.append(end)
    return res

def list_take_drop(l, n):
    took = l[0:n]
    dropped = l[n:]
    return (took, dropped)

# cut list into several lists; for k-folds cross validation
def list_split(l, n):
    x = float(len(l))
    test_n = math.ceil(x / float(n))
    # log('test_n: %d' % test_n)
    res = []
    taken = []
    rest = l
    for _i in range(0, n):
        curr_test, curr_rest = list_take_drop(rest, test_n)
        curr_train = taken + curr_rest
        res.append((curr_train, curr_test))
        taken = taken + curr_test
        rest = curr_rest
    return res

def append_np_array_to_list(a, l):
    for x in a:
        l.append(x)
    return l

def protect_underscores(s):
    return re.sub("_","\_",s)

def regr_plot(title, r2, actual, preds):
    x_min = np.min(actual)
    x_max = np.max(actual)
    y_min = np.min(preds)
    y_max = np.max(preds)
    xy_min = min(x_min, y_min)
    xy_max = max(x_max, y_max)
    data_fn = tempfile.mktemp(prefix='fmqa_regr_data_', suffix='.txt')
    with open(data_fn, 'w') as out:
        for x, y in zip(actual, preds):
            print('%f %f' % (x, y), file=out)
    log('plot data: %s' % data_fn)
    plot_fn = tempfile.mktemp(prefix='fmqa_regr_plot_', suffix='.gpl')
    with open(plot_fn, 'w') as out:
        print("set xlabel 'actual'", file=out)
        print("set ylabel 'predicted'", file=out)
        print("set xtics out nomirror", file=out)
        print("set ytics out nomirror", file=out)
        print("set xrange [%f:%f]" % (xy_min, xy_max), file=out)
        print("set yrange [%f:%f]" % (xy_min, xy_max), file=out)
        print("set key left", file=out)
        print("set size square", file=out)
        print("set title '%s'" % protect_underscores(title), file=out)
        print("g(x) = x", file=out)
        print("f(x) = a*x + b", file=out)
        print("fit f(x) '%s' u 1:2 via a, b" % data_fn, file=out)
        print("plot g(x) t 'perfect' lc rgb 'black', \\", file=out)
        print("'%s' using 1:2 not, \\" % data_fn, file=out)
        print("f(x) t 'fit'", file=out)
    log('plot file: %s' % plot_fn)
    cmd = "gnuplot --persist %s" % plot_fn
    os.system(cmd)

if __name__ == '__main__':
    before = time.time()
    # CLI options parsing
    parser = argparse.ArgumentParser(description = 'train/use a RFR model')
    parser.add_argument('-i',
                        metavar = '<filename>', type = str,
                        dest = 'input_fn',
                        help = 'input data file')
    parser.add_argument('-o',
                        metavar = '<filename>', type = str,
                        dest = 'output_fn',
                        default = '',
                        help = 'predictions output file')
    parser.add_argument('--save',
                        metavar = '<filename>', type = str,
                        dest = 'model_output_fn',
                        default = '',
                        help = 'trained model output file')
    parser.add_argument('--load',
                        metavar = '<filename>', type = str,
                        dest = 'model_input_fn',
                        default = '',
                        help = 'trained model input file')
    parser.add_argument('-s',
                        metavar = '<int>', type = int,
                        dest = 'rng_seed',
                        default = -1,
                        help = 'RNG seed')
    parser.add_argument('-np',
                        metavar = '<int>', type = int,
                        dest = 'nprocs',
                        default = 1,
                        help = 'max number of processes')
    parser.add_argument('-p',
                        metavar = '<float>', type = float,
                        dest = 'train_p',
                        default = 0.8,
                        help = 'training set proportion')
    parser.add_argument('--NxCV',
                        metavar = '<int>', type = int,
                        dest = 'cv_folds',
                        default = 1,
                        help = 'number of cross validation folds')
    # parse CLI ---------------------------------------------------------
    if len(sys.argv) == 1:
        # user has no clue of what to do -> usage
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    input_fn = args.input_fn
    cv_folds = args.cv_folds
    train_fraction = args.train_p
    rng_seed = args.rng_seed
    if rng_seed != -1:
        # only if the user asked for it, make experiments repeatable
        random.seed(rng_seed)
    # -------------------------------------------------------------------
    # load full dataset
    x_list = []
    y_list = []
    n = 0
    for stripped in lines_of_file(input_fn):
        bits, depvar = parse_line(stripped)
        x_list.append(bits)
        y_list.append(depvar)
        n += 1
    x_dim = len(x_list[0])
    full_dataset = list(zip(x_list, y_list))
    random.shuffle(full_dataset)
    log('(lines, cols): %d %d' % (n, x_dim))
    # train and test on the training set
    # (not recommended, just basic check / optimist baseline)
    trained_model = train_regr(x_list, y_list)
    train_preds = test_regr(trained_model, x_list)
    train_r2, train_rmse = compute_r2_rmse(y_list, train_preds)
    log('(R2, RMSE)_train: %.3f %.3f' % (train_r2, train_rmse))
    if cv_folds <= 1:
        log('train_fraction: %.2f' % train_fraction)
        train, test = train_test_split(train_fraction, full_dataset)
        train_X, train_y = unzip(train)
        test_X, test_y = unzip(test)
        model = train_regr(train_X, train_y)
        test_preds = test_regr(model, test_X)
        r2, rmse = compute_r2_rmse(test_y, test_preds)
        log('(R2, RMSE)_test: %.3f %.3f' % (r2, rmse))
        title = 'FMQA p=%.2f R2=%.3f RMSE=%.3f fn=%s' % \
            (train_fraction, r2, rmse, input_fn)
        regr_plot(title, r2, test_y, test_preds)
    else: # cv_folds > 1
        log('NxCV: %d' % cv_folds)
        truth = []
        estimate = []
        for (fold_i, (train, test)) in enumerate(list_split(full_dataset, cv_folds)):
            train_X, train_y = unzip(train)
            test_X, test_y = unzip(test)
            append_np_array_to_list(test_y, truth)
            model = train_regr(train_X, train_y)
            test_preds = test_regr(model, test_X)
            append_np_array_to_list(test_preds, estimate)
            r2, rmse = compute_r2_rmse(test_y, test_preds)
            log('fold %d (R2, RMSE)_test: %.3f %.3f' % (fold_i, r2, rmse))
        assert(len(truth) == len(estimate))
        log('len(truth) == len(estimate): %d' % len(truth))
        actual_a = np.array(truth)
        preds_a = np.array(estimate)
        r2, rmse = compute_r2_rmse(actual_a, preds_a)
        log('global (R2, RMSE)_test: %.3f %.3f' % (r2, rmse))
        title = 'FMQA k=%d R2=%.3f RMSE=%.3f fn=%s' % \
            (cv_folds, r2, rmse, input_fn)
        regr_plot(title, r2, truth, estimate)

# FBR: save model to file, load model from file

# FBR: should we be allowed to optimize factorization_size? looks like a hyper parameter
