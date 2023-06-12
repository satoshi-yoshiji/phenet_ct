import optparse
import sys
import copy

import linecache
import os
import tracemalloc


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def bail(message):
    sys.stderr.write("%s\n" % (message))
    sys.exit(1)


valid_opt_models = ['fmin_powell', 'fmin', 'fmin_l_bfgs_b', 'fmin_ncg', 'fmin_cg']

usage = "usage: multi_fit.py [train|classify|phenotype] --config-file <file1> --config-file <fileN> [options]"
parser = optparse.OptionParser(usage)
parser.add_option("", "--config-file", action='append', default=[])
parser.add_option("", "--output-file", default=None)
parser.add_option("", "--append-to-file", default=None)
parser.add_option("", "--append-to-file-id-col", type="int", default=1)
parser.add_option("", "--chunk", type="int", default=None)  # specific chunk to classify; use with --num-chunks"
parser.add_option("", "--num-chunks", type="int", default=None)  # number of chunks to classify; use with --chunk"
parser.add_option("", "--warnings-file", default=None)
parser.add_option("", "--debug-level", type="int", default=1)  # 0=no log, 1=info, 2=debug, 3=trace
parser.add_option("", "--missing", action='append', default=['', 'NA'])  # values to be considered missing
parser.add_option("", "--delim", default="\t")  # delimiter to parse files
parser.add_option("", "--var-id-file", default=None)  # file with list of variant ids to use
parser.add_option("", "--min-traits", type="int",
                  default=2)  # minimum number of non-missing traits required for variants to be used in training
parser.add_option("", "--mcmc-samp-iter", type="int",
                  default=1000)  # number of samplings each iteration of training algorithm to estimate expected values
parser.add_option("", "--mcmc-samp-burn", type="int",
                  default=100)  # number of samplings each iteration of training algorithm that are thrown away as burn-in
parser.add_option("", "--mcmc-samp-thin", type="int", default=10)  # fraction of samplings each iteration to keep
parser.add_option("", "--train-eps", type="float",
                  default=1e-3)  # difference in log-likelihoods at which to terminate training
parser.add_option("", "--max-train-it", type="int",
                  default=100)  # maximum number of training iterations to use, even if convergence is not reached
parser.add_option("", "--no-normalize", action="store_false", dest="normalize",
                  default="True")  # maximum number of training iterations to use, even if convergence is not reached
parser.add_option("", "--map-max-ests", type="int", default=1000)  # maximum number of times we'll estimate MAP
parser.add_option("", "--map-times-seen-break", type="int", default=10)  # average MAP estimates rather than take mean
parser.add_option("", "--map-opt-fun", default=valid_opt_models[0])  # function used for optimization in MAP
parser.add_option("", "--use-normapprox", action='store_true')  # Use NormalApproximation for MAP estimates
parser.add_option("", "--pymc3", action='store_true')  # Run with pymc3
parser.add_option("", "--trace-out-file", default=None)  # Save location for traces
parser.add_option("", "--tmp-dir", default=None)  # where to write tmp files to

(options, args) = parser.parse_args()

if len(args) < 1:
    bail(usage)

mode = args[0]

train = False
classify = False
phenotype = False
if mode == "train":
    train = True
elif mode == "classify":
    classify = True;
elif mode == "phenotype":
    phenotype = True;
else:
    bail("Unrecognized mode %s" % mode)

if options.append_to_file_id_col < 1:
    bail("--append-to-file-id-col must be 1 or greater")

# set up warnings
warnings_fh = None
if options.warnings_file is not None:
    warnings_fh = open(options.warnings_file, 'w')
else:
    warnings_fh = sys.stderr


# if options.pymc3:
#    if options.tmp_dir is not None:
#        import tempfile
#        tmp_dir = tempfile.mkdtemp(dir=options.tmp_dir)
#        import theano
#        theano.config.compiledir

def warn(message):
    if warnings_fh is not None:
        warnings_fh.write("%s\n" % message)
        warnings_fh.flush()


NONE = 0
INFO = 1
DEBUG = 2
TRACE = 3


def log(message, level=INFO):
    if level <= options.debug_level:
        sys.stderr.write("%s\n" % message)
        sys.stderr.flush()


# first read config file
if len(options.config_file) == 0:
    bail(usage)

traits = {}
endos = {}

ENDO = "endo"
TRAIT = "trait"
OBJ = "obj"

FILE = "file"
ID_COL = "id_col"
EFFECT_COL = "effect_col"
P_COL = "p_col"
SE_COL = "se_col"
MEAN = "mean"
EFFECT = "effect"
BETA = "beta"
SE = "se"
VAR = "var"
FIXED = "fixed"

FH = "fh"
COL_MAP = "col_map"

log("Reading config files...", INFO)
import mpheno

for config_file in options.config_file:
    config_fh = open(config_file, 'r')

    for line in config_fh:
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        # lines in config file are of form
        # [trait] declare trait
        # [trait] file [path]
        # [trait] id_col [integer or header value]
        # [trait] beta_col [integer or header value]
        # [trait] se_col [integer or header value]
        # [trait] beta [value] (optional)
        # [trait] var [value] (optional)
        # [endo] declare endo
        # [endo] mean [value] (optional)
        # [endo] var [value] (optional)
        if len(cols) != 3:
            warn("Skipping line in config file; does not have 3 columns: %s" % line)
        name = cols[0]
        op = cols[1]
        val = cols[2]
        if op == "declare":
            if name in traits or name in endos:
                bail("Error: %s has already been defined" % name)
            if val == TRAIT:
                traits[name] = {}
                traits[name][OBJ] = mpheno.Trait(name)
            elif val == ENDO:
                endos[name] = {}
                endos[name][OBJ] = mpheno.Endo(name)
            else:
                warn("Skipping line in config file; %s is not a type" % val)
        elif name in traits:
            if op == FILE:
                if FILE not in traits[name]:
                    traits[name][FILE] = []
                traits[name][FILE].append(val)
            elif op == ID_COL:
                traits[name][ID_COL] = val
            elif op == EFFECT_COL:
                traits[name][EFFECT_COL] = val
            elif op == SE_COL:
                traits[name][SE_COL] = val
            elif op == P_COL:
                traits[name][P_COL] = val
            elif op == BETA:
                val = float(val)
                traits[name][OBJ].beta = val
            elif op == VAR:
                val = float(val)
                if val <= 0:
                    bail("Variance must be positive (%s)" % val)
                traits[name][OBJ].var = val
            elif op == FIXED:
                if val == BETA or val == VAR:
                    if FIXED not in traits[name]:
                        traits[name][FIXED] = set()
                    traits[name][FIXED].add(val)
            else:
                warn("Skipping line in config file; %s is not an operator for a trait" % op)
        elif name in endos:
            if op == MEAN:
                val = float(val)
                endos[name][OBJ].mean = val
            elif op == VAR:
                val = float(val)
                if val <= 0:
                    bail("Variance must be positive (%s)" % val)
                endos[name][OBJ].var = val
            elif op == FIXED:
                if val == MEAN or val == VAR:
                    if FIXED not in endos[name]:
                        endos[name][FIXED] = set()
                    endos[name][FIXED].add(val)
            else:
                warn("Skipping line in config file; %s is not an operator for an endo" % op)
        else:
            warn("Skipping line in config file; %s has not been defined" % name)

    config_fh.close()

# verify for all traits you have needed files and columns
if train:
    for trait in traits:
        for info in [FILE]:
            if info not in traits[trait]:
                bail("Error: please specify a value for %s for trait %s in the config file" % (info, trait))
        if EFFECT_COL not in traits[trait] and (
                FIXED not in traits[trait] or need_fixed not in traits[trait][FIXED] or need_fixed not in traits[
            trait]):
            bail(
                "Error: please specify a value for %s for trait %s in the config file, or assign %s a value and specify it as fixed for %s" % (
                EFFECT_COL, trait, need_fixed, trait))
        if SE_COL not in traits[trait] and P_COL not in traits[trait] and (
                FIXED not in traits[trait] or need_fixed not in traits[trait][FIXED] or need_fixed not in traits[
            trait]):
            bail(
                "Error: please specify a value for either %s or %s for trait %s in the config file, or assign %s a value and specify it as fixed for %s" % (
                SE_COL, P_COL, trait, need_fixed, trait))

if classify:
    for trait in traits:
        if (EFFECT_COL in traits[trait] and (SE_COL not in traits[trait] and P_COL not in traits[trait])) or (
                SE_COL in traits[trait] and EFFECT_COL not in traits[trait]):
            warn(
                "Both %s and (%s or %s) are required for %s; specifying only one will cause it to be treated as missing" % (
                EFFECT_COL, SE_COL, P_COL, trait))

if classify or phenotype:
    for trait in traits:
        for info in [(BETA, traits[trait][OBJ].beta), (VAR, traits[trait][OBJ].var)]:
            if info[1] is None:
                bail("Error: please specify a value for %s for trait %s in the config file" % (info[0], trait))
    for endo in endos:
        for info in [(MEAN, endos[endo][OBJ].mean), (VAR, endos[endo][OBJ].var)]:
            if info[1] is None:
                bail("Error: please specify a value for %s for endo %s in the config file" % (info[0], endo))

# now open up the files
for trait in traits:
    if FILE not in traits[trait]:
        continue
    traits[trait][FH] = []
    traits[trait][COL_MAP] = []
    for cur_file in traits[trait][FILE]:
        cur_fh = open(cur_file, 'r')
        traits[trait][FH].append(cur_fh)
        header = cur_fh.readline()
        if header is None:
            bail("Error: %s is empty" % cur_file)
        header_cols = header.strip().split(options.delim)
        traits[trait][COL_MAP].append(dict((header_cols[i], i) for i in range(len(header_cols))))

for trait in traits:
    needed_columns = set()
    for info in [EFFECT_COL, SE_COL, P_COL]:
        if info in traits[trait]:
            needed_columns.add(ID_COL)
            needed_columns.add(info)
    for info in needed_columns:
        if info not in traits[trait]:
            bail("Error: please specify a value for %s for trait %s in the config file" % (info, trait))
        for i in range(len(traits[trait][FILE])):
            if traits[trait][info] not in traits[trait][COL_MAP][i]:
                bail("Error: file %s for trait %s does not have column %s" % (
                traits[trait][FILE][i], trait, traits[trait][info]))

use_var_ids = None
if options.var_id_file is not None:
    use_var_ids = set()
    var_id_fh = open(options.var_id_file, 'r')
    for line in var_id_fh:
        var_id = line.strip()
        use_var_ids.add(var_id)
    var_id_fh.close()

var_id_for_chunk = None
if options.num_chunks is not None and options.chunk is not None:
    if options.chunk < 1 or options.chunk > options.num_chunks:
        bail("Error: --chunk must be between 1 and options.num_chunks")
    if options.num_chunks < 1:
        bail("Error: --num-chunks must be at least 1")

    var_id_for_chunk = set()
    for trait in traits:
        if FH not in traits[trait]:
            continue
        for i in range(len(traits[trait][FH])):
            log("Collecting variants for %s..." % traits[trait][FILE][i], INFO)

            for line in traits[trait][FH][i]:
                cols = line.strip('\n').split(options.delim)
                if not len(cols) == len(traits[trait][COL_MAP][i]):
                    warn("Skipping line; number of columns (%s) doesn't match header (%s): %s" % (
                    len(cols), len(traits[trait][COL_MAP][i]), line))

                var_id = None
                if ID_COL in traits[trait]:
                    var_id = cols[traits[trait][COL_MAP][i][traits[trait][ID_COL]]]
                    var_id_for_chunk.add(var_id)
        traits[trait][FH][i].close()

    var_id_for_chunk = sorted(var_id_for_chunk)

    import math

    print("Len of var_id_for_chunk", len(var_id_for_chunk))
    start_index = round((options.chunk - 1) * float(len(var_id_for_chunk)) / options.num_chunks)
    end_index = round(options.chunk * float(len(var_id_for_chunk)) / options.num_chunks)
    var_id_for_chunk = var_id_for_chunk[start_index:end_index]
    var_id_for_chunk = set(var_id_for_chunk)
    print("Chunk: ", options.chunk)
    print("Start index: ", start_index)
    print("End index:", end_index)
    print("Len of var_id_for_chunk", len(var_id_for_chunk))

for trait in traits:
    if FILE not in traits[trait]:
        continue
    traits[trait][FH] = []
    traits[trait][COL_MAP] = []
    for cur_file in traits[trait][FILE]:
        cur_fh = open(cur_file, 'r')
        traits[trait][FH].append(cur_fh)
        header = cur_fh.readline()
        if header is None:
            bail("Error: %s is empty" % cur_file)
        header_cols = header.strip().split(options.delim)
        traits[trait][COL_MAP].append(dict((header_cols[i], i) for i in range(len(header_cols))))

variants = {}
for trait in traits:
    if FH not in traits[trait]:
        continue
    for i in range(len(traits[trait][FH])):
        log("Reading %s..." % traits[trait][FILE][i], INFO)
        for line in traits[trait][FH][i]:
            cols = line.strip('\n').split(options.delim)
            if not len(cols) == len(traits[trait][COL_MAP][i]):
                warn("Skipping line; number of columns (%s) doesn't match header (%s): %s" % (
                len(cols), len(traits[trait][COL_MAP][i]), line))

            var_id = None
            var_beta = None
            var_se = None
            if ID_COL in traits[trait]:
                var_id = cols[traits[trait][COL_MAP][i][traits[trait][ID_COL]]]
                if var_id_for_chunk and var_id not in var_id_for_chunk:
                    continue
                if use_var_ids and var_id not in use_var_ids:
                    continue
            if EFFECT_COL in traits[trait]:
                var_beta = cols[traits[trait][COL_MAP][i][traits[trait][EFFECT_COL]]]
                if var_beta not in options.missing:
                    try:
                        var_beta = float(var_beta)
                    except ValueError:
                        bail("Could not parse %s into a float" % (var_beta))
                else:
                    var_beta = None
            if SE_COL in traits[trait]:
                var_se = cols[traits[trait][COL_MAP][i][traits[trait][SE_COL]]]
                if var_se not in options.missing:
                    try:
                        var_se = float(var_se)
                    except ValueError:
                        bail("Could not parse %s into a float" % (var_se))
                else:
                    var_se = None
            elif P_COL in traits[trait]:
                var_p = cols[traits[trait][COL_MAP][i][traits[trait][P_COL]]]
                if var_p not in options.missing:
                    try:
                        var_p = float(var_p)
                    except ValueError:
                        bail("Could not parse %s into a float" % (var_se))
                    if var_p <= 0 or var_p >= 1:
                        warn("Skipping line; %s is not a valid p-value : %s" % (var_p, line))
                        continue
                    import scipy.stats

                    var_z = -scipy.stats.norm.ppf(var_p / 2)
                    if var_z == 0 or var_beta == 0:
                        var_se = 1
                    else:
                        var_se = abs(var_beta / var_z)
                else:
                    var_p = None

            if train or classify:
                if var_id is not None and var_beta is not None and var_se is not None and var_beta not in options.missing and var_se not in options.missing:
                    if var_se <= 0:
                        warn("Skipping line; se must be positive (%s) : %s" % (var_se, line))
                        continue
                    if var_id not in variants:
                        variants[var_id] = {}
                    variants[var_id][trait] = mpheno.Trait(trait, effect=var_beta, se=var_se)
            elif phenotype:
                if var_id is not None and var_beta is not None and var_beta not in options.missing:
                    if var_id not in variants:
                        variants[var_id] = {}
                    variants[var_id][trait] = mpheno.Trait(trait, effect=var_beta)

# Import relevant modules
import numpy as np

if train:
    log("Fitting parameters...", INFO)
    log_lik = None
    # FIXME: this won't work for multiple endophenotypes
    # FIXME: also may be an issue if endophenotypes can be missing
    for it in range(options.max_train_it):
        log("Iteration %s..." % it)
        # compute new parameters
        trait2_tot = {}
        endo_x_trait_tot = {}
        endo2_for_trait_tot = {}  # total of endo2 values over variants with nonmissing trait

        endo2_tot = {}
        endo_tot = {}

        trait_n = {}
        endo_n = {}

        for endo in endos:
            endo2_for_trait_tot[endo] = {}
            endo_x_trait_tot[endo] = {}
            endo2_tot[endo] = 0
            endo_tot[endo] = 0
            endo_n[endo] = 0
            for trait in traits:
                endo_x_trait_tot[endo][trait] = 0
                endo2_for_trait_tot[endo][trait] = 0

        for trait in traits:
            trait2_tot[trait] = 0
            trait_n[trait] = 0

        new_log_lik = None
        if it > 0:
            new_log_lik = 0

        current_iteration = len(variants)
        for var_id in variants:
            print("current iteration: ", current_iteration)
            if len(variants[var_id]) < options.min_traits:
                continue
            if use_var_ids is not None and var_id not in use_var_ids:
                continue

            log("Examining variant %s" % var_id, TRACE)
            for trait in traits:
                if trait in variants[var_id]:
                    log("%s: %s (%s)" % (trait, variants[var_id][trait].effect, variants[var_id][trait].se), TRACE)

            model_traits = []
            for name in traits:
                if name in variants[var_id]:
                    model_traits.append(mpheno.Trait(name, beta=traits[name][OBJ].beta, var=traits[name][OBJ].var,
                                                     effect=variants[var_id][name].effect,
                                                     se=variants[var_id][name].se))

            model_endos = [endos[name][OBJ] for name in endos]

            M = mpheno.Model(model_endos, model_traits, debug_level=options.debug_level, use_pymc3=options.pymc3)
            M.sample(iter=options.mcmc_samp_iter, burn=options.mcmc_samp_burn, thin=options.mcmc_samp_thin)

            # M.plot_trace('trace_plot_'+var_id)
            for endo in endos:
                if endo not in M.get_node_names():
                    continue
                endo_values = M.get_trace(endo)
                endo2_tot[endo] += np.dot(endo_values, endo_values) / len(endo_values)
                endo_tot[endo] += np.mean(endo_values)
                endo_n[endo] += 1
                for trait in traits:
                    if trait not in M.get_node_names():
                        continue

                    trait_values = M.get_trace(trait)
                    endo_x_trait_tot[endo][trait] += np.dot(endo_values, trait_values) / len(endo_values)
                    endo2_for_trait_tot[endo][trait] += np.dot(endo_values, endo_values) / len(endo_values)

            for trait in traits:
                if trait not in M.get_node_names():
                    continue

                trait_values = M.get_trace(trait)
                trait2_tot[trait] += np.dot(trait_values, trait_values) / len(trait_values)
                trait_n[trait] += 1

            # now compute approximate log likelihood
            if it > 0:
                for endo in endos:
                    endo_values = M.get_trace(endo)
                    mean_endo = np.mean(endo_values)
                    new_log_lik += M.get_normal_like(mean_endo, endos[endo][OBJ].mean, endos[endo][OBJ].var)
                    for trait in traits:
                        if trait not in M.get_node_names():
                            continue
                        trait_values = M.get_trace(trait)
                        mean_trait = np.mean(trait_values)
                        new_log_lik += M.get_normal_like(mean_trait, traits[trait][OBJ].beta * mean_endo,
                                                         traits[trait][OBJ].var)
                for trait in traits:
                    if trait not in M.get_node_names():
                        continue
                    trait_values = M.get_trace(trait)
                    mean_trait = np.mean(trait_values)
                    new_log_lik += M.get_normal_like(variants[var_id][trait].effect, mean_trait,
                                                     variants[var_id][trait].se ** 2)
            current_iteration = current_iteration - 1
        log("loglik=%s" % new_log_lik)
        if log_lik is not None and new_log_lik - log_lik < options.train_eps:
            break
        log_lik = new_log_lik
        log("New parameters:", TRACE)
        for endo in endos:
            if endo_n[endo] == 0:
                continue
            if FIXED not in endos[endo] or MEAN not in endos[endo][FIXED]:
                endos[endo][OBJ].mean = endo_tot[endo] / endo_n[endo]
            if FIXED not in endos[endo] or VAR not in endos[endo][FIXED]:
                endos[endo][OBJ].var = (endo2_tot[endo] - 2 * endos[endo][OBJ].mean * endo_tot[endo] + endo_n[endo] *
                                        endos[endo][OBJ].mean ** 2) / endo_n[endo]
                if endos[endo][OBJ].var < 0:
                    endos[endo][OBJ].var = 0
            log("%s: mean=%s, var=%s" % (endo, endos[endo][OBJ].mean, endos[endo][OBJ].var), TRACE)
            for trait in traits:
                if FIXED not in traits[trait] or BETA not in traits[trait][FIXED]:
                    traits[trait][OBJ].beta = endo_x_trait_tot[endo][trait] / endo2_for_trait_tot[endo][trait]
                if FIXED not in traits[trait] or VAR not in traits[trait][FIXED]:
                    traits[trait][OBJ].var = (trait2_tot[trait] - 2 * traits[trait][OBJ].beta * endo_x_trait_tot[endo][
                        trait] + traits[trait][OBJ].beta ** 2 * endo2_for_trait_tot[endo][trait]) / trait_n[trait]
                    if traits[trait][OBJ].var < 0:
                        traits[trait][OBJ].var = 0

                log("%s: beta=%s, mean=%s, var=%s" % (
                trait, traits[trait][OBJ].beta, traits[trait][OBJ].beta * endos[endo][OBJ].mean,
                traits[trait][OBJ].var), TRACE)

    output_fh = sys.stdout
    if options.output_file:
        output_fh = open(options.output_file, 'w')
    M.draw('test')
    log("Final parameters:", INFO)
    for trait in traits:
        log("%s: beta=%s, mean=%s, var=%s" % (
        trait, traits[trait][OBJ].beta, traits[trait][OBJ].beta * endos[endo][OBJ].mean, traits[trait][OBJ].var), DEBUG)

        norm = 1
        if len(endos) == 1 and options.normalize:
            for endo in endos:
                norm = endos[endo][OBJ].mean

        output_fh.write("%s %s %s\n" % (trait, BETA, traits[trait][OBJ].beta * norm))
        # output_fh.write("%s %s %s\n" % (trait, "fixed", BETA))
        output_fh.write("%s %s %s\n" % (trait, VAR, traits[trait][OBJ].var))
        # output_fh.write("%s %s %s\n" % (trait, "fixed", VAR))

    for endo in endos:
        log("%s: mean=%s, var=%s" % (endo, endos[endo][OBJ].mean, endos[endo][OBJ].var), TRACE)
        output_fh.write("%s %s %s\n" % (endo, MEAN, endos[endo][OBJ].mean / norm))
        output_fh.write("%s %s %s\n" % (endo, VAR, endos[endo][OBJ].var / (norm ** 2)))

        for trait in traits:
            log("%s: beta=%s, mean=%s, var=%s" % (
            trait, traits[trait][OBJ].beta, traits[trait][OBJ].beta * endos[endo][OBJ].mean, traits[trait][OBJ].var),
                TRACE)

    if options.output_file:
        output_fh.close()

if classify or phenotype:

    output_fh = sys.stdout
    if options.output_file:
        output_fh = open(options.output_file, 'w')

    out_delim = "\t"

    trait_means = {}
    trait_devs = {}
    dichotomous_traits = {}

    header_to_write = None
    var_id_to_line = {}
    var_ids = []

    if options.append_to_file is not None:
        append_to_fh = open(options.append_to_file, 'r')
        header_to_write = append_to_fh.readline().strip()
        for line in append_to_fh:
            line = line.strip()
            cols = line.split()
            if len(cols) <= options.append_to_file_id_col:
                warn("Will not write line in append-file; two few columns:\n%s" % line)
            var_id = cols[options.append_to_file_id_col - 1]
            var_id_to_line[var_id] = line
            var_ids.append(var_id)

    if header_to_write is not None:
        output = header_to_write
    else:
        if classify:
            output = "Var_ID"
        elif phenotype:
            output = "IID"

    for endo in endos:
        for value in ["Effect"]:
            if classify:
                output = "%s%s%s_%s_%s" % (output, out_delim, endo, value, 'z')
            output = "%s%s%s_%s_%s" % (output, out_delim, endo, value, 'mean')
            if classify:
                output = "%s%s%s_%s_%s" % (output, out_delim, endo, value, 'sd')
    for trait in traits:
        for value in ["Effect_true"]:
            output = "%s%s%s_%s_%s" % (output, out_delim, trait, value, 'mean')
            if classify:
                output = "%s%s%s_%s_%s" % (output, out_delim, trait, value, 'sd')
            output = "%s%s%s_%s" % (output, out_delim, trait, 'Effect_obs')
            if classify:
                output = "%s%s%s_%s" % (output, out_delim, trait, 'SE_obs')
    if phenotype:
        # normalize of the sample phenotypes
        for name in traits:

            # first determine if dichotomous
            values = set()
            for var_id in variants:
                if len(variants[var_id]) < options.min_traits:
                    continue
                if use_var_ids is not None and var_id not in use_var_ids:
                    continue
                if name not in variants[var_id]:
                    continue
                effect = variants[var_id][name].effect
                values.add(effect)
                if len(values) > 2:
                    break
            if len(values) == 2:
                if 1 in values and 0 in values:
                    dichotomous_traits[name] = lambda x: x
                elif 1 in values and 2 in values:
                    dichotomous_traits[name] = lambda x: x - 1
                else:
                    warn("Trait %s has only two values (%s) but they are not 0/1 or 1/2. Treating as quantitative" % (
                    name, values))

            tot = 0
            tot2 = 0
            n = 0
            for var_id in variants:
                if use_var_ids is not None and var_id not in use_var_ids:
                    continue

                if name in variants[var_id]:
                    effect = variants[var_id][name].effect
                    if name in dichotomous_traits:
                        effect = dichotomous_traits[name](effect)
                    n += 1
                    tot += effect
                    tot2 += effect * effect
            if n > 0:
                trait_means[name] = tot / n
                trait_devs[name] = np.sqrt(tot2 / n - trait_means[name] ** 2)
            else:
                bail("Error: no individuals had values for trait %s" % name)

    output_fh.write("%s\n" % output)
    output_fh.flush()

    variants_to_write = None
    if var_id_to_line:
        variants_to_write = var_ids
    else:
        variants_to_write = sorted(variants)
    # Original chunk code was here

    for var_id in variants_to_write:
        if (var_id in variants) and (len(variants[var_id]) < options.min_traits):
            log("Skipping variant %s" % var_id, DEBUG)
            continue
        if use_var_ids is not None and var_id not in use_var_ids:
            continue

        output = var_id
        if var_id_to_line:
            output = var_id_to_line[var_id]

        if var_id in variants:
            if classify:
                log("Classifying variant %s" % var_id, DEBUG)
            elif phenotype:
                log("Classifying sample %s" % var_id, DEBUG)

            model_traits = []
            used_effects = {}

            trace_save_location = None
            if options.trace_out_file is not None:
                trace_save_location = options.trace_out_file + var_id
            for name in traits:
                effect = None
                se = None
                baseline = 0
                if name in variants[var_id]:
                    effect = variants[var_id][name].effect
                    if name in dichotomous_traits:
                        effect = dichotomous_traits[name](effect)
                    if classify:
                        se = variants[var_id][name].se
                    else:
                        se = trait_devs[name]
                        baseline = trait_means[name]
                        # if trait_devs[name] > 0:
                        #    effect = effect / trait_devs[name]

                used_effects[name] = effect
                model_traits.append(
                    mpheno.Trait(name, beta=traits[name][OBJ].beta, var=traits[name][OBJ].var, effect=effect, se=se,
                                 dichotomous=(name in dichotomous_traits), baseline=baseline))

            model_endos = [endos[name][OBJ] for name in endos]
            M = mpheno.Model(model_endos, model_traits, debug_level=options.debug_level, use_pymc3=options.pymc3)
            # M.compute_map_estimate(max_ests=options.map_max_ests, times_seen_break=options.map_times_seen_break, opt_fun=options.map_opt_fun)
            # Now sample
            M.sample(iter=options.mcmc_samp_iter, burn=options.mcmc_samp_burn, thin=options.mcmc_samp_thin)

            if trace_save_location is not None:
                M.save_trace(trace_save_location)

            for endo in endos:
                # endo_mean = M.get_map_node_value(endo)
                summary_results = M.summary([endo])
                endo_mean = summary_results['mean']

                endo_dev = np.sqrt(np.var(M.get_trace(endo)))
                if endo_dev == 0:
                    endo_dev = 1
                if classify:
                    output = "%s%s%.3g" % (output, out_delim, endo_mean / endo_dev)
                output = "%s%s%.3g" % (output, out_delim, endo_mean)
                # log(M.get_trace(endo), TRACE)
                if classify:
                    output = "%s%s%.3g" % (output, out_delim, endo_dev)
            for trait in traits:
                summary_results = M.summary([trait])
                trait_mean = summary_results['mean']
                # output = "%s%s%.3g" % (output, out_delim, M.get_map_node_value(trait))
                output = "%s%s%.3g" % (output, out_delim, trait_mean)
                if classify:
                    output = "%s%s%.3g" % (output, out_delim, np.sqrt(np.var(M.get_trace(trait))))
                if trait in variants[var_id]:
                    values = [used_effects[trait]]
                    if classify:
                        values += [variants[var_id][trait].se]
                    for value in values:
                        val = "NA"
                        if value is not None:
                            val = value
                        if val == "NA":
                            output = "%s%s%s" % (output, out_delim, val)
                        else:
                            output = "%s%s%.3g" % (output, out_delim, val)
                else:
                    output = "%s%s%s" % (output, out_delim, "NA")
                    if classify:
                        output = "%s%s%s" % (output, out_delim, "NA")
        else:
            for endo in endos:
                output = "%s%s%s" % (output, out_delim, "NA")
                if classify:
                    output = "%s%s%s" % (output, out_delim, "NA")
            if classify or phenotype:
                for trait in traits:
                    output = "%s%s%s" % (output, out_delim, "NA")
                    output = "%s%s%s" % (output, out_delim, "NA")
                    if classify:
                        output = "%s%s%s" % (output, out_delim, "NA")
                        output = "%s%s%s" % (output, out_delim, "NA")

        output_fh.write("%s\n" % output)
        output_fh.flush()
    if options.output_file:
        output_fh.close()

if tracemalloc.is_tracing():
    log("Next displaying memory allocations", DEBUG)
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    log("Done displaying memory allocation", DEBUG)

log("Done with everything!", DEBUG)
