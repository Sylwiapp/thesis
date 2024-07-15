# set of helper functions for the analysis
import os, sys

sys.path.append(os.path.dirname(os.path.join("../../", "IDTxl/idtxl")))

import numpy as np
import hashlib
import matplotlib.pyplot as plt
import pprint
import copy

from mne import io
from idtxl.visualise_graph import plot_network


def setCwdHere():
    """Sets cwd to dir in which this script is located"""
    print(os.getcwd())
    os.chdir(sys.path[0])
    print(os.getcwd())


def loadIDTxl():
    """Loads IDTxl"""
    sys.path.append(os.path.dirname(os.path.join("../../", "IDTxl/idtxl")))


def loadRawEEG(srcDir, subCode, condition):
    dataFilePath = srcDir + "sub-" + subCode + "_task_" + condition + "_run-01.vhdr"
    """..."""
    # e.g. 'sub-ARZ000_task_restEyesCl_run-01.vhdr'
    # maybe this helps with errors: https://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte
    eeg = io.read_raw_brainvision(vhdr_fname=dataFilePath)

    return eeg


def listAllEpochFiles(srcDir,subCode,condition,extension):

    list_files = [file_name for file_name in os.listdir(srcDir) 
                  if subCode in file_name 
                  if condition in file_name 
                  if extension in file_name]
    return list_files
    


def loadRawEEG_epochs(srcDir, subCode, condition):
    """..."""
    
    list_epochs = []
    min_length = np.inf
    for epoch_file in listAllEpochFiles(srcDir, subCode, condition, '.vhdr'):

        file_path = os.path.join(srcDir, epoch_file)
        print(file_path)
        eeg = io.read_raw_brainvision(vhdr_fname=file_path)

        eegSignal = eeg.get_data()

        dims = eegSignal.shape
        if dims[1] < min_length:
            min_length = dims[1]

        list_epochs.append(eegSignal)


    list_epochs_truncated = [epoch[:, 0:min_length] for epoch in list_epochs]
    eeg_epoched = np.stack(list_epochs_truncated, axis=0) # to match rps scheme later in adjustSignalToIDTxl() 
    print(eeg_epoched.shape)

    return eeg_epoched



def adjustSignalToIDTxl(eegSignal, containesEpochedData=0, epochIndices=None):
    """..."""
    from idtxl.data import Data  # IDTxl: data class


    if containesEpochedData:
        print("Noticed epoched signal.")

        if epochIndices is not None:
            eegSignal = eegSignal[epochIndices, :, :]
            print(f"Selected following epochs: {epochIndices}.")

        data = Data(
            data=eegSignal, dim_order="rps", normalise=False, seed=1
        )  # seed=1 to have replication of results

    else:
        data = Data(
            data=eegSignal, dim_order="ps", normalise=False, seed=1
        )  # seed=1 to have replication of results;
        # load data (https://pwollstadt.github.io/IDTxl/html/idtxl_data_class.html)

    return data


def computeLogisticCoupledProcesses(
    x0=0.001,
    y0=0.002,
    N=1000,
    controlParameter=4,
    anticipationParameter=0.4,
    couplingStrenght=1,
):
    """..."""
    # r = 4; % logistic map control parameter
    # a = 0.4; % coupling function anticipation modulation parameter
    # e = 1; % dynamic system coupling strength
    # x0 = 0.001;
    # y0 = 0.002;
    # N = 1000; % number of interations
    #
    # refer to https://www.frontiersin.org/articles/10.3389/fphy.2015.00010/full Section 4.1

    r = controlParameter
    a = anticipationParameter
    e = couplingStrenght

    # logistic map:
    # r - control parameter
    f = lambda x: r * x * (1 - x)

    # coupling function:
    # a - anticipation modulation parameter
    g = lambda x: (1 - a) * f(x) + a * f(f(x))

    # coupled dynamic system (n - next step, p - previous step):
    # e - coupling strength
    x_n = lambda x_p: f(x_p)
    y_n = lambda y_p, x_p: (1 - e) * f(y_p) + e * g(x_p)

    X = np.zeros(N)
    X[0] = x0
    Y = np.zeros(N)
    Y[0] = y0
    for n in np.arange(1, N):
        X[n] = x_n(X[n - 1])
        Y[n] = y_n(Y[n - 1], X[n - 1])

    return np.array([X, Y])


def testDataset(name="step2D"):
    """..."""
    step2Darray = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    step3Darray = np.repeat(step2Darray[:, :, np.newaxis], 100, axis=2)

    logisticCoupledProcesses = computeLogisticCoupledProcesses(
        x0=0.001,
        y0=0.002,
        controlParameter=4,
        anticipationParameter=0.4,
        couplingStrenght=1,
        N=20,
    )

    linear2D = np.tile(np.arange(0, 8), [3, 1])

    datasets = {
        "step2D": step2Darray,
        "step3D": step3Darray,
        "logistic-coupling": logisticCoupledProcesses,
        "linear": linear2D,
    }

    return datasets[name]


def visualizeInputData(data, scatter=0):
    """..."""
    # assuming dim_order - processes, samples

    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    #    plt.rcParams["figure.autolayout"] = True

    nProcesses = data.data.shape[0]
    nSamples = data.data.shape[1]
    t = np.arange(nSamples)
    s = data.data

    fig, axes = plt.subplots(nProcesses, sharex=True, sharey=True)
    fig.suptitle = "Input signals"

    cnt = 0
    for ax in axes:
        if scatter:
            ax.scatter(t, s[cnt,])
        else:
            ax.plot(t, s[cnt,])

        ax.set(xlabel="sample")
        cnt += 1

    plt.show()


def visualize2Processes(data):
    """..."""
    x = data.data[0]
    y = data.data[1]

    plt.scatter(x, y)
    plt.title("X vs Y")

    plt.show()


def showResults(results, targets, withFDR=True):
    """..."""
    # print(results._single_target[1]["te"])
    # print(results._single_target[1]["selected_sources_te"])
    # print(results.get_single_target(1, fdr=False))

    if not withFDR:
        print("WARNING! Showing results WITHOUT FDR correction!")

    for target in targets:
        print(f"Printing result for TARGET: {target}")
        pprint.pprint(results.get_single_target(target, fdr=withFDR))

    results.print_edge_list(weights="max_te_lag", fdr=withFDR)
    plot_network(results=results, weights="max_te_lag", fdr=withFDR)
    plt.show()


def computeMovingMultivariateTransferEntropy(data, settings):
    """..."""
    # settings = {
    # "timeRange": [
    #     0,
    #     2000,
    # ],  # starting and ending sample index, or 'all' i.e. take whole time range of the signal
    # "pastSpan": 50,  # number of samples to look into the past or 'all' which always takes maximal number of samples in the past available
    # "step": 1,  # number of samples to progress the calculation window
    # "targets": [0, 4],  # list of targets
    # "sources": range(0, 64),  # list of sources
    # "cmi_estimator": "JidtGaussianCMI",  # see idtxl.network_analysis.analyse_network
    # "fdr_correction": True,  # see idtxl.network_analysis.analyse_network

    from idtxl.multivariate_te import MultivariateTE

    network_analysis = MultivariateTE()

    timeRange = settings["timeRange"]
    span = settings["pastSpan"]
    step = settings["step"]
    targets = settings["targets"]
    sources = settings["sources"]
    cmi_estimator = settings["cmi_estimator"]
    fdr_correction = settings["fdr_correction"]

    N = timeRange[1] - timeRange[0] + 1
    assert step > 0 and isinstance(step, int), "step should be > 0 and integer"
    # TODO lacking assertion for span

    listOfSampleRanges = []
    if span == "all":
        a = 0
        # windows extraction
        for b in range(N - 1, 0, -step):
            listOfSampleRanges.append((a, b))

    else:
        # windows extraction
        b = N - 1
        a = b - span + 1
        while b >= 0 and b > a:
            listOfSampleRanges.append((a, b))
            b -= step
            a = b - span + 1
            if a < 0:
                a = 0

    resultList = []
    # adding custom fields
    aux = {}
    aux["windows_first_sample"] = []
    aux["windows_last_sample"] = []

    for samplesRange in listOfSampleRanges:
        dataFragment = copy.deepcopy(data)  # memory optimization needed here
        dataFragment.set_data(
            dataFragment.data[:, samplesRange[0] : samplesRange[1] + 1, :],
            dim_order="psr",
        )

        # DEBUG
        # print(samplesRange)
        # visualizeInputData(data=dataFragment, scatter=1)

        maxLag = samplesRange[1] - samplesRange[0] - 1
        if maxLag == 0:
            break

        aux["windows_first_sample"].append(samplesRange[0])
        aux["windows_last_sample"].append(samplesRange[1])
        # one can extent aux if needed

        print(f"DEBUG: range={samplesRange}")
        settings = {
            "cmi_estimator": cmi_estimator,
            "max_lag_sources": maxLag,
            "min_lag_sources": 1,
            "fdr_correction": fdr_correction,
                'cmi_estimator': 'JidtGaussianCMI',
            'n_perm_max_stat': 100,  #added by me
            'n_perm_min_stat': 100,
            'n_perm_omnibus': 100,
            'n_perm_max_seq': 100,
            "alpha_min_stat": 0.05,
            "alpha_mi": 0.05,
            "alpha": 0.05,
            "alpha_max_stat": 0.05, 
            "alpha_omnibus": 0.05,
            "alpha_max_seq": 0.05, #to here
        }

        result = network_analysis.analyse_network(
            settings=settings,
            data=dataFragment,
            targets=targets,
            sources=sources,
        )

        # aggregation of specific result
        # save network snapshot to see the evolution
        resultList.append(result)

    resultList.reverse()  # because we collected windows from the rightmost window

    return resultList, aux


def getRelevantSourceInformation(result, target):
    """..."""
    r = result.get_single_target(target, fdr=False)

    mTE = r["te"]
    if mTE is None:
        mTE = 0
    else:
        mTE = mTE[0]  # strip off list

    relevantSourceInfo = {}
    relevantSourceInfo["sourcesList"] = r["selected_vars_sources"]
    relevantSourceInfo["te"] = r["selected_sources_te"]

    return relevantSourceInfo


def plotSingleTargetMteTimeSeries(resultList, target):
    """..."""
    mTE_TimeSeries = []
    relevantSourcesList = []
    teRelevantSourcesList = []

    for result in resultList:  # single result for single window
        r = result.get_single_target(target, fdr=False)

        mTE = r["te"]
        if mTE is None:
            mTE = 0
        else:
            mTE = mTE[0]  # strip off list

        print(mTE)
        mTE_TimeSeries.append(mTE)
        pass

        relevantSourcesList.append(r["selected_vars_sources"])
        teRelevantSourcesList.append(r["selected_sources_te"])

    mTE_TimeSeries = np.array(mTE_TimeSeries)
    sampleAxis = np.array(range(0, len(mTE_TimeSeries)))
    for t, y in zip(sampleAxis, mTE_TimeSeries):
        if y == np.inf:
            plt.arrow(
                t,
                0,
                0,
                1,
                head_width=0.1,
                head_length=0.2,
                fc="red",
                ec="red",
            )

        else:
            plt.stem([t], [y], basefmt="gray", linefmt="b-")

    plt.title(f"Total mTE: [listRelevantSources] -> {target}")
    plt.xlabel("Sample")
    plt.ylabel("mTE")
    plt.show()


def createSourceTargetMteDict(resultList, aux, target):
    """..."""
    # Returns dictionary of the form:
    # (source, t): list of mte values

    relevantSourcesList = []
    teRelevantSourcesList = []
    for result in resultList:  # single result for single window
        relevantSourceInfo = getRelevantSourceInformation(target=target, result=result)
        relSrc = relevantSourceInfo["sourcesList"]
        teRelSrc = relevantSourceInfo["te"]

        relevantSourcesList.append(relSrc)
        teRelevantSourcesList.append(teRelSrc)

    # construct dict keyed by (source,t) values are list of mTE values
    mTE_vs_source_time = {}
    iResult = 0
    for result in relevantSourcesList:
        if result == []:
            iResult += 1
            continue

        iPair = 0
        for source_lag_pair in result:
            source = source_lag_pair[0]

            # convert lags to t
            t = aux["windows_last_sample"][iResult]

            if (source, t) not in mTE_vs_source_time.keys():
                mTE_vs_source_time[(source, t)] = []

            mTE_vs_source_time[(source, t)].append(
                teRelevantSourcesList[iResult][iPair]
            )

            iPair += 1

        iResult += 1

    return mTE_vs_source_time


def plotSigleSourceTargetMteTimeSeries(mTE_vs_source_time, source, targetLabel):
    """..."""
    source_meanMTE_vs_time = {
        key: sum(values) / len(values) for key, values in mTE_vs_source_time.items()
    }

    source_stdMTE_vs_time = {
        key: np.std(values) for key, values in mTE_vs_source_time.items()
    }

    # debug
    print(source_meanMTE_vs_time)
    print(source_stdMTE_vs_time)

    # Create the scatter plot with error bars
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size

    timeSamples = [key[1] for key in source_meanMTE_vs_time if key[0] == source]  # time
    meanMTE_forSource = [
        source_meanMTE_vs_time[key]
        for key in source_meanMTE_vs_time
        if key[0] == source
    ]  # mean mTE values for given source
    stdMTE_forSource = [
        source_stdMTE_vs_time[key] for key in source_stdMTE_vs_time if key[0] == source
    ]  # std of mTE values

    for t, y in zip(timeSamples, meanMTE_forSource):
        if y == np.inf:
            plt.arrow(
                t,
                0,
                0,
                1,
                head_width=0.1,
                head_length=0.2,
                fc="red",
                ec="red",
            )

        else:
            plt.stem([t], [y], basefmt="gray", linefmt="b-")

        plt.errorbar(
            timeSamples,
            meanMTE_forSource,
            yerr=stdMTE_forSource,
            fmt="o",
            markersize=8,
            capsize=5,
            label="mTE std",
            ecolor="k",
            color="blue",
        )

    # Add labels and a legend
    plt.xlabel("Sample")
    plt.ylabel(f"mTE")
    plt.title(f"Source {source} -> {targetLabel}: mTE vs time")
    # plt.legend()

    # Show the plot or save it to a file
    plt.grid(True)  # Optional: Add grid lines
    plt.show()


def getSignalHashString(signal):
    """..."""
    #  convert to string
    signalAsString = str(signal)
    # convert to hash
    hashString = hashlib.md5(signalAsString.encode("utf-8")).hexdigest()

    return hashString
