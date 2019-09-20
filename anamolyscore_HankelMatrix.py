from timeit import default_timer as timer
import csv
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy.linalg import hankel             # To calculate Hankel Toeplitz Matrix for bins at each time window
from scipy.linalg import svd
from sklearn.metrics import confusion_matrix            # (ali) testing generation of Confusion Matrix plots
from sklearn.utils.multiclass import unique_labels


def hankel_matrix(n):
    n_appended = n
    # if odd number of elements in "n", we append average of "n" to array "n"
    if len(n) % 2 == 1:
        n_appended.append(sum(n)/float(len(n)))
        dim = (len(n_appended) / 2)
    else:
        dim = (len(n_appended) / 2)

    hm = hankel(n_appended)

    h = hm[0:int(dim), 0:int(dim)]        # real hankel matrix
    # eigenvalues, eigenvectors = LA.eig(h)
    u, svd_h, Vh = svd(h)
    # return eigenvalues, eigenvectors, svd_h, h
    return svd_h, h


# Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# def plot_confusion_matrix(y_true, y_pred, classes,
#                          normalize=False,
#                          title=None,
#                          cmap=plt.cm.Blues):
# (tn, fp, fn, tp)
# TP=tp, FN=fn, FP=fp, TN=tn
def plot_confusion_matrix(TP, FN, FP, TN,
                          classes=None,         # This is an np.array(['Anomaly', 'Bening']) set below
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    cm = np.ndarray(shape=(2, 2), buffer=np.array([TP, FN, FP, TN]), dtype=float)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = np.array(['Anomaly', 'Benign'], dtype='<U10')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'     # if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# arguments are 6 list of floating point numbers, and last one is list of times in datetime format
def plot_list(lst_data_only, lst_avg_cpu_nice, lst_avg_cpu_system, lst_avg_cpu_iowait, lst_avg_cpu_steal, lst_avg_cpu_idle, lst_time):
    plt.clf()

    plt.xlabel("Date-Time")
    plt.ylabel("CPU Usage")
    plt.title("CPU Usage averages for User, System, IO-Wait, Steal, Nice, Idle (iostat)")
    plt.grid(True)
    plt.plot(lst_time, lst_data_only, 'bo', label='avg_cpu_user')
    plt.plot(lst_time, lst_avg_cpu_nice, 'g+', label="avg_cpu_nice")
    plt.plot(lst_time, lst_avg_cpu_system, 'r*', label="avg_cpu_system")
    plt.plot(lst_time, lst_avg_cpu_iowait, 'c:', label='avg_cpu_iowait')
    plt.plot(lst_time, lst_avg_cpu_steal, 'm--', label="avg_cpu_steal")
    # plt.plot(lst_avg_cpu_idle, lst_time,'k-.', label="avg_cpu_idle")
    plt.legend(loc='upper left')
    plt.show()


# If the file is a flatline, we write the output.csv with anomaly_score=0, label=0 for all data-points
def flatlineFileNoAnomaly(datafile):
    lst_data = []
    with open(datafile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lst_data.append(row)

    outputfname = datafile[:datafile.rindex('NAB')] + "NAB/results/xinfinityHankel/" + \
                   datafile[datafile.rindex('data') + 5:datafile.rindex('/') + 1] + \
                   "xinfinityHankel_" + datafile[datafile.rindex('/') + 1:datafile.rindex('.')] + ".csv"

    with open(outputfname, 'wb') as csvfile:
        outwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        outwriter.writerow(['timestamp', 'value', 'anomaly_score', 'label'])

        for x in range(0, len(lst_data)):
            outwriter.writerow([str(lst_data[x]['timestamp']), str(lst_data[x]['value']), 0, 0])
    return


# [DONE] TODO: removes repeating anomaly_index numbers
# TODO: random data generated with NORMAL distribution generates some negative numbers???
def genDataAndAnomalyIndexes(dataLen=0, anomPercent=0, ri=0, usingNABdata=False, nabFile="empty", nabLstTime=None):
    unique_anomaly_indexes = []
    if not usingNABdata:
        # s  -> normal data points (which is then merged with anomalous data, so it's all of the data-points),
        # su -> anomalous data points
        np.random.seed(ri)
        mu = np.random.uniform(0.50, 1.0)
        np.random.seed(ri)
        sigma = np.random.uniform(0.1, 0.5)     # mean and standard deviation
        np.random.seed(ri)
        s = np.random.normal(mu, sigma, dataLen)
        np.random.seed(ri)
        su = np.random.normal(mu*4, sigma*.5, int(len(s) * anomPercent))
        np.random.seed(ri)
        anomaly_index = np.random.randint(0, int(len(s)), int(len(su)))   # random anomaly indexes between [0,len(su)-1]
        unique_anomaly_indexes = np.unique(anomaly_index)

        for i in range(0, len(unique_anomaly_indexes)):
            s[unique_anomaly_indexes[i]] = su[i]

        unique_anomaly_indexes.sort()

        return s.tolist(), unique_anomaly_indexes.tolist()

    elif usingNABdata:
        # now read corresponding labels from NAB/labels/combined_labels.json
        # TODO: alternatively, we can assume everything in NAB/labels/combined_windows.json is anomaly
        # TODO: I guess in that case, we would have less False Positives.
        label_file_path = nabFile[:nabFile.rindex("NAB")] + "NAB"+os.sep+"labels"+os.sep+"combined_labels.json"

        with open(label_file_path, 'r') as f:
            labels_dict = json.load(f)

        anomaly_labels_time = labels_dict[nabFile[nabFile.rindex("data"+os.sep)+5:]]

        for t in anomaly_labels_time:
            # cur_t = str(cur_t[:index] + "." + cur_t[index + 1:]).replace("T", " ")
            dt_t = dt.datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")
            x = nabLstTime.index(dt_t)
            unique_anomaly_indexes.append(x)

        assert len(anomaly_labels_time) == len(unique_anomaly_indexes)

        # return only the "anomaly_indexes", because we already read raw data points from current NAB csv file
        return unique_anomaly_indexes


def anomaly_detection(datafile, ri=0, nab=False):
    # we use this in NAB dataset
    number_of_data_points_in_window = 29
    number_of_data_points_to_shift = 3

    # for random artificial datasets
    time_window_in_seconds = 300
    time_window_shift = 40
    random_index = ri

    lst_data = []
    lst_time = []
    lst_data_only = []

    # TESTING list of Dr. Bruno's anomaly scores of rdist
    lst_anomaly_scores_Hankel_Rank = []

    # list of hankel matrix for each window
    lst_hankel = []

    # detection_rate
    #true_detection_rateH = []
    #false_detection_rateH = []

    if not nab:
        if not os.path.isfile(datafile):
            print("[!] Couldn't find json data file %s" % datafile)
            sys.exit()

        with open(datafile, 'r') as f:
            data_dict = json.load(f)

        print("[+] Total number of items in tree_root: %d" % (len(data_dict["tree_root"])))

        # Read JSON file and populate lst_time with Time values
        for i in data_dict["tree_root"]:
            cur_t = i["iostat"]["date_time"]
            index = cur_t.rfind(":")
            cur_t = str(cur_t[:index] + "." + cur_t[index + 1:]).replace("T", " ")
            cur_t = dt.datetime.strptime(str(cur_t[:-3]), '%Y-%m-%d %H:%M:%S.%f')
            lst_time.append(cur_t)

        # Generate Random, Anomaly Data
        np.random.seed(random_index)
        anomaly_percentage = np.random.uniform(0.05, 0.15)
        np.random.seed(random_index)
        random_length_of_data = int(np.random.uniform(1500, 2500))
        (lst_random_cpu_data, lst_indices_artificial_anomalies) = \
            genDataAndAnomalyIndexes(dataLen=random_length_of_data, anomPercent=anomaly_percentage, ri=random_index)
        lst_data_only = lst_random_cpu_data

        # If the lst_time has less values than number of data points,
        # top up lst_time with new time_points until their length are equal
        time_dif_btw = lst_time[1] - lst_time[0]
        if len(lst_time) < len(lst_data_only):
            while len(lst_time) < len(lst_data_only):
                lst_time.append(lst_time[-1] + time_dif_btw)
        elif len(lst_time) > len(lst_data_only):
            while len(lst_time) < len(lst_data_only):
                lst_time.pop()  # remove last element

        print("[+] Size of anomaly list: %d " % (len(lst_data_only)))

        total_experiment_in_seconds = (lst_time[-1] - lst_time[0]).total_seconds()
        max_lst_data_only = float(max(lst_data_only))
        min_lst_data_only = float(min(lst_data_only))
        print("[+] Total Duration for experiment: %d" % total_experiment_in_seconds)

    # Read data from nab dataset's current input CSV file
    else:
        if not os.path.isfile(datafile):
            print "[!] Couldn't find csv data file %s" % datafile
            sys.exit()

        with open(datafile, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lst_data.append(row)

        for i in lst_data:
            cur_t = i['timestamp']
            cur_t = dt.datetime.strptime(str(cur_t), '%Y-%m-%d %H:%M:%S')
            lst_time.append(cur_t)
            lst_data_only.append(float(i['value']))

        print "[+] Number of data-points in current CSV file: %d " % len(lst_data_only)

        total_experiment_in_seconds = (lst_time[len(lst_time) - 1] - lst_time[0]).total_seconds()
        print "[+] Experiment starts at: %s, ends at: %s" % (str(lst_time[0]), str(lst_time[len(lst_time) - 1]))
        print "[+] Total Duration for experiment: %d seconds, or %d minutes, or %d hours, or %d days." \
              % (total_experiment_in_seconds,
                 (total_experiment_in_seconds / 60.0),
                 (total_experiment_in_seconds / 3600.0),
                 (total_experiment_in_seconds / 86400.0))

        # find time difference between ( lst_time[1] - lst_time[0] * 20 ) -> time_window_in_seconds
        # time_window_in_seconds = 100*60
        time_window_in_seconds = (lst_time[1] - lst_time[0]).seconds * number_of_data_points_in_window
        # time_window_shift = 20*30
        time_window_shift = (lst_time[1] - lst_time[0]).seconds * number_of_data_points_to_shift

        max_lst_data_only = float(max(lst_data_only))
        min_lst_data_only = float(min(lst_data_only))

        # If the MAX and MIN values in the whole dataset are equal,
        # that means time-series data is flatline, w/o anomalies
        if max_lst_data_only == min_lst_data_only:
            flatlineFileNoAnomaly(datafile)
            return

        # Since we are using NAB Data in this section, we will need to find the corresponding
        # labels from NAB dataset and insert the index for anomalous points in the
        # lst_indices_artificial_anomalies. genDataAndAnomalyIndexes function will return them

        lst_indices_artificial_anomalies = \
            genDataAndAnomalyIndexes(usingNABdata=True, nabFile=datafile, nabLstTime=lst_time)


    # [DONE] TODO: We need to slide the time window so that it overlaps with the previous window
    greenwich = lst_time[0]  # First time point from the experiment's log file.
    i = 0
    number_of_time_shifts = 0  # at each iteration we will shift the current window "time_window_shift"
    starting_index = 0  # starting index for current time window

    # list of 2 value tuples, will keep (start_index, ending_index) for each window
    lst_window_start_end_indices = []

    while i < len(lst_time):
        total_shift = number_of_time_shifts * time_window_shift
        number_of_time_shifts += 1

        curtime = greenwich + dt.timedelta(seconds=total_shift)

        # find the current window's starting index,
        # so that lst_time[starting_index] is less than or equal to curtime
        # lst_time[ starting_index ] <= curtime

        while lst_time[starting_index] <= curtime:
            starting_index += 1

        starting_index -= 1
        i = starting_index  # reset "i" to start from the start_index for the current window
        curtime = lst_time[starting_index]  # reset curtime to starting time for the current window

        endtime = curtime + dt.timedelta(seconds=time_window_in_seconds)  # upper bound for time record in window

        while (curtime <= endtime) and (i < len(lst_time)):  # loop until we found the index for final time record
            i += 1
            if i >= len(lst_time):
                break
            curtime = lst_time[i]

        ending_index = i - 1  # index for biggest time value in the current time window

        # add (starting_index, ending_index) to list of window indexes
        lst_window_start_end_indices.append((starting_index, ending_index))

        x = lst_data_only[starting_index:ending_index + 1]  # CPU values from the current time window

        # very last window might have less elements than previous windows,
        # we need to expand the number of elements to match size of previous windows.
        if len(lst_window_start_end_indices) > 1:
            while len(x) != lst_window_start_end_indices[0][1] - lst_window_start_end_indices[0][0] + 1:
                if len(x) > lst_window_start_end_indices[0][1] - lst_window_start_end_indices[0][0] + 1:
                    x.pop(0)
                else:  # in case number of elements in x is less than first window's size
                    x.append(sum(x) / float(len(x)))

        svd_h, H = hankel_matrix(x)
        lst_hankel.append(H)  # store Hankel matrix of current window
        # lst_eigenvalues.append(e_values)
        # lst_eigenvectors.append(`e_vectors)
        # lst_svd_hankel.append(svd_h[1])  # USV

    # These are the weights for KL calculations
    m1 = 0.7
    m2 = 0.25
    m3 = 0.05

    # epsilon
    epsilon = 2.1
    # Moving Average of f
    lst_mvavgH = [0]
    # Standard Deviation, that are recursively updated below
    lst_stdH = [0]
    # anomaly threshold
    lst_anomaly_runningavgH = [0]
    # difference between f(w) and moving averages
    lst_deltaH = [epsilon/2, epsilon/2, epsilon/2]
    # In paper gamma is used in Eq.7 to calculate MU_w
    gamma = 0.3
    # In paper alpha is used in Eq.8 to calculate f(w)
    alpha = 0.59

    # this will count till 3 before calculating new moving averages
    reset_wait_counter = 0

    # anomaly detected
    b_anomaly_detected = False

    # right after an anomaly, we need to start counting,
    # keep another boolean to detect the start of counting time
    b_start_timer = False

    ############################################################
    for i in range(3, len(lst_hankel)):
        ############### Hankel and RANK comp############## KORKUT
        h1 = m1 * lst_hankel[i - 1]
        h2 = m2 * lst_hankel[i - 2]
        h3 = m3 * lst_hankel[i - 3]
        h4 = h1 + h2 + h3
        threshold = .7              # threshold for svd to filter
        Hd = lst_hankel[i] - h4     # TODO: Should we take ABSOLUTE Values?? Some table cells have negative values???
        svd_Hd = svd(Hd)[1]
        ratio = svd_Hd / svd_Hd[0]
        svd_Hd[ratio < threshold] = 0  # only dominant svd stays
        rankHd = len(np.nonzero(svd_Hd)[0])  # rank of diff
        hW = rankHd
        # List of anomaly scores "lst_anomaly_scores_T" is used now, in the paper f(w)
        if i == 3:
            # lst_anomaly_scores_T.append(tl3)
            lst_anomaly_scores_Hankel_Rank.append(hW)
            lst_deltaH.append(abs(lst_anomaly_scores_Hankel_Rank[-1] - lst_anomaly_runningavgH[-1]))

        if i > 3:
            # Adding Moving Average (mvavg) and Standard Deviation (std), gamma (given)
            # lst_anomaly_scores_T[i] -> f(w)
            if b_start_timer and not b_anomaly_detected and 3 >= reset_wait_counter > 0:
                #                ###### same for hankel
                lst_mvavgH.append(0)
                lst_stdH.append(0)
                lst_anomaly_scores_Hankel_Rank.append(0)
            else:
                #                ###### same for hankel
                lst_mvavgH.append((gamma * lst_mvavgH[i - 4]) + ((1 - gamma) * lst_anomaly_scores_Hankel_Rank[i - 4]))
                std_dev_tmp = np.sqrt(gamma * (lst_stdH[i - 4] ** 2) + ((1 - gamma) * (hW - lst_mvavgH[i - 3]) ** 2))
                lst_stdH.append(std_dev_tmp)
                lst_anomaly_scores_Hankel_Rank.append(hW)
            # lst_anomaly_runningavg -> paper's nu_{w-1} + alpha*sigma{w-1}  Equation-7
            # lst_mvavg -> paper's mu
            # lst_std -> paper's sigma
            #            lst_anomaly_runningavg.append(lst_mvavg[i - 4] + alpha * lst_std[i-4])
            #            lst_delta.append(lst_anomaly_scores_T[-1] - lst_anomaly_runningavg[-1])
            #            ### same for hankel
            lst_anomaly_runningavgH.append(lst_mvavgH[i - 4] + alpha * lst_stdH[i - 4])
            lst_deltaH.append(abs(lst_anomaly_scores_Hankel_Rank[-1] - lst_anomaly_runningavgH[-1]))
            # LEFT HERE - I need to run hankel rank bound for anomaly detection
            if abs(lst_deltaH[-1]) > epsilon and not b_anomaly_detected:
                b_anomaly_detected = True
                # reset_wait_counter += 1

            # We are in ANOMALY REGION, check for leaving ANOMALY
            elif abs(lst_deltaH[-1]) > epsilon and b_anomaly_detected:
                # do nothing
                continue
            # Going back below epsilon threshold,
            # change the boolean(detected) to false,
            # start the counter (reset_wait_counter)
            elif abs(lst_deltaH[-1]) <= epsilon and b_anomaly_detected:
                b_anomaly_detected = False
                b_start_timer = True

            if b_start_timer and reset_wait_counter < 3:
                reset_wait_counter += 1
            elif b_start_timer and reset_wait_counter == 3:
                b_start_timer = False
                reset_wait_counter = 0

    correct_detection_counter = 0           # true_positive
    false_positive_detection_counter = 0
    lst_anomaly_indices = lst_indices_artificial_anomalies[:]
    # find windows that has anomalies
    for x in range(len(lst_deltaH)):  # For each window, check if it's delta is more than epsilon
        if abs(lst_deltaH[x]) >= epsilon:
            # y = str(greenwich + dt.timedelta(seconds=(x * time_window_shift)))
            # z = str(greenwich + dt.timedelta(seconds=((x * greenwich + dt.timedelta) + time_window_in_seconds)))
            # print("[+] Window Index of Anomaly: %d, Window Start(sec): %s, "
            #      "Window End(sec): %s, lst_delta[x]: %.4f, x: %d, "
            #      "epsilon: %.4f" % (x, y, z, abs(lst_deltaH[x]), x, epsilon))

            # If so, that means our method claims there is anomaly in current window
            # Find the time indices for the current window (which is same as anomalous CPU value indices)
            window_limits = lst_window_start_end_indices[x]

            # If current window indices include an anomaly,
            # increment the counter, remove that anomaly index from the list
            total_number_of_anomalies_left_undetected = len(lst_anomaly_indices)
            current_anomaly_index = 0
            current_window_already_TP = False
            # for current_anomaly_index in lst_anomaly_indices:
            while total_number_of_anomalies_left_undetected > 0 and current_anomaly_index < len(lst_anomaly_indices):
                if window_limits[0] <= lst_anomaly_indices[current_anomaly_index]:
                    if lst_anomaly_indices[current_anomaly_index] <= window_limits[1]:
                        '''
                        print("[+] Correct anomaly detected!! " 
                              "lst_anomaly_indices[current_anomaly_index]: %d, " 
                              "window_limits[0]: %d, window_limits[1]: %d" %
                              (lst_anomaly_indices[current_anomaly_index], window_limits[0], window_limits[1]))
                        '''
                        correct_detection_counter += 1
                        current_window_already_TP = True

                        del (lst_anomaly_indices[current_anomaly_index])
                        total_number_of_anomalies_left_undetected -= 1

                    # since anomaly indices are sorted, if current index is larger than upper window index, we can stop
                    # searching through the rest of the indices, they will all be larger.
                    else:
                        if not current_window_already_TP:
                            false_positive_detection_counter += 1
                        break
                else:
                    # Our method thinks it's anomaly, but in randomly generated CPU values list
                    # current index doesn't have an anomalous value
                    # false_positive_detection_counter += 1
                    current_anomaly_index += 1

    # [DONE] TODO:  lst_indices_artificial_anomalies is not the original ANOMALY indexes, because in line#420 we remove
    # [DONE] TODO:  the indices that are found by our algorithm !!! FIX THIS BUG !!!
    # lst_indices_artificial_anomalies is the original ND-ARRAY for anomalous CPU values

    # true_positive_detection_rate = (float(correct_detection_counter) / len(lst_indices_artificial_anomalies))
    # false_positive_detection_rate =
    #       (float(false_positive_detection_counter) / float(len(lst_data_only)-len(lst_indices_artificial_anomalies)))

    # From Dr. Issa's anomaly detection paper:
    # False Positive Rate (FPR) , Detection Rate (DR)
    # DR is same as Recall
    # FPR = 100 * ( sum(FP) / sum(FP+TN) )
    # DR  = 100 * ( sum(TP) / sum(TP+FN) )

    precision = recall = FPR = 0
    TP = float(correct_detection_counter)
    FP = float(false_positive_detection_counter)
    TN = (len(lst_data_only) - len(lst_indices_artificial_anomalies)) - FP
    FN = len(lst_indices_artificial_anomalies) - TP

    # precision = TP / (TP + FP)
    if (float(TP) + float(FP)) != 0:
        precision = float(TP) / (float(TP) + float(FP))
    else:
        precision = 0

    # recall = TP / (TP + FN)
    if (float(TP) + float(FN)) != 0:
        recall = TP / (float(TP) + float(FN))
    else:
        recall = 0

    # False Positive Rate
    if (float(FP) + float(TN)) != 0:
        FPR = float(FP) / (float(FP) + float(TN))
    else:
        FPR = 0

    # RANDOM ARTIFICIAL DATA
    if not nab:
        print "[+] #Anomalies:%d, TP: %.4f, FP: %.4f, TN: %.4f, FN: %.4f, " \
              "recall: %.4f, precision: %.4f, FPR: %.4f\n" % \
              (len(lst_indices_artificial_anomalies), TP, FP, TN, FN, recall, precision, FPR)
        return TP, FP, TN, FN, recall, precision, FPR, len(lst_indices_artificial_anomalies), len(lst_data_only)

    # NAB DATA
    else:
        print "[+] #Anomalies:%d, TP: %.4f, FP: %.4f, TN: %.4f, FN: %.4f, " \
              "recall: %.4f, precision: %.4f, FPR: %.4f, DataFile: %s\n" % \
              (len(lst_indices_artificial_anomalies), TP, FP, TN, FN, recall, precision, FPR,
               datafile[datafile.rfind('data/') + 5:])

        #####...This will make the 3-subplot graph a: h(w), b: delta_h, c: lst_data_only... #####
        # We can do Conditional print for top-5 NAB files,
        top_5_nab_files = ["exchange-3_cpm_results.csv", "exchange-4_cpm_results.csv", "exchange-4_cpc_results.csv", "ec2_disk_write_bytes_1ef3de.csv", "ec2_cpu_utilization_fe7f93.csv"]
        if datafile[datafile.rfind('/') + 1:] in top_5_nab_files:
            print "[+] Creating 3-subplot graph"
            w_shift_amount = lst_window_start_end_indices[1][0] - lst_window_start_end_indices[0][0]
            w_size = lst_window_start_end_indices[0][1] - lst_window_start_end_indices[0][0] + 1
            plt.clf()
            plt.figure(figsize=(12.8, 9.6))
            plt.subplot(3, 1, 1)
            plt.ylabel("Hankel Anomaly Score")
            plt.title("(a) Anomaly Score Graph\n#Windows: %d, "
                      r"#Pts in Window: %d, "
                      r"#Pts in Slide: %d, "
                      r"$\delta_1$: %.2f, "
                      r"$\delta_2$: %.2f, "
                      r"$\delta_3$: %.2f, "
                      r"$\alpha$: %.2f, "
                      r"$\gamma$: %.2f, "
                      r"$\epsilon$: %.2f" %
                      ((len(lst_deltaH)), w_size, w_shift_amount, m1, m2, m3, alpha, gamma, epsilon))
            plt.grid(True)
            plt.plot(lst_anomaly_scores_Hankel_Rank, 'b', label='h(w)')  # f(w)
            plt.plot(lst_anomaly_runningavgH, 'r-.',
                     label=r"$(\mu_{w-1} + \alpha \sigma_{w-1})$")  # nu_{w-1} + alpha*sigma{w-1}
            plt.legend(loc='upper left', frameon=False)
            plt.subplot(3, 1, 2)
            # plt.xlabel("Sliding Time Window")
            plt.ylabel(r"Delta$_H$")
            plt.stem(lst_deltaH, 'g:', linewidth=2,
                     label=r"$h(w) - \mu_{w-1} + \alpha \sigma_{w-1}$")  # delta, difference between f(w) and moving averages
            plt.plot(epsilon * np.ones(len(lst_deltaH)), 'y', label=r"$\epsilon$")
            # plt.plot(-1 * epsilon * np.ones(len(lst_deltaH)), 'y', label="Epsilon")
            plt.legend(loc='upper left', frameon=False)
            plt.title(r"(b) Difference for each Window - Delta$_H$")
            plt.subplot(3, 1, 3)
            plt.xlabel("Sliding Time Window")
            plt.ylabel(r"%s Data" % datafile[datafile.rfind('/') + 1:])
            plt.title("(c) TP: %.4f, FP: %.4f, #Anomalies: %d" % (TP, FP, len(lst_indices_artificial_anomalies)))
            plt.plot(lst_data_only, 'bo', label='Artificial Data')
            # plt.plot(lst_time, 2 * np.ones(len(lst_data_only)), 'r-', label="Lower Bound on Anomalous")
            plt.legend(loc='upper left', frameon=False)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("hw_top5NABplots/"+datafile[datafile.rfind('/') + 1:]+"_.png", format='png')

        return TP, FP, TN, FN, recall, precision, FPR, len(lst_indices_artificial_anomalies), len(lst_data_only)


def main():
    user = "ali"              # user="ali" | "korkut"
    computer = "desktop"      # computer = "desktop" | laptop
    # If you want to run the Hankel method against labeled NAB dataset, set this Boolean var to TRUE
    runAgainstNABdata = True        # False

    if "ali" in user:
        if "desktop" in computer:
            # NAB DATASET:      ali's SUNY-Desktop
            nab_data_folder = "/home/tekeoglu/MEGAsync/uvic/NAB/data"
            hA_jsonsfolder = "/home/tekeoglu/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorA/stat/jsons"
            hB_jsonsfolder = "/home/tekeoglu/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorB/stat/jsons"

        elif "laptop" in computer:
            # NAB DATASET:      ali's HP-Laptop
            nab_data_folder = "/home/tekeoglu/MEGA/MEGAsync/uvic/NAB/data"
            hA_jsonsfolder = "/home/tekeoglu/MEGA/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorA/stat/jsons/"
            hB_jsonsfolder = "/home/tekeoglu/MEGA/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorB/stat/jsons/"
        else:
            print "[-] wrong computer"
            exit()
    elif "korkut" in user:
        if "desktop" in computer:
            # Dr Korkut abi's office computer
            nab_data_folder = "C:\\Users\\bekirok\\Documents\\MEGA Ali\\NAB\\data"
            hA_jsonsfolder = "C:\\Users\\bekirok\\Documents\\MEGA Ali\\ISOT-CID\\logs\\phase2\\hypervisorA\\stat\\jsons"
            hB_jsonsfolder = "C:\\Users\\bekirok\\Documents\\MEGA Ali\\ISOT-CID\\logs\\phase2\\hypervisorB\\stat\\jsons"
        elif "laptop" in computer:
            # Dr Korkut abi's Laptop directories
            hA_jsonsfolder = "C:\\Users\\Korkut\\Documents\\MEGA\\uvic\\ISOT-CID\\logs\\phase2\\hypervisorA\\stat\\jsons"
            hB_jsonsfolder = "C:\\Users\\Korkut\\Documents\\MEGA\\uvic\\ISOT-CID\\logs\\phase2\\hypervisorB\\stat\\jsons"
            nab_data_folder = "C:\\Users\\Korkut\\Documents\\MEGA\\uvic\\NAB\\data"
    else:
        print "[-] wrong user"
        exit()

    hA_jsons = os.listdir(hA_jsonsfolder)
    hB_jsons = os.listdir(hB_jsonsfolder)

    # Save the resulting stats for each file from NAB data-set
    lst_TPFPTNFN_AnomLenDataLenFileName = []

    if not runAgainstNABdata:
        np.random.seed(30)
        random_integer = np.random.randint(0, 5000, 10000)  # random anomaly indexes between [0,len(su)-1]
        unique_random_integer = np.unique(random_integer)
        counter = 0
        totalIteration = 125

        for iteration in range(0, totalIteration):
            for x in hA_jsons:
                (TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data) = \
                    anomaly_detection(os.path.join(hA_jsonsfolder, x), unique_random_integer[counter])

                lst_TPFPTNFN_AnomLenDataLenFileName.append((TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data,
                                                            (str(x) + "_" + str(iteration))))
                # true_detection_rateH.append(true_positive_detection_rate)
                # false_detection_rateH.append(false_positive_detection_rate)

                # len_Anomaly.append(len_anomaly)
                # len_Data.append(len_data)
                print "[+]counter: %d, x: %s, iteration: %d" % (counter, x, iteration)
                counter += 1

            print("[+] Done with Hypervisor-A jsons...Iteration: %d / %d") % (iteration, totalIteration-1)

            for y in hB_jsons:
                (TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data) = \
                    anomaly_detection(os.path.join(hB_jsonsfolder, y), unique_random_integer[counter])

                lst_TPFPTNFN_AnomLenDataLenFileName.append((TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data,
                                                            (str(x) + "_" + str(iteration))))

                # len_Anomaly.append(len_anomaly)
                # len_Data.append(len_data)
                counter += 1

            print("[+] Done with Hypervisor-B jsons...Iteration: %d / %d") % (iteration, totalIteration-1)

        # PLOT HERE THE RESULTS OF Randomly Generated Artificial Data
        all_RECALLs = []
        all_PRECISIONs = []
        all_FPRs = []
        for curResult in lst_TPFPTNFN_AnomLenDataLenFileName:
            all_RECALLs.append(curResult[4])
            all_PRECISIONs.append(curResult[5])
            all_FPRs.append(curResult[6])

        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.subplot(2, 1, 1)
        plt.hist(all_PRECISIONs, bins=np.linspace(0.0, 1.0, num=20))
        plt.xlabel("Distribution of Precision (PR) in %d Simulations" % len(lst_TPFPTNFN_AnomLenDataLenFileName))
        plt.ylabel("#simulations in each Bin")
        plt.title("Presicion")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.hist(all_FPRs, bins=np.linspace(0.0, 1.0, num=20))
        plt.xlabel("Distribution of False Positive Rate (FPR) in %d Simulations" % len(lst_TPFPTNFN_AnomLenDataLenFileName))
        plt.ylabel("#simulations in each Bin")
        plt.title("False Positive Detection Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ArtificialRandomDataPlot/artifrandomHistogram.png", format='png')

    # run with NAB dataset
    else:
        data_folders = [f for f in os.listdir(nab_data_folder) if os.path.isdir(os.path.join(nab_data_folder, f))]

        if data_folders.__contains__('output'):
            data_folders.remove('output')

        # ALi Comp
        if "ali" in user:
            separation_char = '/'

        # Korkut Comp
        elif "korkut" in user:
            separation_char = '\\'
            for df in data_folders:
                files = [f for f in os.listdir(nab_data_folder + '\\' + df)
                         if os.path.isfile(os.path.join(nab_data_folder + '\\' + df, f))]
        else:
            print "[-] wrong user"
            exit()

        for df in data_folders:
            files = [f for f in os.listdir(nab_data_folder + separation_char + df)
                     if os.path.isfile(os.path.join(nab_data_folder + separation_char + df, f))]

            for i in range(len(files)):
                if files[i] == "art_flatline.csv":
                    continue
                start = timer()
                print "[+] Starting file: %s" % (files[i])
                # Ali Comp
                #(TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data) = \
                   # anomaly_detection(nab_data_folder + '/' + df + '/' + files[i], nab=runAgainstNABdata)

                (TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data) = \
                    anomaly_detection(nab_data_folder + separation_char + df + separation_char + files[i], nab=runAgainstNABdata)
                end = timer()

                lst_TPFPTNFN_AnomLenDataLenFileName.append((TP, FP, TN, FN, rc, pr, FPR, len_anomaly, len_data, files[i]))

                print "[+] Finished file: %s, in %f seconds." % (files[i], (end - start))

        print "[+] Done with all of NAB Data Corpus!! \n\nNow printing results: "

        print "[len_anomaly]\t\t[TP]\t\t[FP]\t\t[TN]\t\t[FN]\t\t" \
              "[recall]\t\t[precision]\t\t[FPR]\t\t[len_data]\t\t[fileName]"

        for i in lst_TPFPTNFN_AnomLenDataLenFileName:
            print "#Anomalies:%d, TP: %.4f, FP: %.4f, TN: %.4f, FN: %.4f, " \
                  "recall: %.4f, precision: %.4f, FPR: %.4f, #data: %d, DataFile: %s" % \
                  (i[7], i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[8], i[9])

        with open('results.csv', 'wb') as csvfile:
            outwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            outwriter.writerow(
                ['len_anomaly', 'TP', 'FP', 'TN', 'FN', 'Recall', 'Precision', 'FPR', 'len_data', 'fname'])

            for i in lst_results:
                outwriter.writerow([str(i[7]), str(i[0]), str(i[1]), str(i[2]),
                                    str(i[3]), str(i[4]), str(i[5]), str(i[6]), str(i[8]), str(i[9])])

        # Try to plot confusion matrix for each of the NAB file's result
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        # plot_confusion_matrix(y_test, y_pred, classes=class_names,
        #                      title='Confusion matrix, without normalization')

        for i in lst_TPFPTNFN_AnomLenDataLenFileName:
            tn = i[2]
            fp = i[1]
            fn = i[3]
            tp = i[0]
            plot_confusion_matrix(TP=tp, FN=fn, FP=fp, TN=tn,
                                  title='TP: %d, FN: %d, FP: %d, TN: %d, %s' % (
                                  int(tp), int(fn), int(fp), int(tn), i[9]),
                                  normalize=False)
            plt.savefig("ConfMatrices/" + i[9] + ".png", format='png')

        # Plot normalized confusion matrix
        # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
        #                      title='Normalized confusion matrix')
        plt.show()

    return lst_TPFPTNFN_AnomLenDataLenFileName


if __name__ == "__main__":
    lst_results = main()
    print "[+] Program Finished--DONE!!!!!!"
