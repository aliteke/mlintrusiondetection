import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import datetime as dt
from sklearn.utils.extmath import softmax   # Dr. Bruno
from scipy.stats import entropy             # Dr. Bruno
from scipy.linalg import hankel             # To calculate Hankel Toeplitz Matrix for bins at each time window
from numpy import linalg as LA              # LA.eig() will calculate EigenValues and EigenVectors of Henkel Matrix


def hankel_matrix(n):
    hm = hankel(n)
    if len(hm) % 2 == 1:
        dim = len(hm) / 2 + 1
    else:
        dim = len(hm) / 2       # last element will be lost (0)

    h = hm[0:dim, 0:dim]        # real hankel matrix
    eigenvalues, eigenvectors = LA.eig(h)
    return eigenvalues, eigenvectors


def vmstat_d_disk_reads_sda_total():
    total_number_of_bins = 20
    time_window_in_seconds = 1000
    time_window_shift = 10

    jsondata="../full_data.json"

    if not os.path.isfile(jsondata):
        print "[!] Couldn't find json data file %s" % jsondata
        sys.exit()

    with open(jsondata, 'r') as f:
        data_dict = json.load(f)

    print "[+] Total number of items in tree_root: %d" % (len(data_dict["tree_root"]))

    lst_sda_read_total = []
    lst_sda_time = []
    for i in data_dict["tree_root"]:
        for j in i["vmstat_d"]["list_stats"]:
            cur_t = i["vmstat_d"]["date_time"]
            index = cur_t.rfind(":")
            cur_t = str(cur_t[:index]+"."+cur_t[index+1:]).replace("T", " ")
            cur_t = dt.datetime.strptime(str(cur_t[:-3]),'%Y-%m-%d %H:%M:%S.%f')

            lst_sda_time.append(cur_t)

            bytes = int(j["disk_reads"][0]["sda"][0]["total"])
            lst_sda_read_total.append(bytes)

    total_experiment_in_seconds = (lst_sda_time[len(lst_sda_time) - 1] - lst_sda_time[0]).total_seconds()
    max_read_amount = max(lst_sda_read_total)
    min_read_amount = min(lst_sda_read_total)
    delta_read_bytes = max_read_amount-min_read_amount
    bin_width = delta_read_bytes / total_number_of_bins
    bin_edges = range(min_read_amount, max_read_amount, bin_width)

    # list of 2 values, will keep (start_index, ending_index) for each window
    lst_window_start_end_indices = []


    i = 0
    while i < len(lst_sda_time):
        starting_index = i				# starting index for time window
        curtime = lst_sda_time[i]
        endtime = curtime + dt.timedelta(seconds=time_window_in_seconds)

        while (curtime <= endtime) and (i < len(lst_sda_time)):
            i += 1
            if i >= len(lst_sda_time):
                break
            curtime = lst_sda_time[i]

        ending_index = i-1				            # final index in the current time window
        lst_window_start_end_indices.append((starting_index, ending_index))

        plt.clf()						            # clear the figure
        plt.xlabel("Total Disk Read")
        plt.ylabel("# of Elements in a Bin)")
        plt.title("vmstat_d, (Total Disk Reads from sda)," + "\n" +
                  "#bins: %d, sliding_time_window: %d sec, time_delta: %d" %
                  (total_number_of_bins, time_window_in_seconds, (lst_sda_time[ending_index]-lst_sda_time[starting_index]).total_seconds()) +
                  "\n" + "curtime: {}".format(str(lst_sda_time[starting_index])))
        plt.grid(True)
        # n, bins, patches = plt.hist(lst_sda_read_total[starting_index:ending_index], bins=total_number_of_bins, normed=True)
        n, bins, patches = plt.hist(lst_sda_read_total[starting_index:ending_index],
                                    bins=bin_edges,
                                    range=[min_read_amount,max_read_amount],
                                    normed=True)
        cur_mean = np.mean(bins)
        cur_stddev = np.std(bins)
        y = mlab.normpdf(bins, cur_mean, cur_stddev)
        plt.plot(bins, y, '--')

        #print"[+] #bins: %d, time_window: %d sec, from-to: %s-%s, delta: %d, init_index: %d, end_index: %d" % (total_number_of_bins, time_window_in_seconds, str(lst_sda_time[starting_index]), str(lst_sda_time[ending_index]), (lst_sda_time[ending_index]-lst_sda_time[starting_index]).total_seconds(), starting_index, ending_index)
        plt.show()
        plt.savefig("fixed_bins/bins_sda_total_disk_read_vmstatd{}.png".format(i), dpi=500)

"""
    plt.subplots_adjust( bottom=0.2 )
    plt.xticks( rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S.%f')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis_date()

    plt.xlabel("Time")
    plt.ylabel("sda Disk Read-Total(Bytes)")
    plt.title("vmstat_d command, total disk reads from sda")
    plt.grid(True)
    plt.plot(lst_sda_time, lst_sda_read_total)
    plt.show()
    #plt.savefig("sda_total_disk_read_vmstatd.png")
    print "Almost done"
"""


# arguments are 6 list of floating point numbers, and last one is list of times in datetime format
def plot_list(lst_avg_cpu_user, lst_avg_cpu_nice, lst_avg_cpu_system, lst_avg_cpu_iowait, lst_avg_cpu_steal, lst_avg_cpu_idle, lst_time):
    plt.clf()

    plt.xlabel("Date-Time")
    plt.ylabel("CPU Usage")
    plt.title("CPU Usage averages for User, System, IO-Wait, Steal, Nice, Idle (iostat)")
    plt.grid(True)
    plt.plot(lst_time, lst_avg_cpu_user, 'bo', label='avg_cpu_user')
    plt.plot(lst_time, lst_avg_cpu_nice, 'g+', label="avg_cpu_nice")
    plt.plot(lst_time, lst_avg_cpu_system, 'r*', label="avg_cpu_system")
    plt.plot(lst_time, lst_avg_cpu_iowait, 'c:', label='avg_cpu_iowait')
    plt.plot(lst_time, lst_avg_cpu_steal, 'm--', label="avg_cpu_steal")
    # plt.plot(lst_avg_cpu_idle, lst_time,'k-.', label="avg_cpu_idle")
    plt.legend(loc='upper left')
    plt.show()



def generate_syn_cpu_data( data_length, perc ):
    mu, sigma = 0.75, 0.2  # mean and standard deviation
    s = np.random.normal(mu, sigma, data_length)
    su = np.random.uniform(2, 7, int(len(s) * perc))
    anomaly_index = np.random.randint(1, int(len(s)), int(len(su)))
    for i in range(0, len(su) - 1):
        s[anomaly_index[i]] = su[i]

    anomaly_index.sort()
    return s, anomaly_index

def iostat_cpu_usage(jsondata):
    total_number_of_bins = 20
    time_window_in_seconds = 100
    time_window_shift = 20

    if not os.path.isfile(jsondata):
        print "[!] Couldn't find json data file %s" % jsondata
        sys.exit()

    with open(jsondata, 'r') as f:
        data_dict = json.load(f)

    print "[+] Total number of items in tree_root: %d" % (len(data_dict["tree_root"]))

    # we will take average of 3 readings at every iostat command data record.
    lst_avg_cpu_user = []
    lst_avg_cpu_nice = []
    lst_avg_cpu_system = []
    lst_avg_cpu_iowait = []
    lst_avg_cpu_steal = []
    lst_avg_cpu_idle = []

    # Synthetic Random Anomaly Data
    lst_syn_cpu_data = []

    # this will collect the time iostat was run at each time
    lst_time = []

    # this list of Arrays will keep the softmax'ed x
    lst_softmaxed = []

    # List of anomaly
    anomaly_scores = []

    # TESTING list of Dr. Bruno's anomaly scores
    lst_anomaly_scores_T = []

    # list of EigenValues for each window, calculated from HenkelMatrix for that time window's bins array
    lst_eigenvalues = []

    # list of EigenVectors for each window, calculated from HenkelMatrix for that time window's bins array
    lst_eigenvectors = []

    for i in data_dict["tree_root"]:
        cur_t = i["iostat"]["date_time"]
        index = cur_t.rfind(":")
        cur_t = str(cur_t[:index] + "." + cur_t[index + 1:]).replace("T", " ")
        cur_t = dt.datetime.strptime(str(cur_t[:-3]), '%Y-%m-%d %H:%M:%S.%f')
        lst_time.append(cur_t)

        # "iowait": "0.08", "system": "0.26", "idle": "97.74", "user": "1.93", "cpu_nice": "0.00", "steal": "0.00"
        avg_cpu_iowait_sum = 0
        avg_cpu_system_sum = 0
        avg_cpu_idle_sum = 0
        avg_cpu_user_sum = 0
        avg_cpu_cpu_nice_sum = 0
        avg_cpu_steal_sum = 0

        for j in i["iostat"]["list_stats"]["list_stats"]:
            avg_cpu_iowait_sum += float(j["avg-cpu"]["iowait"])
            avg_cpu_system_sum += float(j["avg-cpu"]["system"])
            avg_cpu_idle_sum += float(j["avg-cpu"]["idle"])
            avg_cpu_user_sum += float(j["avg-cpu"]["user"])
            avg_cpu_cpu_nice_sum += float(j["avg-cpu"]["cpu_nice"])
            avg_cpu_steal_sum += float(j["avg-cpu"]["steal"])

        # bytes = int(j["disk_reads"][0]["sda"][0]["total"])
        # lst_sda_read_total.append(bytes)
        lst_avg_cpu_user.append(avg_cpu_user_sum/3.0)
        lst_avg_cpu_nice.append(avg_cpu_cpu_nice_sum/3.0)
        lst_avg_cpu_system.append(avg_cpu_system_sum/3.0)
        lst_avg_cpu_iowait.append(avg_cpu_iowait_sum/3.0)
        lst_avg_cpu_steal.append(avg_cpu_steal_sum/3.0)
        lst_avg_cpu_idle.append(avg_cpu_idle_sum/3.0)

    # Adding plot function for CPU values (Feb 22)
    # plot_list(lst_avg_cpu_user, lst_avg_cpu_nice, lst_avg_cpu_system, lst_avg_cpu_iowait, lst_avg_cpu_steal, lst_avg_cpu_idle, lst_time)

    # Generate Random, Anomaly Data
    (lst_random_cpu_data, anomaly_artificial_index) = generate_syn_cpu_data(len(lst_avg_cpu_user), 0.10)

    lst_avg_cpu_user = lst_random_cpu_data.tolist()

    # calculation for one parameter.
    # TODO We should make this part a function, so that we could call for different CPU parameters
    total_experiment_in_seconds = (lst_time[len(lst_time) - 1] - lst_time[0]).total_seconds()
    print "[+] Total Duration for experiment: %d" % total_experiment_in_seconds
    max_avg_cpu_user = max(lst_avg_cpu_user)
    min_avg_cpu_user = min(lst_avg_cpu_user)
    delta_avg_cpu_user = max_avg_cpu_user - min_avg_cpu_user    # Distance between maximum value and min value
    bin_width = delta_avg_cpu_user / total_number_of_bins       # size of each bin, depending on the number of bins
    bin_edges = np.arange(min_avg_cpu_user, max_avg_cpu_user, bin_width).tolist()  # calculate each bin's boundaries
    bin_edges.append(max_avg_cpu_user)

    # TODO: We need to slide the time window so that it overlaps with the previous window
    greenwich = lst_time[0]                     # First time point from the experiment's log file.
    i = 0
    number_of_time_shifts = 0                   # at each iteration we will shift the current window "time_window_shift"
    starting_index = 0                          # starting index for current time window

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
        i = starting_index                      # reset "i" to start from the start_index for the current window
        curtime = lst_time[starting_index]      # reset curtime to starting time for the current window

        endtime = curtime + dt.timedelta(seconds=time_window_in_seconds)  # upper bound for time record in window

        while (curtime <= endtime) and (i < len(lst_time)):     # loop until we found the index for final time record
            i += 1
            if i >= len(lst_time):
                break
            curtime = lst_time[i]

        ending_index = i - 1                    # index for biggest time value in the current time window

        # add (starting_index, ending_index) to list of window indexes
        lst_window_start_end_indices.append((starting_index, ending_index))

        plt.clf()                               # clear the figure
        plt.xlabel("IOSTAT_Avg_CPU_user")
        plt.ylabel("# of Elements in a Bin)")
        plt.title("iostat, (avg_cpu_usage_for_user)," + "\n" +
                  "#bins: %d, sliding_time_window: %d sec, actual_time_delta: %d" %
                  (total_number_of_bins, time_window_in_seconds,
                   (lst_time[ending_index] - lst_time[starting_index]).total_seconds()) +
                  "\n" + "curtime: {}".format(str(lst_time[starting_index])))
        plt.grid(True)

        x = lst_avg_cpu_user[starting_index:ending_index+1]              # CPU values from the current time window
        n, bins, patches = plt.hist(x,
                                    bins=bin_edges,
                                    range=[min_avg_cpu_user, max_avg_cpu_user],
                                    normed=False,
                                    rwidth=0.85
                                    )

        # NOTE: If you would like to see the images created, un-comment the next line
        #       To save those images in a sub_folder on your computer, uncomment the following line.
        # plt.show()
        # plt.savefig("fixed_bins/iostat_avg_cpu_user/bins_avg_cpu_user_iostat_window_number{}.png".format(i), dpi=500)

        # TODO: Calculate Hankel - Toeplitz Matrix, then calculate EigenValues and EigenVectors
        #       for the bin distribution at the current time window.
        #       number of data points stored at each bin is stored in array => "n"
        e_values, e_vectors = hankel_matrix(x)
        lst_eigenvalues.append(e_values)
        lst_eigenvectors.append(e_vectors)

        # TODO: x should contain the array which has the numbers for each bin at each window of 100 seconds
        #       UPDATE: array "n" returned from plt.hist function above, contains the data we need!
        x1 = np.asarray(n)
        x2 = np.reshape(x1, (1, len(x1)))
        x3 = -x2
        x4 = softmax(x3)
        x5 = np.reshape(x4, len(n))
        x6 = x5.tolist()

        lst_softmaxed.append(x6)        # Probability distribution of cpu usage

        print "[+] Window#: %d, " \
              "#bins: %d, " \
              "time_shift: %d sec, " \
              "window_size: %d sec, " \
              "from-to: %s-%s, " \
              "delta: %d, " \
              "init_index: %d, " \
              "end_index: %d, " \
              "len(x): %d, " \
              "x: %s, " \
              "EigenValues of Hankel(x) : %s" % \
              (number_of_time_shifts, total_number_of_bins, time_window_shift, time_window_in_seconds, str(lst_time[starting_index]),
               str(lst_time[ending_index]), (lst_time[ending_index] - lst_time[starting_index]).total_seconds(),
               starting_index, ending_index, len(x), x, str(e_values))

        """
        # Dr. Korkut abi, wanted to see the distribution for softmax(X) into 20 bins between 0 to 1
        # Calculate bins for PDF of X vector
        plt.clf()  # clear the figure
        max_P = 1
        min_P = 0
        delta_P = max_P - min_P
        bin_width_P = float(delta_P) / float(total_number_of_bins)  # size of each bin, depending on the number of bins
        bin_edges_P = np.arange(min_P, max_P, bin_width_P).tolist()  # calculate each bin's boundaries
        bin_edges_P.append(max_P)
        nP, binsP, patchesP = plt.hist(x6,
                                    bins=bin_edges_P,
                                    range=[min_P, max_P],
                                    normed=False,
                                    rwidth=0.85
                                    )
        plt.show()
        """

    # Now we went through whole array of values, calculated softmaxes, it's time to calculate anomaly_scores
    print "[+] Size of lst_softmaxed: %d" % (len(lst_softmaxed))

    # These are the weights for KL calculations
    m1 = 0.7
    m2 = 0.25
    m3 = 0.05

    # epsilon
    epsilon = 0.025

    # Moving Average of f
    lst_mvavg = [0]

    # Standard Deviation, that are recursively updated below
    lst_std = [0]

    # anomaly threshold
    lst_anomaly_runningavg = [0]

    # difference between f(w) and moving averages
    lst_delta = [epsilon]

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
    # calculate KL distance starting from index 3.
    # Will compare current item, (i), with  (i-1), (i-2), (i-3)
    # m1 * KL( lst_softmaxed[i], lst_sofmaxed[i-1] ) +
    # m2 * KL( lst_softmaxed[i], lst_softmaxed[i-2] ) +
    # m3 * KL ( lst_softmaxed[i], lst_softmaxed[i-3])
    ############################################################
    for i in range(3, len(lst_softmaxed)):
        kl1 = entropy(lst_softmaxed[i], lst_softmaxed[i-1])
        kl2 = entropy(lst_softmaxed[i-1], lst_softmaxed[i])
        kl3 = kl1 + kl2

        kl4 = entropy(lst_softmaxed[i], lst_softmaxed[i-2])
        kl5 = entropy(lst_softmaxed[i-2], lst_softmaxed[i])
        kl6 = kl4 + kl5

        kl7 = entropy(lst_softmaxed[i], lst_softmaxed[i-3])
        kl8 = entropy(lst_softmaxed[i-3], lst_softmaxed[i])
        kl9 = kl7 + kl8

        # lst_softmaxed -> paper's equation-(6)
        # TESTING
        j1 = [z * m1 for z in lst_softmaxed[i - 1]]
        j2 = [z * m2 for z in lst_softmaxed[i - 2]]
        j3 = [z * m3 for z in lst_softmaxed[i - 3]]
        j4 = [sum(index1) for index1 in zip(j1, j2, j3)]

        tl1 = entropy(lst_softmaxed[i], j4)
        tl2 = entropy(j4, lst_softmaxed[i])
        tl3 = tl1 + tl2

        anomaly_scores.append((m1 * kl3) + (m2 * kl6) + (m3 * kl9))     # NOT USED, left from previous implementation.

        # List of anomaly scores "lst_anomaly_scores_T" is used now, in the paper f(w)
        if i == 3:
            lst_anomaly_scores_T.append(tl3)

        if i > 3:
            # Adding Moving Average (mvavg) and Standard Deviation (std), gamma (given)
            # lst_anomaly_scores_T[i] -> f(w)
            if b_start_timer and not b_anomaly_detected and 3 >= reset_wait_counter > 0:
                lst_mvavg.append(0)
                lst_std.append(0)
                lst_anomaly_scores_T.append(0)

            else:
                lst_mvavg.append((gamma * lst_mvavg[i-4]) + ((1-gamma) * lst_anomaly_scores_T[i-4]))
                std_dev_tmp = np.sqrt(gamma * (lst_std[i - 4] ** 2) + ((1 - gamma) * (tl3 - lst_mvavg[i-3])**2))
                lst_std.append(std_dev_tmp)
                lst_anomaly_scores_T.append(tl3)

            # lst_anomaly_runningavg -> paper's nu_{w-1} + alpha*sigma{w-1}  Equation-7
            # lst_mvavg -> paper's mu
            # lst_std -> paper's sigma
            lst_anomaly_runningavg.append(lst_mvavg[i - 4] + alpha * lst_std[i-4])
            lst_delta.append(lst_anomaly_scores_T[-1] - lst_anomaly_runningavg[-1])

            if lst_delta[-1] > epsilon and not b_anomaly_detected:
                b_anomaly_detected = True
                # reset_wait_counter += 1

            # We are in ANOMALY REGION, check for leaving ANOMALY
            elif lst_delta[-1] > epsilon and b_anomaly_detected:
                # do nothing
                continue
            # Going back below epsilon threshold,
            # change the boolean(detected) to false,
            # start the counter (reset_wait_counter)
            elif lst_delta[-1] <= epsilon and b_anomaly_detected:
                b_anomaly_detected = False
                b_start_timer = True

            if b_start_timer and reset_wait_counter < 3:
                reset_wait_counter += 1
            elif b_start_timer and reset_wait_counter == 3:
                b_start_timer = False
                reset_wait_counter = 0

    plt.clf()
    fig = plt.figure(figsize=(12.8, 9.6))
    plt.subplot(3, 1, 1)
    plt.xlabel("Sliding Time Window")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Graph\n#Windows: %d, window: %d sec, "
              "win_slide: %d sec, m1: %.2f, m2: %.2f, m3: %.2f, "
              "alpha: %.2f, gamma: %.2f, epsilon: %.2f" %
              ((len(anomaly_scores) + 3), time_window_in_seconds, time_window_shift, m1, m2, m3, alpha, gamma, epsilon))
    plt.grid(True)
    plt.plot(lst_anomaly_scores_T, 'b', label='f(w)')         # f(w)
    plt.plot(lst_anomaly_runningavg, 'r', label=r"$(\mu_{w-1} + \alpha \sigma_{w-1})$")   # nu_{w-1} + alpha*sigma{w-1}
    plt.legend(loc='upper left')
    plt.subplot(3, 1, 2)
    plt.xlabel("Sliding Time Window")
    plt.ylabel(r"$\mu_{w-1} + \alpha \sigma_{w-1}$")
    plt.plot(lst_delta, 'g', label="Delta")                    # delta, difference between f(w) and moving averages
    plt.plot(epsilon * np.ones(len(lst_delta)), 'y', label="Epsilon")
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.xlabel("Time")
    plt.ylabel(r"Syntetic CPU Usage")
    # plt.plot(lst_avg_cpu_user, 'g', label="CPU")
    plt.plot(lst_time, lst_avg_cpu_user, 'bo', label='CPU_fake_data')
    plt.plot(lst_time, 2*np.ones(len(lst_avg_cpu_user)), 'r', label="Delta")
    plt.legend(loc='upper left')

    # plt.show()

    correct_detection_counter = 0
    # find windows that has anomalies
    for x in range(len(lst_delta)):
        if lst_delta[x] >= epsilon:
            y = str(greenwich + dt.timedelta(seconds=(x * 20)))
            z = str(greenwich + dt.timedelta(seconds=((x * 20) + 100)))
            print "[+] Window Index of Anomaly: %d, Window Start(sec): %s, " \
                  "Window End(sec): %s, lst_delta[x]: %.4f, x: %d, " \
                  "epsilon: %.4f" % (x, y, z, lst_delta[x], x, epsilon)

            window_limits = lst_window_start_end_indices[x]

            for current_anomaly_index in anomaly_artificial_index:
                if window_limits[0] <= current_anomaly_index <= window_limits[1]:
                    print "[+] Correct anomaly detected!! current_anomaly_index: %d, " \
                          "window_limits[0]: %d, window_limits[1]: %d" % \
                          (current_anomaly_index, window_limits[0], window_limits[1])
                    correct_detection_counter += 1
                    del(current_anomaly_index[anomaly_artificial_index])

    print "[+] Detection Rate: " + str(float(correct_detection_counter) / anomaly_artificial_index.size)

    pathtostats = "/".join(jsondata.split("/")[:-2])
    filename = jsondata.split("/")[-1][:-5]

    imagefilename = (pathtostats + "/anomalies/anomaly_score_%s.png") % filename
    plt.savefig(imagefilename, dpi=1000, bbox_inches='tight')
    plt.close(fig)

    '''
    plt.clf()                               # clear the figure
    plt.xlabel("Sliding Time Window")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Graph\n#Windows: %d, window: %d sec, win_slide: %d sec" % ((len(anomaly_scores)+3),  time_window_in_seconds, time_window_shift))
    plt.grid(True)
    plt.plot(anomaly_scores)                # Plots the Anomaly Scores from KL calculations
    plt.show()
    # plt.savefig("fixed_bins/iostat_avg_cpu_user/anomaly_score.png".format(i), dpi=500)
    '''

    print "[+] Size of Anomaly_Scores: %d" % (len(anomaly_scores))


def main():
    # hA_jsonsfolder = "/home/tekeoglu/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorA/stat/jsons/"
    # hB_jsonsfolder = "/home/tekeoglu/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorB/stat/jsons/"
    hA_jsonsfolder = "/home/tekeoglu/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorA/stat/jsons"
    hB_jsonsfolder = "/home/tekeoglu/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorB/stat/jsons"
    
    #hA_jsonsfolder = "C:\\Users\\bekirok\\Documents\\MEGA\\uvic\\ISOT-CID\\logs\\phase2\\hypervisorA\\stat\\jsons"
    #hB_jsonsfolder = "C:\Users\bekirok\Documents\MEGA\uvic\ISOT-CID\logs\phase2\hypervisorB\stat\jsons"

    hA_jsons = os.listdir(hA_jsonsfolder)
    hB_jsons = os.listdir(hB_jsonsfolder)

    for x in hA_jsons:
        iostat_cpu_usage(os.path.join(hA_jsonsfolder, x))

    print"[+] Done with Hypervisor-A jsons..."

    for y in hB_jsons:
        iostat_cpu_usage(os.path.join(hB_jsonsfolder, y))

    print"[+] Done with Hypervisor-B jsons..."


if __name__ == "__main__":
    main()
