import json
import os.path
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
        print "[!] Couldn't find json data file %s" % (jsondata)
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

        print"[+] #bins: %d, time_window: %d sec, from-to: %s-%s, delta: %d, init_index: %d, end_index: %d" % (total_number_of_bins, time_window_in_seconds, str(lst_sda_time[starting_index]), str(lst_sda_time[ending_index]), (lst_sda_time[ending_index]-lst_sda_time[starting_index]).total_seconds(), starting_index, ending_index)
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


def iostat_cpu_usage():
    total_number_of_bins = 20
    time_window_in_seconds = 100
    time_window_shift = 20
    jsondata = "../full_data.json"

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

    # this will collect the time iostat was run at each time
    lst_time = []

    # this list of Arrays will keep the softmax'ed x
    lst_softmaxed = []

    # List of anomaly
    anomaly_scores = []

    # TESTING list of Dr. Bruno's anomaly scores
    anomaly_scores_T = []

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
    alpha = 0.6
    beta = 0.25
    gamma = 0.15

    # Moving Average of f
    lst_mvavg = [0]
    # Standard Deviation, that are recursively updated below
    lst_std = [0]
    # anomaly threshold
    lst_anomaly_threshold = [0]

    # In Dr. Bruno's paper this is defined as Gamma, we call it Zeta here
    zeta = 0.6
    theta = 0.59

    k = -1

    ############################################################
    # calculate KL distance starting from index 3.
    # Will compare current item, (i), with  (i-1), (i-2), (i-3)
    # alpha * KL( lst_softmaxed[i], lst_sofmaxed[i-1] ) + 
    # beta * KL( lst_softmaxed[i], lst_softmaxed[i-2] ) + 
    # gamma * KL ( lst_softmaxed[i], lst_softmaxed[i-3])
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

        # TESTING
        j1 = [z * alpha for z in lst_softmaxed[i - 1]]
        j2 = [z * beta for z in lst_softmaxed[i - 2]]
        j3 = [z * gamma for z in lst_softmaxed[i - 3]]
        j4 = [sum(index1) for index1 in zip(j1, j2, j3)]

        tl1 = entropy(lst_softmaxed[i], j4)
        tl2 = entropy(j4, lst_softmaxed[i])
        tl3 = tl1 + tl2

        anomaly_scores.append((alpha * kl3) + (beta * kl6) + (gamma * kl9))
        anomaly_scores_T.append(tl3)

        k = k+1
        if i > 3:
            # Adding Moving Average (mvavg) and Standard Deviation (std), Zeta (given)
            lst_mvavg.append((zeta * lst_mvavg[k-1]) + ((1-zeta) * anomaly_scores_T[k-1]))

            std_dev_tmp = np.sqrt(zeta * (lst_std[k - 1] ** 2) + ((1 - zeta) * (tl3 - lst_mvavg[k])**2))
            lst_std.append(std_dev_tmp)

            lst_anomaly_threshold.append(lst_mvavg[k - 1] + theta * lst_std[k-1])



    plt.clf()
    plt.xlabel("Sliding Time Window")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Graph\n#Windows: %d, window: %d sec, win_slide: %d sec" % (
    (len(anomaly_scores) + 3), time_window_in_seconds, time_window_shift))
    plt.grid(True)
    plt.plot(anomaly_scores_T, 'b')  # Plots the Anomaly Scores from KL calculations
    plt.plot(lst_anomaly_threshold, 'r')  # Plots the Anomaly Scores from KL calculations
    plt.show()


        
    plt.clf()                               # clear the figure
    plt.xlabel("Sliding Time Window")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Graph\n#Windows: %d, window: %d sec, win_slide: %d sec" % ((len(anomaly_scores)+3),  time_window_in_seconds, time_window_shift))
    plt.grid(True)
    plt.plot(anomaly_scores)                # Plots the Anomaly Scores from KL calculations
    plt.show()
    # plt.savefig("fixed_bins/iostat_avg_cpu_user/anomaly_score.png".format(i), dpi=500)

    print "[+] Size of Anomaly_Scores: %d"%(len(anomaly_scores))


def main():
    iostat_cpu_usage()
    print"[+] Done..."


if __name__ == "__main__":
    main()
