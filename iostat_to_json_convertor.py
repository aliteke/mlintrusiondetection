import os, sys, json
from easydict import EasyDict as edict		# pip install edict  #https://github.com/makinacorpus/easydict


def main():
	filepath="stat_2018-02-15.log"
	full_json = edict({"tree_root":[]})
	
	dict_iostat_vmstat_a_d = edict({"iostat":"", "vmstat_d":"", "vmstat_a":""})

	with open(filepath) as fp:
		line = fp.readline()
		
		while line:
			if "iostat" in line:
				#print "[--] line: " + line

				date_time = fp.readline().strip()					#	2018-02-15T09:21:13:608862676
				curline=fp.readline()											#	Linux 3.10.0-327.13.1.el7.x86_64 (poseidon0050.wgcloud.uvic.ca) 	02/15/2018 	_x86_64_	(32 CPU)
				name=curline.split(" ")[2][1:-1]					#	poseidon0050.wgcloud.uvic.ca

				dict_iostat_vmstat_a_d.iostat = {"date_time" : date_time, "name" : name }

				dict_list = edict({"list_stats": []})
				for i in range(1,4):				# [1,2,3]
					#print "[+] Iteration: %d" % (i)
					fp.readline()						#	\n
					fp.readline()						#	avg-cpu:  %user   %nice %system %iowait  %steal   %idle

					x=""
					for i in range(0,6):
						x+=fp.readline()

					t = avg_cpu_extractor(x)
					#print json.dumps(t)
					dict_list.list_stats.append(t)
				
				dict_iostat_vmstat_a_d.iostat["list_stats"] = dict_list

				#print dict_iostat_vmstat_a_d
				line = fp.readline()
			
			
			elif "vmstat d" in line:
				print "[--if vmstat d--] "+line
				date_time = fp.readline().strip()					#	2018-02-15T09:21:15:611565736 
				dict_vmstat_d = edict( { "date_time":date_time, "list_stats": [] } )
				fp.readline()						#	disk- ------------reads------------ ------------writes----------- -----IO------
				fp.readline()						#	       total merged sectors      ms  total merged sectors      ms    cur    sec
				x=""
				for i in range(0,3):
					x=""
					for j in range(0,3):			# read the next 3 lines
						x+=fp.readline()
					t = vmstat_d_extractor(x)	# call extractor on 3 lines
					#print "[+] JSON for i:%d, j:%d: %s \n\n" % (i,j,json.dumps(t))
					dict_vmstat_d.list_stats.append(t)

				#print "\n\n[+] dict_vmstat_d [+]\n %s" % json.dumps(dict_vmstat_d)
				dict_iostat_vmstat_a_d.vmstat_d = dict_vmstat_d
				line = fp.readline()


			elif "vmstat a" in line:
				#print "[--if vmstat a--] "+line
				date_time = fp.readline().strip()					#	2018-02-15T09:21:17:619152353
				dict_vmstat_a = edict( { "date_time":date_time, "list_stats": [] } )
				fp.readline()						#	procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
				fp.readline()						#	 r  b   swpd   free  inact active   si   so    bi    bo   in   cs us sy id wa st
				for i in range(0,3):
					x=fp.readline()
					t = vmstat_a_extractor(x)
					dict_vmstat_a.list_stats.append(t)

				#print "\n\n[+] dict_vmstat_a [+]\n %s" % json.dumps(dict_vmstat_a)
				dict_iostat_vmstat_a_d.vmstat_a = dict_vmstat_a
				line = fp.readline()
				#sys.exit()

			else:
				print "[!!!] Line did not match both if and elif, reading next line..."
				line = fp.readline()

			if (dict_iostat_vmstat_a_d.vmstat_a is not '') and (dict_iostat_vmstat_a_d.vmstat_d is not '') and  (dict_iostat_vmstat_a_d.iostat is not ''):
				print "[!!!!!! ] vmstat_a and vmstat_d and iostat are not NULL!!!!"
				full_json.tree_root.append( dict_iostat_vmstat_a_d )
				dict_iostat_vmstat_a_d = edict({"iostat":"", "vmstat_d":"", "vmstat_a":""})

	with open('full_data.json', 'w') as outfile:
		json.dump(full_json, outfile)
		print "[+] Finished writing JSON into file... Check full_data.json file!!!"

def avg_cpu_extractor(inputLines):
	x = inputLines.splitlines()

	cpuLine=x[0]
	tmp=cpuLine.split()
	avg_cpu_user=tmp[0]
	avg_cpu_nice=tmp[1]
	avg_cpu_system=tmp[2]
	avg_cpu_iowait=tmp[3]
	avg_cpu_steal=tmp[4]
	avg_cpu_idle=tmp[5]
	#print "[0]: cpuLine\t" + cpuLine

	deviceLine=x[3]
	sda_name= deviceLine.split()[0]
	sda_tps = deviceLine.split()[1]
	sda_kB_read_s = deviceLine.split()[2]
	sda_kB_wrtn_s = deviceLine.split()[3]
	sda_kB_read = deviceLine.split()[4]
	sda_kB_wrtn = deviceLine.split()[5]

	#print "[1]: " + deviceLine

	deviceLine=x[4]	#dm-0

	#print "[2]: " + deviceLine	
	dm0_name = deviceLine.split()[0]
	dm0_tps = deviceLine.split()[1]
	dm0_kB_read_s = deviceLine.split()[2]
	dm0_kB_wrtn_s = deviceLine.split()[3]
	dm0_kB_read = deviceLine.split()[4]
	dm0_kB_wrtn = deviceLine.split()[5]

	deviceLine=x[5]	#dm-1
	#print "[3]: " + deviceLine
	dm1_name = deviceLine.split()[0]
	dm1_tps = deviceLine.split()[1]
	dm1_kB_read_s = deviceLine.split()[2]
	dm1_kB_wrtn_s = deviceLine.split()[3]
	dm1_kB_read = deviceLine.split()[4]
	dm1_kB_wrtn = deviceLine.split()[5]

	#print "[4]: dm1_kB_wrtn: " + dm1_kB_wrtn	

	d= {"avg-cpu":{ "user": avg_cpu_user, "cpu_nice": avg_cpu_nice, "system": avg_cpu_system, "iowait": avg_cpu_iowait, "steal": avg_cpu_steal, "idle": avg_cpu_idle}, "DeviceStats:": [ {"d_name": sda_name, "tps":sda_tps, "kB_read_s": sda_kB_read_s, "kB_wrtn_s":  sda_kB_wrtn_s, "kB_read": sda_kB_read, "kB_wrtn":sda_kB_wrtn }, {"d_name": dm0_name, "tps":dm0_tps, "kB_read_s": dm0_kB_read_s, "kB_wrtn_s":  dm0_kB_wrtn_s, "kB_read": dm0_kB_read, "kB_wrtn":dm0_kB_wrtn }, {"d_name": dm1_name, "tps":dm1_tps, "kB_read_s": dm1_kB_read_s, "kB_wrtn_s":  dm1_kB_wrtn_s, "kB_read": dm1_kB_read, "kB_wrtn":dm1_kB_wrtn }  ] }
	
	return edict(d)


def vmstat_d_extractor(ls):
	x = ls.splitlines()
	#print "[+] vmstat_d_extractor: " + str(x)
	l0= x[0].split()
	l1= x[1].split()
	l2= x[2].split()

	d = { 
		"disk_reads":[
				{ l0[0]:[ {"total":l0[1], "merged":l0[2], "sectors":l0[3], "ms":l0[4] } ]},
				{ l1[0]:[ {"total":l1[1], "merged":l1[2], "sectors":l1[3], "ms":l1[4] } ]},
				{ l2[0]:[ {"total":l2[1], "merged":l2[2], "sectors":l2[3], "ms":l2[4] } ]}
		],
		"disk_writes":[
				{ l0[0]:[ {"total":l0[5], "merged":l0[6], "sectors":l0[7], "ms":l0[8] } ]},
				{ l1[0]:[ {"total":l1[5], "merged":l1[6], "sectors":l1[7], "ms":l1[8] } ]},
				{ l2[0]:[ {"total":l2[5], "merged":l2[6], "sectors":l2[7], "ms":l2[8] } ]}
		],
		"disk_IO":[
				{ l0[0]:[ {"cur":l0[9], "sec":l0[10]} ]},
				{ l1[0]:[ {"cur":l1[9], "sec":l1[10]} ]},
				{ l2[0]:[ {"cur":l2[9], "sec":l2[10]} ]}
		]
	}
	return edict(d)


def vmstat_a_extractor(ls):
	x = ls.splitlines()
	#print "[+] vmstat_a_extractor: " + str(x)
	l0= x[0].split()
	
	d = { "procs": {"r" : l0[0] , "b":l0[1]}, 
				"memory":{ "swpd":l0[2], "free":l0[3], "inact":l0[4], "active":l0[5] }, 
				"swap":{"si":l0[6], "so":l0[7]}, 
				"io":{"bi":l0[8],"bo":l0[9]}, 
				"system":{"in":l0[10],"cs":l0[11]}, 
				"cpu":{"us":l0[12],"sy":l0[12], "id":l0[13], "wa":l0[14], "st":15} 
	}
	return edict(d)

if __name__ == "__main__":
    main()
