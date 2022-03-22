import csv
import os
import sys
import subprocess
import shlex

file_name = sys.argv[1]
input1 = sys.argv[2]
input2 = sys.argv[3]
input3 = sys.argv[4]

if len(sys.argv) != 5:
    print("Insufficient arguments")
    sys.exit()

command = "./vta" + file_name + "BundleProfiling vta" + file_name + "Input.bin " + input1 + " " + input2 + " " + input3
proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    proc.communicate(timeout=10)
except subprocess.TimeoutExpired:
    os.system("pkill -9 -ef ./vta" + file_name + "BundleProfiling")

data_list = []

f = open('./'+file_name+'_data.csv','a+')
wr = csv.writer(f)

with open('./'+file_name+'.csv', 'r') as raw:
    reader = csv.reader(raw)
    
    line_num = 0

    for line in reader:
        data_list.append(line)
        line_num += 1

    if line_num < 1: # no output
        result_list = ["error"]

    elif line_num == 1: # simulator case 1
        result_list = data_list[0]
        if len(result_list) > 17:
            if os.path.isfile("./output.bin"):
                command_result = os.system("diff output.bin "+"vta"+file_name+"Golden.bin")
                if command_result == 0:
                    result_list.append("vaild")
                else:
                    result_list.append("invaild")
            else:
                result_list.append("error")
        else:
            result_list.append("error")
            
    elif line_num == 5: # simulator case 2
        result_list = data_list[0][:-1]
        result_list += data_list[4]
        if os.path.isfile("./output.bin"):
            command_result = os.system("diff output.bin "+"vta"+file_name+"Golden.bin")
            if command_result == 0:
                result_list.append("vaild")
            else:
                result_list.append("invaild")
        else:
            result_list.append("error")

    else: # board
        if os.path.isfile("./output.bin"):
            command_result = os.system("diff output.bin "+"vta"+file_name+"Golden.bin")
            if command_result == 0:
                data_list[2] = data_list[2][0].split()
                result_list = data_list[0]
                result_list.append(data_list[2][13])
            else:
                result_list = data_list[0]
                result_list.append(-1)
        else:
            result_list = data_list[0]
            result_list.append("error")
    
    wr.writerow(result_list)

if os.path.isfile("./output.bin"):
    os.remove("./output.bin")
