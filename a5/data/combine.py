import csv
import sys


numnodes = 7
numthreads = 16
k_start = 8
k_end = 12

def generate():
    for k in range(k_start, k_end+1):
        fd = open("k_"+ str(k)+'.csv', "wb")
        writer = csv.writer(fd,delimiter=',')
        writer.writerow(range(numthreads+1))

        for nodes in range(1, numnodes+1):
            row = [nodes]
            for threads in range(1, numthreads+1):
                f = open(str(nodes)+"nodes_"+str(threads)+"threads_riemann.csv", "rb")
                reader = csv.reader(f)

                for r in reader:
                    if r[0] == str(k):
                        row.append(r[2])
            f.close()
            writer.writerow(row)
        fd.close()



if __name__ == "__main__":
    generate()
