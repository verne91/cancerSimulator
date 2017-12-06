#!/user/bin/env python
import numpy as np
import pandas as pd
from collections import Counter
from sympy import binomial
from multiprocessing import Pool
import csv
import sys
import time
import random
import argparse
import os

#fitness w_j
def fitness(s, k, x):
    sumAllj = 0
    for l in range(k+1):
        sumAllj += ((1+s)**l)*x[l]
    w = np.array([0 for i in range(k+1)], dtype=float)
    for j in range(k+1):
        w[j] = (1+s)**j/sumAllj
    return w

#weights for sampling
def weights(d, k, u, w, x):
    theta = np.array([0 for i in range(k+1)], dtype=float)
    for j in range(k+1):
        tsum = 0
        for i in range(j+1):
            tsum += binomial(d-i,j-i)*(u**(j-i))*((1-u)**(d-j))*w[i]*x[i]
        theta[j] = tsum
    return theta

def count_sampling(index, sample_weights):
    return Counter(np.random.choice(index, 1000000, replace=True, p=sample_weights))

def count_mutation(gene_num, mutation_rate):
    return Counter(np.random.binomial(gene_num, mutation_rate, 1000000))

def sum_counter(count_list):
    result = count_list[0]
    if len(count_list) == 1:
        return result
    else:
        for c in count_list[1:]:
            result += c
        return result


def main():
    #parameters
    parser = argparse.ArgumentParser(description="A model to simulate cancer progression")
    parser.add_argument('-a', '--alpha', default=0.0015, type=float, dest='alpha', help='overall growth factor (default: 0.0015)')
    parser.add_argument('-c', '--initCellNum', default=1000000, type=int, dest='init_cell_num', metavar='INIT_CELL_NUM', help='initial population size (default: 1e6)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-M', '--mutRate', type=float, dest='mutation_rate', metavar='MUT_RATE', help='fixed mutation rate (default: 1e-7)')
    group.add_argument('-m', '--mutRange', nargs=2, type=float, dest='mutation_range', metavar=('MUT_RATE_RANGE_START', 'MUT_RATE_RANGE_END'), help="the range of random mutation rate (e.g. 1e-7 1e-5)")
    parser.add_argument('-d', '--driverGeneNum', default=100, type=int, dest='driver_gene_num', metavar='DRIVER_GENE_NUM', help='the number of driver genes (default: 100)')
    parser.add_argument('-g', '--maxMutGeneNum', default=20, type=int, dest='max_mutation_gene_num', metavar='MAX_MUT_GENE_NUM', help='the number of mutation genes needed to progress to cancer(default: 20)')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('-S', '--selAdv', type=float, dest='mutation_selective_advantage', metavar="MUT_SEL_ADV", help="mutation selective advantage (default: 0.01)")
    group1.add_argument('-s', '--selAdvRange', nargs=2, type=float, dest='sel_adv_range', metavar=('SEL_ADV_RANGE_START','SEL_ADV_RANGE_END'), help='the range of selective advantage (e.g. 0.01 0.05)')
    parser.add_argument('-p', '--processNum', default=1, type=int, dest='process_num', metavar='PROCESS_NUM', help="the number of process (default: 1)")
    parser.add_argument('-sp', '--subpopulation', default=1, type=int, dest='subpop', metavar='SUB_POP_NUM', help='the number of subpopulation (default: 1)')
    parser.add_argument('-o', '--outputDir', required=True, dest='out_dir', metavar="OUT_DIR", help="output directory")
    args = parser.parse_args()

    #check output directory
    if os.path.exists(args.out_dir):
        os.rmdir(args.out_dir)
        os.makedirs(args.out_dir)
    else:
        os.makedirs(args.out_dir)

    #fixed mutation rate or not (default: 1e-7)
    if args.mutation_rate:
        fixed_mut_rate = True
        mutation_rate = args.mutation_rate
    elif args.mutation_range:
        fixed_mut_rate = False
        mutation_range = args.mutation_range
    else:
        fixed_mut_rate = True
        mutation_rate = 1e-7

    #fixed selective advantage or not (default: 0.01)
    if args.mutation_selective_advantage:
        fixed_sel_adv = True
        mutation_selective_advantage = args.mutation_selective_advantage
    elif args.sel_adv_range:
        fixed_sel_adv = False
        sel_adv_range = args.sel_adv_range
    else:
        fixed_sel_adv = True
        mutation_selective_advantage = 0.01

    #subpopulation number
    init_sub_pop_size = round(args.init_cell_num/args.subpop)
    sub_pop_sizes = [np.array([init_sub_pop_size for i in range(args.subpop)])]

    #count generation
    generation = 0

    #init record
    sub_pop_records = []
    for i in range(args.subpop):
        sub_pop_records.append([np.array([init_sub_pop_size] + [0 for i in range(args.max_mutation_gene_num)])])

    overall_records = [[sum(sub_pop_sizes[0])] + [0 for i in range(args.max_mutation_gene_num)]]

    #record time
    start_time = time.time()
    flag = 0

    #index for sampling
    index = np.array(range(args.max_mutation_gene_num+1))
    #begin loop
    while(True):
        #determine the mutation rate and selective advantage for this generation
        if fixed_mut_rate:
            this_gen_mut_rate = mutation_rate
        else:
            this_gen_mut_rate = random.uniform(mutation_range[0], mutation_range[1])
        if fixed_sel_adv:
            this_gen_sel_adv = mutation_selective_advantage
        else:
            this_gen_sel_adv = random.uniform(sel_adv_range[0], sel_adv_range[1])


        #determine the overall population size
        #sum the population size of each sub subpopulation
        current_record = overall_records[generation]
        current_pop_size = sum(current_record)
        x = current_record/current_pop_size
        w = fitness(this_gen_sel_adv, args.max_mutation_gene_num, x)
        next_pop_size = round((1 + args.alpha*sum(x*w))*current_pop_size)


        #calculate the proportion of each subpopulation size in the next generation
        mutation_number = np.array([sum(index*r[generation]) for r in sub_pop_records])
        sum_mut = sum(mutation_number)
        # print(mutation_number)
        if sum_mut > 0:
            sub_pop_size = np.round(mutation_number/sum_mut*next_pop_size)
        else:
            sub_pop_size = np.array([round(next_pop_size/args.subpop) for i in range(args.subpop)])
        sub_pop_sizes.append(sub_pop_size)
        # print(sub_pop_size)

        #begin simulation for each subpopulation
        for subpi in range(args.subpop):
            current_record = sub_pop_records[subpi][generation]
            Nt = sum(current_record)
            x = current_record/Nt
            w = fitness(this_gen_sel_adv, args.max_mutation_gene_num, x)
            Nt1 = sub_pop_size[subpi]
            sample_weights = weights(args.driver_gene_num, args.max_mutation_gene_num, this_gen_mut_rate, w, x)

            #sample
            q, r = divmod(int(Nt1), 1000000)
            try:
                counts = Counter(np.random.choice(index, r, replace=True, p=sample_weights))
            except ValueError:
                residue = 1 - sum(sample_weights)
                sample_weights[-1] += residue
                counts = Counter(np.random.choice(index, r, replace=True, p=sample_weights))
            if q > 0:
                count_list = []
                def log_count(x):
                    count_list.append(x)
                pool = Pool(args.process_num)
                for i in range(q):
                    pool.apply_async(count_sampling, args = (index, sample_weights), callback = log_count)
                pool.close()
                pool.join()
                counts += sum_counter(count_list)

            next_record = np.array([0 for i in range(args.max_mutation_gene_num+1)])
            for j in range(args.max_mutation_gene_num+1):
                next_record[j] = counts[j]

            #mutate
            for j in range(args.max_mutation_gene_num-1,-1,-1):
                cell_num = counts[j]
                gene_num = args.driver_gene_num - j

                q, r = divmod(cell_num, 1000000)
                add_counts = Counter(np.random.binomial(gene_num, this_gen_mut_rate, r))
                if q > 0:
                    add_count_list = []
                    def log_add_count(x):
                        add_count_list.append(x)
                    pool = Pool(args.process_num)
                    for i in range(q):
                        pool.apply_async(count_mutation, args = (gene_num, this_gen_mut_rate), callback = log_add_count)
                    pool.close()
                    pool.join()
                    add_counts += sum_counter(add_count_list)

                for k in add_counts:
                    if k > 0:
                        try:
                            next_record[j+k] += add_counts[k]
                            next_record[j] -= add_counts[k]
                        except IndexError:
                            if j+k > args.max_mutation_gene_num:
                                flag = 1
                                print("should finish")
                                break
                            else:
                                print("ERROR")

            sub_pop_records[subpi].append(next_record)

        # finish all the subpopulation
        # add generation
        generation += 1

        #log the overall record
        temp = []
        for i in range(args.max_mutation_gene_num+1):
            temp.append(sum([s[generation][i] for s in sub_pop_records]))
        overall_records.append(temp)
        

        #log
        c_time = time.time() - start_time
        logout = [" %ss " % (c_time), str(generation), str(next_pop_size)]+ list(map(str,sub_pop_size))
        print("\t".join(logout))

        #check end
        if temp[args.max_mutation_gene_num] > 0 or flag == 1:
            break

    #output overall growth record
    out_overall_csv_path = os.path.join(args.out_dir, "overall_mutation_log.csv")
    with open(out_overall_csv_path,"w") as fout:
        csvwriter = csv.writer(fout)
        csvwriter.writerows(overall_records)
    #output subpopulation size
    out_subpop_csv_path = os.path.join(args.out_dir, "subpop_growth_log.csv")
    with open(out_subpop_csv_path,"w") as fout:
        csvwriter = csv.writer(fout)
        csvwriter.writerows(sub_pop_sizes)
    #output growth for each subpopulation
    for i in range(args.subpop):
        out_subpop_mt_csv_path = os.path.join(args.out_dir, "subpop_mutation_log-"+str(i+1)+".csv")
        with open(out_subpop_mt_csv_path,"w") as fout:
            csvwriter = csv.writer(fout)
            csvwriter.writerows(sub_pop_records[i])

if __name__ == "__main__":
    main()
