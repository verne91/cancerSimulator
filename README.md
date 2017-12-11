A model to simulate cancer progression. It supports dyanmic randome parameters and multiple subpopulations.
```
Usage:
python model8.py [-h] [-a ALPHA] [-c INIT_CELL_NUM]
                 [-M MUT_RATE | -m MUT_RATE_RANGE_START MUT_RATE_RANGE_END]                 
                 [-d DRIVER_GENE_NUM] [-g MAX_MUT_GENE_NUM]                
                 [-S MUT_SEL_ADV | -s SEL_ADV_RANGE_START SEL_ADV_RANGE_END]                
                 [-p PROCESS_NUM] [-sp SUB_POP_NUM] -o OUT_DIR
```

-h: print help information

-a: overall growth factor (default: 0.0015)

-c: initial overall population size (default: 1e6)

-M: fixed mutation rate (default: 1e-7)

-m: dynamic random mutation rate range

-d: driver gene number (default: 100)

-g: the number of mutation genes needed to progress to cancer (default: 20)

-S: fixed selective advantage (default: 0.01)

-s: dynamic random selective advantage range

-p: process number for multiprocessing (default: 1)

-sp: subpopulation number (default: 1)

-o: path of output directory (only required parameter)
