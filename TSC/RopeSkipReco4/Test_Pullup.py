from tools import TSC_Pullup as TSC_PU

tsc = TSC_PU.TSC("refer/", 31, "refer/ytxs.csv", 31)
tsc.count_peak()