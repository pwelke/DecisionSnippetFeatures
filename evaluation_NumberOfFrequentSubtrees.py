#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

#dataset adult

#ratio of Frequent Subtrees compared to unpruned: unpruned, sigma 0_0, 0_1, 0_2, 0_3
RF_5_number_FS = [1, 1.1, 1.03, 0.86, 0.69]
RF_10_number_FS = [1, 1.15, 1.33, 1.45, 1.52]
RF_15_number_FS = [1, 1.12, 1.29, 1.36, 1.39]
RF_20_number_FS = [1, 1.09, 1.22, 1.27, 1.28]

x_axis = ['unpruned','sig 0.0','sig 0.1',' sig 0.2','sig 0.3']

plt.plot(x_axis,RF_5_number_FS, 'o-', label='FS_ratio_md_5')
plt.plot(x_axis,RF_10_number_FS, 'o-', label='FS_ratio_md_10')
plt.plot(x_axis,RF_15_number_FS, 'o-', label='FS_ratio_md_15')
plt.plot(x_axis,RF_20_number_FS, 'o-', label='FS_ratio_md_20')
#plt.legend(loc='lower left')
#plt.show()


RF_5_avg_size = [1.29, 1.29, 1.63, 1.65, 1.66]
RF_10_avg_size = [1.73, 1.73, 1.94, 1.98, 2.01]
RF_15_avg_size = [1.89, 1.92, 2.03, 2.07, 2.09]
RF_20_avg_size = [1.93, 2.0, 2.09, 2.12, 2.17]

plt.plot(x_axis,RF_5_avg_size, 'o-', label='avg_size_md_5')
plt.plot(x_axis,RF_10_avg_size, 'o-', label='avg_size_md_10')
plt.plot(x_axis,RF_15_avg_size, 'o-', label='avg_size_md_15')
plt.plot(x_axis,RF_20_avg_size, 'o-', label='avg_size_md_20')
plt.legend(loc='best')
plt.show()
