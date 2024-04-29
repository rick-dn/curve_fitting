from matplotlib import pyplot as plt

x = ['gbm_4', 'astro_3', 'astro_3', 'gbm_4', 'astro_2', 'oligo_3', 'astro_3', 'gbm_4', 'oligo_3', 'oligo_3', 'gbm_4', 'gbm_4', 'astro_3', 'control', 'control', 'control', 'control']
fl_int_single = [59.07,	57.32, 56.11, 95.45, 67.17,	35.62, 46.02, 82.83, 75.26, 40.96, 83.05, 63.37, 61.31, 32.85, 25.88, 22.91, 19.3]
fl_conc_single = [22.76, 10.68, 14.58, 48.41, 15.09, 5.22, 7.09, 44.3, 39.41, 5.72, 49.04, 24.09, 17.21, 5.24, 3.23, 2.14, 1.82]

fl_int_joint = [63.5, 57.26, 77.83, 94.33, 69.18, 36.84, 46.17, 81.98, 75.74, 45.61, 80.65, 60.87, 60.56, 34.3, 20.77, 22.9, 18.98]
fl_conc_joint = [33.5, 10.66, 31.03, 66.76, 15.85, 4.92, 7.19, 46.92, 38.99, 7.24, 55.6, 24.42, 17.1, 5.32, 2.64, 2.13, 1.83]

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)

# fl_int_single
axis[0, 0].scatter(x, fl_int_single)
axis[0, 0].set_title("fl_int_single")

# fl_int_joint
axis[0, 1].scatter(x, fl_int_joint)
axis[0, 1].set_title("fl_int_joint")

# fl_conc_single
axis[1, 0].scatter(x, fl_conc_single)
axis[1, 0].set_title("fl_conc_single")

# fl_conc_joint
axis[1, 1].scatter(x, fl_conc_joint)
axis[1, 1].set_title("fl_conc_joint")

plt.show()
