import numpy as np
import matplotlib.pyplot as plt


## ==================================================================
## ADHD
if 1:
    # ###layers k=3, c=3
    if 1:
        # data to plot
        n_groups = 2
        acc = [0.626628555, 0.634802788, 0.642009018, 0.643957197]
        f1 = [0.612299067, 0.638820079, 0.6570274, 0.646551233]

        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#F40103',
        label='1-Layer')

        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F9BF01',
        label='2-Layer')
        
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#51FF00',
        label='3-Layer')
        
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#46CCFF',
        label='4-Layer')


        plt.xlabel('Layer Number Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        ax.tick_params(labelsize=12)
        plt.ylim(0.5, max(values.reshape(-1))+0.01)
        # plt.ylim(0.5,1)
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()

    ## ###channels  k=3
    if 0:
        # data to plot
        n_groups = 2
        acc = [0.604079405, 0.613757387, 0.626628555, 0.639434785]
        f1 = [0.642196024, 0.609341892, 0.612299067, 0.634872583]

        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='c=1')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F79F02',
        label='c=2')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F41103',
        label='c=3')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#800201',
        label='c=4')


        ax.tick_params(labelsize=12)
        plt.xlabel('Channel Number Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()

    ## ###kernel 
    if 0:
        # data to plot
        n_groups = 2
        acc = [0.623879797, 0.604079405, 0.641367743, 0.598344454]
        f1 = [0.627973641, 0.642196024,0.65585958, 0.594374737]

        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#1D20FF',
        label='k=2')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#00ffff',
        label='k=3')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#54FFBF',
        label='k=4')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='k=5')


        plt.xlabel('Kernel Size Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        ax.tick_params(labelsize=12)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()



## ==================================================================
## HIV-fMRI
if 0:
    ## ###channels 
    if 1:
        # data to plot
        n_groups = 2
        acc = [0.707070707, 0.621212121, 0.765151515, 0.734848485]
        f1 = [0.7, 0.506349206, 0.759440559, 0.733333333]
        # create plot
        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='c=1')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F79F02',
        label='c=2')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F41103',
        label='c=3')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#800201',
        label='c=4')

        ax.tick_params(labelsize=12)
        plt.xlabel('Channel Number Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()


    ## ###kernel 
    if 1:
        # data to plot
        n_groups = 2
        acc = (0.648989899, 0.707070707, 0.646464646, 0.646464646)
        f1 = (0.63030303, 0.7, 0.6, 0.666666667)

        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#1D20FF',
        label='k=2')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#00ffff',
        label='k=3')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#54FFBF',
        label='k=4')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='k=5')


        plt.xlabel('Kernel Size Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        ax.tick_params(labelsize=12)
        # plt.xlabel('Channel Number Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()


## ==================================================================
## HIV-DTI
if 0:
    ## ###channels  k=4
    if 1:
        # data to plot
        n_groups = 2
        acc = [0.554945055, 0.675824174, 0.5989011, 0.551282053]
        f1 = [0.455128205, 0.723389355, 0.665849672, 0.584401711]

        # create plot
        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 0.8
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='c=1')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F79F02',
        label='c=2')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F41103',
        label='c=3')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#800201',
        label='c=4')

        ax.tick_params(labelsize=12)
        plt.xlabel('Channel Number Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()

    ## ###kernel 
    if 0: 
        # data to plot
        n_groups = 2
        acc = [0.503663004, 0.424908425, 0.554945055, 0.401098901]
        f1 = [0.443627451, 0.462378168, 0.455128205, 0.423351159]

        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#1D20FF',
        label='k=2')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#00ffff',
        label='k=3')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#54FFBF',
        label='k=4')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='k=5')


        plt.xlabel('Kernel Size Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
    

        ax.tick_params(labelsize=12)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()


## ==================================================================
## BP-fMRI
if 0:

    ## ###channels  k=3
    if 0:

        # data to plot
        n_groups = 2
        acc = [0.649305556, 0.577335859,0.452967172,0.536300505]
        f1 = [0.696538644, 0.466936572, 0.389236546, 0.66215867]
        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='c=1')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F79F02',
        label='c=2')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#F41103',
        label='c=3')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#800201',
        label='c=4')


        ax.tick_params(labelsize=12)
        plt.xlabel('Channel Number Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()


    ## ###kernel 
    if 0:

        # data to plot
        n_groups = 2
        acc = [0.452967172, 0.649305556, 0.610220225, 0.515467172]
        f1 = [0.43498818, 0.696538644, 0.609351635, 0.608844389]

        values = np.array([acc, f1])
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 1
        gap = bar_width + 0.05

        rects1 = plt.bar(index, values[:,0], bar_width, edgecolor='black',
        alpha=opacity,
        color='#1D20FF',
        label='k=2')
        rects2 = plt.bar(index+gap, values[:,1], bar_width,edgecolor='black',
        alpha=opacity,
        color='#00ffff',
        label='k=3')
        rects3 = plt.bar(index+2*gap, values[:,2], bar_width,edgecolor='black',
        alpha=opacity,
        color='#54FFBF',
        label='k=4')
        rects4 = plt.bar(index+3*gap, values[:,3], bar_width,edgecolor='black',
        alpha=opacity,
        color='#CFFF2F',
        label='k=5')


        plt.xlabel('Kernel Size Comparison', fontsize=14)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        ax.tick_params(labelsize=12)
        # plt.ylabel('Scores')
        # plt.title('Scores by person')
        # ax.set_xticks([])
        plt.xticks(index + bar_width*2, ('Accuracy', 'F1'))
        plt.legend(loc='upper center', prop={'weight':'regular', 'size':14})

        plt.tight_layout()
        plt.show()




