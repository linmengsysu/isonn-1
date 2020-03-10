import numpy as np
import pickle
import math 

from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from scipy.special import softmax as softmax

# from matplotlib.colors import LinearSegmentedColormap

# # Create custom colormaps
# cdict = {'red': ((0.0, 1.0, 1.0),   # Full red at the first stop
#                  (0.5, 0.0, 0.0),   # No red at second stop
#                  (1.0, 1.0, 1.0)),  # Full red at final stop
#         #
#         'green': ((0.0, 0.0, 0.0),  # No green at all stop
#                  (0.5, 0.0, 0.0),   # 
#                  (1.0, 0.0, 0.0)),  # 
#         #
#         'blue': ((0.0, 0.0, 0.0),   # No blue at first stop
#                  (0.5, 1.0, 1.0),   # Full blue at second stop
#                  (1.0, 0.0, 0.0))}  # No blue at final stop

# cmap = LinearSegmentedColormap('Rd_Bl', cdict, 256)


# top = cm.get_cmap('Reds_r', 128)
# bottom = cm.get_cmap('Blues', 128)

# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128))))
# newcmp = ListedColormap(newcolors, name='RedBlue')

colormap = 'autumn'
pixels = 10

def load_dataset():
    data_dir = '../../data/kdd17/ADHD_fMRI_3_fold'
    fold_count = 3

    print('load_dataset')
    filename = data_dir + '/fold_' + str(fold_count)
    print(filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
   
    train_graph = data['train']['X']
    test_graph = data['test']['X']
    

    (n_graph, hw) = train_graph.shape
    n_H = int(np.sqrt(float(hw)))
    test_graph = np.array(test_graph)

        
    train_graph = train_graph.reshape(n_graph, 1, n_H, n_H)
    test_graph = test_graph.reshape(-1, 1, n_H, n_H)

    return train_graph[0], test_graph[0]

def get_all_P(k):
    n_P = np.math.factorial(k)
    P_collection = np.zeros([n_P, k, k])
    perms = permutations(range(k), k)

    count = 0
    for p in perms:
        for i in range(len(p)):
            P_collection[count, i, p[i]] = 1
        count += 1

    return P_collection

def plot():
    fkernel = 'ADHD_1_layer_k3c4_kernels'
    f = open(fkernel, 'rb')
    kernels = pickle.load(f)
    kernels = np.array([[k.numpy() for k in kernels]])
    # num_cols = kernels[0].shape[0]
    # fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    # for j, ax in enumerate(axs.flat):
    #     ax.imshow(kernels[0][j], cmap="hot")
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.imshow(kernels[0][0], cmap='hot')
    # plt.show()
    return kernels

def kernelsWithP(kernels, permutations):
    kernelsPerm = []
    # print(kernels.shape, kernels.shape, permutations.shape)
    layer = kernels.shape #[c,k,k]
    c, k, k = kernels.shape
    for ker in range(c):

        tmp = np.matmul(permutations, kernels[ker].reshape(1, k, k))
        kperms = np.matmul(tmp, np.transpose(permutations, (0, 2, 1)))
        # print('kperms', kperms.shape)
        kernelsPerm.append(kperms)

    kernelsPerm = np.array(kernelsPerm)
    num_rows = kernelsPerm.shape[0]
    num_cols = 3#kernelsPerm.shape[1]
    # print('kernelsPerm.shape', kernelsPerm.shape)
    fig, axs = plt.subplots(nrows=2, ncols=3)
    # print(axs.shape, axs.flat)
    for j, ax in enumerate(axs.flat):
        # print(j/num_cols, j%num_cols)
        # if j%num_cols == 0:
            # ax.set_title("Kernel %s \n" % (int(j/num_cols)+1), fontsize=12, rotation=0)
        im= ax.imshow(kernelsPerm[0][j%num_cols], cmap=colormap)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # ax._frameon = False
    # fig.text(0, 1, 'Kernel', ha='left', rotation=15, wrap=True)
    # rows = ['Kernel {}'.format(row) for row in [1,2,3,4]] 
    # for ax, row in zip(axs[:,0], rows): 
    #     # print
    #     ax.set_ylabel(row, rotation=90, size='large') 

    fig.colorbar(im,ax=axs, shrink=0.6)
    plt.savefig('case/kernel_layer1.pdf')
    # j = 0
    # for i in range(1, 19):
    #     ax = fig.add_subplot(3,6,i)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     im= ax.imshow(kernelsPerm[int(j/num_cols)][j%num_cols], cmap="viridis")
    #     j+=1
        # ax.set_title('Plot title ' + str(i))

    # plt.tight_layout()
    plt.show()
    plt.clf()
    # del fig
    # del ax
    # del axs
    return kernelsPerm

def simiarity(G, kernelsPerm, layer):
    c, p, k, k = kernelsPerm.shape
    print('kernelsPerm.shape', kernelsPerm.shape)
    b, n_H_prev, n_W_prev = G.shape
    isoGs = []
    isoGperms = []
    n_H = n_H_prev - k + 1
    n_W = n_W_prev - k + 1
    
    for i in range(b):
        isoGb = []
        isoGp = []
        for ker in [2]:
            kernels = kernelsPerm[ker,:,:,:].reshape(p,k,k)
            # print('kernels', kernels.shape)
            isoGperm = np.zeros((p, n_H, n_W))
            isoG = np.zeros((n_H, n_W))
            for h in range(n_H):
                for w in range(n_W):
                    x_slice = G[i, h:h+k, w:w+k]
                    sim = np.linalg.norm(x_slice-kernels,ord='fro',axis=(1,2))
                    # print('sim', sim.shape)
                    isoGperm[:,h,w] = sim
                    isoG[h,w] = min(sim)
            isoGb.append(isoG)
            isoGp.append(isoGperm)
        isoGperms.append(isoGp)
        isoGs.append(isoGb)

    isoGs = np.array(isoGs).squeeze()
    # isoGs = softmax(-isoGs.squeeze(),axis=0)
    isoGperms = np.array(isoGperms).squeeze()

    print('isoGs, isoGperms', isoGs.shape, isoGperms.shape)
    # num_rows = isoGs.shape[0]
    num_cols = isoGs.shape[0]

    # fig, axs = plt.subplots(nrows=1, ncols=num_cols, clear=True)
    # # print(axs.shape, axs.flat)
    # # print(axs.shape, type(axs))
    # for j, ax in enumerate(axs.flat):
    #     # print(j/num_cols, j%num_cols)
    #     im= ax.imshow(isoGs[j%num_cols], cmap=colormap)
    #     # ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    #     ax.set_xlabel('Features with K{}'.format(j+1))
    # fig.colorbar(im,ax=axs, shrink=0.6)
    # # fig.suptitle('Layer {} Simiarity'.format(layer))
    # plt.show()
    for j in range(p):
        fig, ax = plt.subplots(nrows=1, ncols=1, clear=True)
        im= ax.imshow(isoGperms[j,:pixels,:pixels], cmap=colormap)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im,ax=ax, shrink=0.6)
        
        ax.set_title('Features with Permutation {}'.format(j+1))
        plt.savefig('case/sim_k1p{}.pdf'.format(j+1))
        plt.show()
        plt.clf()
        # del fig
        # del ax

    
    for j in range(1):
        fig, ax = plt.subplots(nrows=1, ncols=1, clear=True)
        im= ax.imshow(isoGs[:pixels,:pixels], cmap=colormap)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im,ax=ax, shrink=0.6)
        
        ax.set_title('Features with Optimal Permutation')
        
        plt.savefig('case/sim_k1_optimal.pdf')
        plt.show()
        plt.clf()
        # del fig
        # del ax
    return isoGs




G1, _ = load_dataset()
kernels = plot()
permutations = get_all_P(kernels[0].shape[-1])
kernelsPerm1 = kernelsWithP(kernels[0], permutations)
# G1 = G1.reshape(1,1, G1.shape[-1], G1.shape[-1])
fig, ax = plt.subplots()
im=ax.imshow(G1.squeeze()[:pixels,:pixels], cmap=colormap)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.colorbar(im,ax=ax, shrink=0.6)
# fig.suptitle('Layer {} Orginal'.format(1))
plt.savefig('case/original_brain.pdf')
plt.show()
plt.clf()
# del fig
# del ax

G = simiarity(G1, kernelsPerm1,1)

# print('G2', G.shape)
# num_cols = G.shape[0]
# # fig.clf()
# fig, axs = plt.subplots(nrows=1, ncols=num_cols)
# # print(axs.shape, type(axx))
# for j, ax in enumerate(axs.flat):
#     # print(j/num_cols, j%num_cols)
#     im= ax.imshow(G[j%num_cols], cmap=colormap)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# fig.colorbar(im, ax=axs, shrink=1)
# # fig.suptitle('Layer {} Orginal'.format(2))
# del fig
# del axs
# kernelsPerm2 = kernelsWithP(kernels[1], permutations)
# G = simiarity(G, kernelsPerm2, 2)


# print('G3', G.shape)
# num_cols = G.shape[0]
# # fig.clf()
# fig, axs = plt.subplots(nrows=1, ncols=num_cols)
# # print(axs.shape, type(axx))
# for j, ax in enumerate(axs.flat):
#     # print(j/num_cols, j%num_cols)
#     im= ax.imshow(G[j%num_cols], cmap=colormap)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# # fig.colorbar(im, ax=axs, shrink=0.6)
# # fig.suptitle('Layer {} Orginal'.format(3))
# del fig
# del axs
# kernelsPerm3 = kernelsWithP(kernels[2], permutations)
# G = simiarity(G, kernelsPerm3, 3)



