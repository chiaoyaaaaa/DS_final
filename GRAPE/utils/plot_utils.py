import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def plot_curve(data, plot_file, keys=None, 
                clip=True, label_min=True, label_max=False, label_end=True):
    if not keys:
        keys = data.keys()
    # plt.figure()
    plt.figure(figsize=(10, 10))
    for i, key in enumerate(keys):
        plt.subplot(len(keys),1,i+1)
        if clip:
            limit = 2*np.mean(np.abs(data[key]))
            y = np.clip(data[key],-limit,limit)
        else:
            y = data[key]
        plt.plot(y, linewidth=1.,label=key)
        if 'accuracy' in key:
            plt.plot(np.argmax(data[key]),np.max(data[key]),'o',
                    label="max: {:.3g}".format(np.max(data[key])))
        elif label_min:
            plt.plot(np.argmin(data[key]),np.min(data[key]),'o',
                    label="min: {:.3g}".format(np.min(data[key])))
        if label_end:
            plt.plot(len(data[key])-1,data[key][-1],'o',
                    label="end: {:.3g}".format(data[key][-1]))
        # plt.legend()            
        plt.legend(loc = 1)
    plt.savefig(plot_file)
    plt.close()

def plot_sample(data, plot_file, groups, num_points=20):
    plt.figure()
    for i,keys in enumerate(groups):
        plt.subplot(len(groups),1,i+1)
        for key in keys:
            interval = int(data[key].shape[0]/num_points)
            y = data[key][::interval]
            plt.plot(y, linewidth=1., label=key)
        plt.legend()
    plt.savefig(plot_file)
    plt.close()


# add
# For task 3 (unsupervised learning)
def plot_confusion_matrix(pred_label, true_label, class_num, plot_file):
    mat = confusion_matrix(pred_label, true_label)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
    xticklabels=list(range(class_num)),
    yticklabels=list(range(class_num)))
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    plt.savefig(plot_file)
    plt.close()

def plot_kmeans(X, centers, pred_label, true_label, class_num, plot_file):
    pca = PCA(2)
    df = pca.fit_transform(np.concatenate([X, centers]))
    # sns.scatterplot(x = df[:-class_num, 0],
    #                 y = df[:-class_num, 1],
    #                 hue = pred_label,
    #                 # style = true_label,
    #                 palette = "deep",
    #                 legend = True)
    plt.figure(figsize=(10, 8))
    plt.scatter(df[:-class_num, 0][pred_label == 0], 
                df[:-class_num, 1][pred_label == 0], 
                color = 'royalblue',
                label = '0')
    plt.scatter(df[:-class_num, 0][pred_label == 1], 
                df[:-class_num, 1][pred_label == 1], 
                color = 'yellowgreen',
                label = '1')
    plt.scatter(df[:-class_num, 0][~(pred_label == true_label)],
                df[:-class_num, 1][~(pred_label == true_label)],
                color = 'red', 
                facecolors='none', 
                label = 'False')
    plt.legend()
    plt.plot(df[-class_num:, 0],
            df[-class_num:, 1],
            '+',
            markersize=15,
            color = 'red')

    plt.savefig(plot_file)
    plt.close()


# add
def plot_contrastive(embd, df_y, epoch, plot_file):
    pca = PCA(2)
    x_pca = pca.fit_transform(embd.cpu().detach().numpy())

    plt.scatter(x_pca[df_y == 0][:, 0], 
                x_pca[df_y == 0][:, 1], 
                color = 'royalblue',
                label = '0')  
    plt.scatter(x_pca[df_y == 1][:, 0], 
                x_pca[df_y == 1][:, 1],
                color = 'yellowgreen',
                label = '1')  
    plt.title('Epoch: {}'.format(epoch))  
    plt.xlabel('PCA 1') 
    plt.ylabel('PCA 2') 
    plt.legend()
    plt.savefig(plot_file)
    plt.show()