from itertools import cycle
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    # ax = plt.subplot()
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Confusion Matrix')
    ax = fig.add_axes([0.15, 0.18, 0.7, 0.8])
    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1), )
    sns.heatmap(cm, annot=True, fmt='g',
                ax=ax)
    ax.xaxis.set_ticklabels(['Mild', 'Moderate', 'No \n DR', 'Proliferate \n DR', 'Severe'])
    ax.yaxis.set_ticklabels(['Mild', 'Moderate', 'No \n DR', 'Proliferate \n DR', 'Severe'])
    plt.ylabel('Actual', fontsize=10, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10, fontweight='bold')
    path = "./Results/Confusion.png"
    plt.savefig(path)
    plt.show()


def Plot_ROC():
    lw = 2
    cls = ['IR-CNN', 'FLNN', 'DBN', 'CNN', 'MRP-WPA-MWFF-CNN']
    colors1 = cycle(["#65fe08", "#4e0550", "#f70ffa", "#a8a495", "#004577", ])
    for i, color in zip(range(5), colors1):  # For all classifiers
        Predicted = np.load('roc_score.npy', allow_pickle=True)[i].astype('float')
        Actual = np.load('roc_act.npy', allow_pickle=True)[i].astype('int')
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label="{0}".format(cls[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=10, fontweight='bold')
    plt.title("ROC Curve", fontsize=10, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 10, 'weight': 'bold'})
    path1 = "./Results/roc.png"
    plt.savefig(path1)
    plt.show()


def Plot_Results_Feature():
    Eval = np.load('Eval_ALL_Feat.npy', allow_pickle=True)
    Terms = np.asarray(
        ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score',
         'MCC', 'FOR', 'PT', 'BA', 'FM', 'BM', 'MK', 'PLHR', 'Lrminus', 'DOR', 'Prevalence', 'TS'])
    Graph_Term = np.array([0, 3, 4, 5, 6]).astype(int)
    value = Eval[2, :, 4:]

    Eval = np.load('Eval_ALL_Feat.npy', allow_pickle=True)
    BATCH = [1, 2, 3, 4]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(4)
        ax.bar(X + 0.00, Graph[:, 5], color='#0804f9', edgecolor='k', width=0.12, hatch="..", label="IR-CNN")
        ax.bar(X + 0.12, Graph[:, 6], color='#b1d1fc', edgecolor='k', width=0.12, hatch="..", label="FLNN")
        ax.bar(X + 0.23, Graph[:, 7], color='#be03fd', edgecolor='k', width=0.12, hatch='..', label="DBN")
        ax.bar(X + 0.36, Graph[:, 8], color='lime', edgecolor='k', width=0.12, hatch="..", label="CNN")
        ax.bar(X + 0.48, Graph[:, 9], color='k', edgecolor='w', width=0.12, hatch="//", label="MRP-WPA-MWFF-CNN")
        plt.xticks(X + 0.25, ('R-ViT', 'VGG16', 'MobileNet', '    Weighted  Feature'), fontsize=10, fontweight='bold')
        plt.xlabel('Feature', fontsize=10, fontweight='bold')
        plt.ylabel(Terms[Graph_Term[j]], fontsize=10, fontweight='bold')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True,
                   prop={'size': 10, 'weight': 'bold'})
        path = "./Results/Feature_%s_Med.png" % ((Terms[Graph_Term[j]]))
        plt.savefig(path)
        plt.show()


def plot_results_conv():
    conv = np.load('Fitness.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['DOA-MWFF-CNN', 'COA-MWFF-CNN', 'DGO-MWFF-CNN', 'WPA-MWFF-CNN', 'MRP-WPA-MWFF-CNN']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print(
        '--------------------------------------------------- Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='DeepPink', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
             label='DOA-MWFF-CNN')
    plt.plot(iteration, conv[1, :], color='cyan', linewidth=3, marker='*', markerfacecolor='green', markersize=12,
             label='COA-MWFF-CNN')
    plt.plot(iteration, conv[2, :], color='DeepPink', linewidth=3, marker='*', markerfacecolor='Yellow',
             markersize=12,
             label='DGO-MWFF-CNN')
    plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta', markersize=12,
             label='WPA-MWFF-CNN')
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black', markersize=12,
             label='MRP-WPA-MWFF-CNN')
    plt.xlabel('Iteration', fontsize=10, fontweight='bold')
    plt.ylabel('Cost Function', fontsize=10, fontweight='bold')
    plt.legend(loc=1, prop={'size': 10, 'weight': 'bold'})
    path = "./Results/Conv.png"
    plt.savefig(path)
    plt.show()


def Plot_Results_method():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Steps = [1, 2, 3, 4, 5]
    Classifier = [4, 8, 16, 32, 48]

    # Define a list of colors
    # colors = plt.cm.tab20.colors  # You can use other colormaps or define your own colors
    colors = ['#A0522D', '#C20078', '#0343DF', '#FE420F', '#054907']

    Eval = np.load('Eval_ALL_batch.npy', allow_pickle=True)

    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[1], Eval.shape[0]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[l, k] = Eval[k, l, Graph_Term[j] + 4]

        # Plotting
        plt.figure(figsize=(7, 5))
        data = Graph[5:, :]
        bar_patches = []
        for idx, (row, classifier) in enumerate(zip(data, Classifier)):
            bar_positions = np.arange((idx * 5) + idx, (idx + 1) * 5 + idx)
            for bar_pos, value, color in zip(bar_positions, row, colors[:len(row)]):
                bar_patch = plt.bar(bar_pos, value, width=0.7, color=color)
                bar_patches.append(bar_patch)
                # Add legend entry for each bar
                # plt.text(bar_pos, value + 0.01, f'{classifier}', ha='center', va='bottom')

        plt.xlabel('Networks', fontsize=10, fontweight='bold')
        plt.ylabel(Terms[Graph_Term[j]], fontsize=10, fontweight='bold')
        # Adjust x-axis labels
        plt.xticks(ticks=np.arange(0, Eval.shape[2], Eval.shape[0] + 1) + round(data.shape[0] / 2),
                   labels=['IR-CNN', 'FLNN', 'DBN', 'CNN', '        MRP-WPA-MWFF-CNN'], fontsize=10, fontweight='bold')

        # Create legend
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(labels=[f'Batch Size - {clf}' for clf in Classifier],
                   loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True,
                   prop={'size': 10, 'weight': 'bold'})

        path1 = "./Results/Batch_%s_Method.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def Plot_Results():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt', 'CSI',
             'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Eval = np.load('Eval_ALL_batch.npy', allow_pickle=True)
    Steps = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]

        fig = plt.figure()
        fig.canvas.manager.set_window_title('Steps Per Epochs - Algorithm')
        X = np.arange(Graph.shape[0])
        plt.plot(Steps, Graph[:, 0], color='#A0522D', linewidth=4, marker='>', markerfacecolor='k',
                 markersize=12,
                 label="DOA-MWFF-CNN")
        plt.plot(Steps, Graph[:, 1], color='#C20078', linewidth=4, marker='>', markerfacecolor='k',
                 markersize=12,
                 label="COA-MWFF-CNN")
        plt.plot(Steps, Graph[:, 2], color='#0343DF', linewidth=4, marker='>', markerfacecolor='k',
                 markersize=12,
                 label="DGO-MWFF-CNN")
        plt.plot(Steps, Graph[:, 3], color='#FE420F', linewidth=4, marker='>', markerfacecolor='k', markersize=12,
                 label="WPA-MWFF-CNN")
        plt.plot(Steps, Graph[:, 4], color='k', linewidth=4, marker='>', markerfacecolor='white', markersize=12,
                 label="MRP-WPA-MWFF-CNN")
        plt.xlabel('Batch Size', fontsize=10, fontweight='bold')
        plt.xticks(X + 1, ('4', '8', '16', '32', '48'), fontsize=10, fontweight='bold')
        plt.ylabel(Terms[Graph_Term[j]], fontsize=10, fontweight='bold')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True, prop={'size': 10, 'weight': 'bold'})
        path = "./Results/Batch_%s_Alg.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def Plot_table():
    Eval = np.load('Eval_All_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt', 'CSI',
             'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Term = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(int)
    Graph_Term_Array = np.array(Graph_Term)
    Algorithm = ['Activation Function', 'DOA-MWFF-CNN', 'COA-MWFF-CNN', 'DGO-MWFF-CNN', 'WPA-MWFF-CNN', 'MRP-WPA-MWFF-CNN']
    Classifier = ['Activation Function', 'IR-CNN', 'FLNN', 'DBN', 'CNN', 'MRP-WPA-MWFF-CNN']
    variation = ['Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid']
    value = Eval[:, :, 4:]
    Table = PrettyTable()
    Table.add_column(Algorithm[0], variation[0:])
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], [item for sublist in value[:, j, Graph_Term] for item in sublist])
    print('-------------------------------------------------- Activation Function - Algorithm Comparison -',
          'Accuracy',
          '--------------------------------------------------')
    print(Table)
    Table = PrettyTable()
    Table.add_column(Classifier[0], variation[0:])
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1],
                         [item for sublist in value[:, len(Algorithm) + j - 1, Graph_Term] for item in sublist])
    print('--------------------------------------------------- Activation Function - Classifier Comparison -',
          'Accuracy',
          '--------------------------------------------------')
    print(Table)


if __name__ == '__main__':
    # Plot_Confusion()
    # Plot_ROC()
    # plot_results_conv()
    # Plot_Results_method()
    # Plot_Results()
    # Plot_table()
    Plot_Results_Feature()
