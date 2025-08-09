import numpy as np
from prettytable import PrettyTable



def plot_results():
    eval = np.load('Eval_ALL_batch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt', 'CSI',
             'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]
    Algorithm = ['TERMS', 'DO', 'RSO', 'DMO', 'CHIO', 'PROPOSED']
    Classifier = ['TERMS', 'LSTM', 'RNN', 'ResNet', 'ARDLSTM-AM', 'PROPOSED']
    value1 = eval[4, :, 4:]

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value1[j, :])
    print('-------------------------------------------------- Batch size - Algorithm Comparison',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 2):
        Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
    Table.add_column(Classifier[5], value1[4, :])
    print('-------------------------------------------------- Batch size - Classifier Comparison',
          '--------------------------------------------------')
    print(Table)



if __name__ == '__main__':
    plot_results()