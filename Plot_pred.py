import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def plot_CPA_KPA():
    Terms = ['CPA Attack', 'KPA Attack']
    for m in range(len(Terms)):
        Eval = np.load('Eval_attack.npy', allow_pickle=True)[m]
        learnper = [10, 20, 30, 40, 50]

        plt.plot(learnper, Eval[:, 0], color='aqua', linewidth=3, marker='h', markerfacecolor='#13EAC9',
                 markersize=14,
                 label="DES")
        plt.plot(learnper, Eval[:, 1], color='#800080', linewidth=3, marker='h', markerfacecolor='#7E1E9C',
                 markersize=14,
                 label="AES")
        plt.plot(learnper, Eval[:, 2], color='#FFA500', linewidth=3, marker='h', markerfacecolor='#F97306',
                 markersize=14,
                 label="ECC")
        plt.plot(learnper, Eval[:, 3], color='#00008B', linewidth=3, marker='h', markerfacecolor='#030764',
                 markersize=14,
                 label="PLCM")
        plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
                 label="MRP-WPA-PLCM")
        x = [10, 20, 30, 40, 50]
        labels = ['1', '2', '3', '4', '5']
        plt.xticks(x, labels)

        plt.xlabel('Cases')
        plt.ylabel('Correlation Coefficient')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Algorithm_%s.png" % (Terms[m])
        plt.savefig(path1)
        plt.show()


def plot_ENC_DEC_CONS_MEM_1():
    Terms = ['Total Consumption time', 'Memory Size', 'Encryption time', 'Decryption time']
    for m in range(len(Terms)):
        Eval = np.load('Eval_time.npy', allow_pickle=True)[m]
        learnper = [1, 2, 3, 4, 5]
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        X = np.arange(Eval.shape[0])
        ax.bar(X + 0.00, Eval[:, 5], color='#FFA500', width=0.10, label="DES")
        ax.bar(X + 0.10, Eval[:, 6], color='#FF00FF', width=0.10, label="AES")
        ax.bar(X + 0.20, Eval[:, 7], color='#A52A2A', width=0.10, label="ECC")
        ax.bar(X + 0.30, Eval[:, 8], color='#0000FF', width=0.10, label="PLCM")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="MRP-WPA-PLCM")
        # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))

        labels = ['5', '10', '15', '20', '25']
        plt.xticks(X, labels)
        plt.xlabel('BLOCK SIZE')
        plt.ylabel(Terms[m])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results/%s.png" % (Terms[m])
        plt.savefig(path)
        plt.show()


def plot_KEY():
    Eval = np.load('key.npy', allow_pickle=True)
    learnper = [10, 20, 30, 40, 50]
    plt.plot(learnper, Eval[:, 0], color='aqua', linewidth=3, marker='h', markerfacecolor='#13EAC9',
             markersize=14,
             label="DES")
    plt.plot(learnper, Eval[:, 1], color='#800080', linewidth=3, marker='h', markerfacecolor='#7E1E9C',
             markersize=14,
             label="AES")
    plt.plot(learnper, Eval[:, 2], color='#FFA500', linewidth=3, marker='h', markerfacecolor='#F97306',
             markersize=14,
             label="ECC")
    plt.plot(learnper, Eval[:, 3], color='#00008B', linewidth=3, marker='h', markerfacecolor='#030764',
             markersize=14,
             label="PLCM")
    plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="MRP-WPA-PLCM")

    plt.xlabel('Case')
    plt.ylabel('Correlation Coefficient')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/KEY.png"
    plt.savefig(path1)
    plt.show()


def Plot_alg():
    Terms = ['Entropy', 'Throughput', 'Avalanche Effect', 'Bit Error Rate (BER)', 'Memory usage (MB)', 'Ciphertext']
    for m in range(len(Terms)):
        Eval = np.load('Eval_encry.npy', allow_pickle=True)[m]
        learnper = [1, 2, 3, 4, 5]
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        X = np.arange(Eval.shape[0])
        ax.bar(X + 0.00, Eval[:, 5], color='#FFA500', width=0.10, label="DES")
        ax.bar(X + 0.10, Eval[:, 6], color='#FF00FF', width=0.10, label="AES")
        ax.bar(X + 0.20, Eval[:, 7], color='#A52A2A', width=0.10, label="ECC")
        ax.bar(X + 0.30, Eval[:, 8], color='#0000FF', width=0.10, label="PLCM")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="MRP-WPA-PLCM")
        # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))

        labels = ['5', '10', '15', '20', '25']
        plt.xticks(X, labels)
        plt.xlabel('BLOCK SIZE')
        plt.ylabel(Terms[m])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results/%s.png" % (Terms[m])
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    Plot_alg()
    plot_KEY()
    plot_ENC_DEC_CONS_MEM_1()
    plot_CPA_KPA()
