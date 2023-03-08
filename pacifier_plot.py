import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    positive = [0.95, 0.9065, 0.9029, 0.9, 0.8936]
    negetive = [0.05, 0.0935, 0.0971, 0.1, 0.1064]

    p1 = [329, 312, 311, 227]
    p2 = [329, 313, 314, 228]
    p3 = [326, 311, 309, 226]
    p4 = [333, 316, 315, 230]

    n_groups = 4

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4

    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, p1, bar_width,
        alpha=opacity, color='b',
        error_kw=error_config,
        label='w1')

    rects2 = ax.bar(index + bar_width, p2, bar_width,
                alpha=opacity, color='m',
                error_kw=error_config,
                label='w2')

    rects3 = ax.bar(index + bar_width + bar_width, p3, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='w3')

    rects4 = ax.bar(index + bar_width + bar_width + bar_width, p4, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='w4')

    ax.set_xticks(index + 4 * bar_width / 4)
    ax.set_xticklabels(('732252283', '47684938', '758099411', '235105995'))

    ax.legend()
    plt.xlabel(u"product")
    plt.ylabel(u'sales volume')
	 
    fig.tight_layout()
    plt.savefig('sale.png', dpi=200)
    plt.show()


def little(a, b):
    return a/(a + b), b/(a + b)


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = ["great", "like", "well", "good"]
    num_list1 = [0.8935944944415034, 0.7532258064516129, 0.8064257028112449, 0.8200716845878137]
    num_list2 = [(1 - x) for x in num_list1]
	
    label_list = ["stopped", "disappoint", "disappointment", "poor"]
    num_list2 = [0.7602339181286549, 0.6681034482758621, 0.6666666666666666, 0.6888888888888889]
    num_list1 = [(1 - x) for x in num_list2]

    x = range(len(num_list1))
    rects1 = plt.bar(x, height=num_list1, width=0.45, alpha=0.8, color='red', label="positive")
    rects2 = plt.bar(x, height=num_list2, width=0.45, color='green', label="negetive", bottom=num_list1)
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.xticks(x, label_list)
    plt.xlabel("word")
    plt.legend()
    plt.savefig("bad_probability.jpeg")
    plt.show()
