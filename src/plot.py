import matplotlib.pyplot as plt

# ===== CartPole-v0 Trials =====

cp_v0 = [
    {
        "arch": "8x8",
        "lr": [2.5e-3, 5e-3, 7.5e-3, 1e-2],
        "boost": [0, 2.57, 1.53, 11.4]
    },
    {
        "arch": "16x16",
        "lr": [2.5e-3, 5e-3, 7.5e-3, 1e-2],
        "boost": [0, 2.45, 5.67, 13.2]
    },
    {
        "arch": "32x32",
        "lr": [2.5e-3, 5e-3, 7.5e-3, 1e-2],
        "boost": [0, 13.4, 29.8, 9.6]
    },
    {
        "arch": "48x48",
        "lr": [1e-3, 2.5e-3, 5e-3, 7.5e-3],
        "boost": [0, 6.44, 13.7, 17.4]
    },
    {
        "arch": "64x64",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 0, 11.2, 21.2]
    },
    {
        "arch": "80x80",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 0, 15.9, 17.2]
    },
    {
        "arch": "96x96",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 0, 3.85, 18.6]
    },
    {
        "arch": "112x112",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 0, 28.9, 41.5]
    },
    {
        "arch": "128x128",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 0, 3.04, 15.8]
    }
]

cp_v1 = [
    {
        "arch": "16x16",
        "lr": [2.5e-3, 5e-3, 7.5e-3, 1e-2],
        "boost": [2.77, 5.7, 16.4, 17.8]
    },
    {
        "arch": "32x32",
        "lr": [1e-3, 2.5e-3, 5e-3, 7.5e-3],
        "boost": [0, 11.2, 30.8, 39.8]
    },
    {
        "arch": "48x48",
        "lr": [1e-3, 2.5e-3, 5e-3, 7.5e-3],
        "boost": [17.9, 27.3, 16.5, 14.2]
    },
    {
        "arch": "64x64",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 14.8, 33.3, 66.9]
    },
    {
        "arch": "80x80",
        "lr": [5e-4, 7.5e-4, 1e-3, 2.5e-3],
        "boost": [0, 0, 22.8, 18.3]
    },
    {
        "arch": "96x96",
        "lr": [5e-4, 7.5e-4, 1e-3, 2.5e-3],
        "boost": [1.7, 0, 2.01, 42.7]
    },
    {
        "arch": "112x112",
        "lr": [5e-4, 7.5e-4, 1e-3, 2.5e-3],
        "boost": [0, 9.96, 18.4, 16.8]
    },
    {
        "arch": "128x128",
        "lr": [5e-4, 7.5e-4, 1e-3, 2.5e-3],
        "boost": [0, 3.67, 24.2, 40.7]
    }

]

cp_srl_v0 = [
    {
        "SRL samples": "SRL samples: 0",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-2],
        "boost": [0, 0, 11.2, 21.2]
    },
    {
        "SRL samples": "SRL samples: 1k",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-2],
        "boost": [0, 1.0, 3.8, 8.5]
    },
    {
        "SRL samples": "SRL samples: 10k",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-2],
        "boost": [0, 0, 0, 0.6]
    },
]

cp_srl_v1 = [
    {
        "SRL samples": "SRL samples: 0",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 14.8, 33.3, 66.9]
    },
    {
        "SRL samples": "SRL samples: 1k",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [1.4, 5.5, 12.8, 24.6]
    },
    {
        "SRL samples": "SRL samples: 10k",
        "lr": [7.5e-4, 1e-3, 2.5e-3, 5e-3],
        "boost": [0, 0, 1.2, 4.5]
    },
]

for idx, item in enumerate(cp_srl_v1):
    plt.plot(item["lr"], item["boost"], label=item["SRL samples"])
plt.xscale('log')
plt.xlabel('Learning Rate (Log scale)')
plt.ylabel('Entropy Thresholding Sample Efficiency Boost (%)')
plt.legend(loc='best')
plt.show()


