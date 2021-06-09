import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

shape_vec = ['o', 'o', 'o', 'o', 'o', 'o', '^', '^',  '^', '^',  '^', '^', 's', 's', 's', 's', 's', 's']
edgecolor_vec = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
facecolor_vec = ['r', 'g', 'b', 'none', 'none', 'none', 'r', 'g', 'b', 'none', 'none', 'none',
                 'r', 'g', 'b', 'none', 'none', 'none']

# Panel 1
cluster_mean = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [1.5, 1.5], [2.5, 1.5], [3.5, 1.5],
                [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [1.5, 2.5], [2.5, 2.5], [3.5, 2.5],
                [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [1.5, 3.5], [2.5, 3.5], [3.5, 3.5]])
np.random.seed(1)
np.random.shuffle(cluster_mean)
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

z = 0.2 * np.array([[0, 0], [-1, 0], [1, 0], [-0.5, 1], [0.5, 1], [-0.5, -1], [0.5, -1]])

for i in range(18):
    # r = 0.33 * np.random.uniform(0, 1, 30); theta = 2 * np.pi * np.random.uniform(0, 1, 30)
    # x = cluster_mean[i, 0] + r * np.sin(theta); y = cluster_mean[i, 1] + r * np.cos(theta)
    x = cluster_mean[i, 0] + z[:, 0]
    y = cluster_mean[i, 1] + z[:, 1]

    s = shape_vec[i]
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]

    # for xp, yp, m in zip(x, y, cluster):
    #     ax.scatter([xp], [yp], marker=m)
    ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)

    circle = plt.Circle((cluster_mean[i, 0], cluster_mean[i, 1]), 0.33, color='black', fill=False, ls='--')
    ax.add_patch(circle)
plt.show()

# Panel 2
z = 0.14 * np.array([[-1, 4], [0, 4], [1, 4],
                    [-1.5, 3], [-0.5, 3], [0.5, 3], [1.5, 3],
                    [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
                    [-2.5, 1], [-1.5, 1], [-0.5, 1], [0.5, 1], [1.5, 1], [2.5, 1],
                    [-3, 0], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0], [3, 0],
                    [-2.5, -1], [-1.5, -1], [-0.5, -1], [0.5, -1], [1.5, -1], [2.5, -1],
                    [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2],
                    [-1.5, -3], [-0.5, -3], [0.5, -3], [1.5, -3],
                    [-0.5, -4], [0.5, -4]])
z[:, 1] = 0.68 * z[:, 1]
cluster_mean = np.array([[0, 0], [0, 1], [0.9, 0.5]])

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

edgecolor_vec = ['r', 'g', 'b', 'r', 'g', 'b']
facecolor_vec = ['r', 'g', 'b', 'none', 'none', 'none']

s = 'o'
z0 = z
np.random.shuffle(z0)
for i in range(6):
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[0, 0] + z0[7*i+j, 0]
        y = cluster_mean[0, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((cluster_mean[0, 0], cluster_mean[0, 1]), 0.48, color='black', fill=False, ls='--')
ax.add_patch(circle)

s = '^'
z0 = z
np.random.shuffle(z0)
for i in range(6):
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[1, 0] + z0[7*i+j, 0]
        y = cluster_mean[1, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((cluster_mean[1, 0], cluster_mean[1, 1]), 0.48, color='black', fill=False, ls='--')
ax.add_patch(circle)

s = 's'
z0 = z
np.random.shuffle(z0)
for i in range(6):
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[2, 0] + z0[7*i+j, 0]
        y = cluster_mean[2, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((cluster_mean[2, 0], cluster_mean[2, 1]), 0.48, color='black', fill=False, ls='--')
ax.add_patch(circle)
plt.show()


# Panel 3
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

shape_vec = ['o', 'o', '^', '^', 's', 's']

facecolor_vec = ['r', 'none', 'r', 'none', 'r', 'none']
edgecolor_vec = ['r', 'r', 'r', 'r', 'r', 'r']
z0 = z
np.random.shuffle(z0)
for i in range(6):
    s = shape_vec[i]
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[0, 0] + z0[7*i+j, 0]
        y = cluster_mean[0, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((cluster_mean[0, 0], cluster_mean[0, 1]), 0.48, color='black', fill=False, ls='--')
ax.add_patch(circle)

facecolor_vec = ['g', 'none', 'g', 'none', 'g', 'none']
edgecolor_vec = ['g', 'g', 'g', 'g', 'g', 'g']
z0 = z
np.random.shuffle(z0)
for i in range(6):
    s = shape_vec[i]
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[1, 0] + z0[7*i+j, 0]
        y = cluster_mean[1, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((cluster_mean[1, 0], cluster_mean[1, 1]), 0.48, color='black', fill=False, ls='--')
ax.add_patch(circle)

facecolor_vec = ['b', 'none', 'b', 'none', 'b', 'none']
edgecolor_vec = ['b', 'b', 'b', 'b', 'b', 'b']
z0 = z
np.random.shuffle(z0)
for i in range(6):
    s = shape_vec[i]
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[2, 0] + z0[7*i+j, 0]
        y = cluster_mean[2, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((cluster_mean[2, 0], cluster_mean[2, 1]), 0.48, color='black', fill=False, ls='--')
ax.add_patch(circle)
plt.show()


# Panel 4
z1 = 0.13 * np.array([[-2, 4], [-1, 4], [0, 4], [1, 4], [2, 4],
                    [-2.5, 3], [-1.5, 3], [-0.5, 3], [0.5, 3], [1.5, 3], [2.5, 3],
                    [-3, 2], [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [3, 2],
                    [-3.5, 1], [-2.5, 1], [-1.5, 1], [-0.5, 1], [0.5, 1], [1.5, 1], [2.5, 1], [3.5, 1],
                    [-4, 0], [-3, 0], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                    [-3.5, -1], [-2.5, -1], [-1.5, -1], [-0.5, -1], [0.5, -1], [1.5, -1], [2.5, -1], [3.5, -1],
                    [-3, -2], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [3, -2],
                    [-2.5, -3], [-1.5, -3], [-0.5, -3], [0.5, -3], [1.5, -3], [2.5, -3],
                    [-2, -4], [-1, -4], [0, -4], [1, -4], [2, -4],
                    [-0.5, -5], [0.5, -5]])
z2 = 0.13 * np.array([[-0.5, 5], [0.5, 5],
                    [-2, 4], [-1, 4], [0, 4], [1, 4], [2, 4],
                    [-2.5, 3], [-1.5, 3], [-0.5, 3], [0.5, 3], [1.5, 3], [2.5, 3],
                    [-3, 2], [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [3, 2],
                    [-3.5, 1], [-2.5, 1], [-1.5, 1], [-0.5, 1], [0.5, 1], [1.5, 1], [2.5, 1], [3.5, 1],
                    [-4, 0], [-3, 0], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                    [-3.5, -1], [-2.5, -1], [-1.5, -1], [-0.5, -1], [0.5, -1], [1.5, -1], [2.5, -1], [3.5, -1],
                    [-3, -2], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [3, -2],
                    [-2.5, -3], [-1.5, -3], [-0.5, -3], [0.5, -3], [1.5, -3], [2.5, -3],
                    [-2, -4], [-1, -4], [0, -4], [1, -4], [2, -4]])
assert len(z1[:, 0]) == 63
assert len(z2[:, 0]) == 63
z1[:, 1] = 0.83 * z1[:, 1]
z2[:, 1] = 0.83 * z2[:, 1]
cluster_mean = np.array([[0, 0], [0, 1.21]])

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

shape_vec = ['o', 'o', 'o', '^', '^', '^', 's', 's', 's']
edgecolor_vec = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
z0 = z1
np.random.shuffle(z0)
for i in range(9):
    s = shape_vec[i]
    ec = edgecolor_vec[i]
    fc = 'none'
    for j in range(7):
        x = cluster_mean[0, 0] + z0[7*i+j, 0]
        y = cluster_mean[0, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((0, -0.03), 0.6, color='black', fill=False, ls='--')
ax.add_patch(circle)

shape_vec = ['o', 'o', 'o', '^', '^', '^', 's', 's', 's']
edgecolor_vec = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
facecolor_vec = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
z0 = z2
np.random.shuffle(z0)
for i in range(9):
    s = shape_vec[i]
    ec = edgecolor_vec[i]
    fc = facecolor_vec[i]
    for j in range(7):
        x = cluster_mean[1, 0] + z0[7*i+j, 0]
        y = cluster_mean[1, 1] + z0[7*i+j, 1]
        ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec, s=100)
circle = plt.Circle((0, 1.23), 0.6, color='black', fill=False, ls='--')
ax.add_patch(circle)
ax.set_ylim([-0.65, 1.95])
plt.show()


# # OLD
# # Panel 3
# cluster_mean = np.array([[1.0, 1.0], [1.0, 2.0], [1.8, 1.5], [1.0, 1.0], [1.0, 2.0], [1.8, 1.5],
#                 [1.0, 1.0], [1.0, 2.0], [1.8, 1.5], [1.0, 1.0], [1.0, 2.0], [1.8, 1.5],
#                 [1.0, 1.0], [1.0, 2.0], [1.8, 1.5], [1.0, 1.0], [1.0, 2.0], [1.8, 1.5]])
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.axis("off")
# for i in range(18):
#     r = 0.5 * np.random.uniform(0, 1, 20)
#     theta = 2 * np.pi * np.random.uniform(0, 1, 20)
#     x = cluster_mean[i, 0] + r * np.sin(theta)
#     y = cluster_mean[i, 1] + r * np.cos(theta)
#
#     s = shape_vec[i]
#     ec = edgecolor_vec[i]
#     fc = facecolor_vec[i]
#
#     ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec)
#     circle = plt.Circle((cluster_mean[i, 0], cluster_mean[i, 1]), 0.5, color='black', fill=False, ls='--')
#     ax.add_patch(circle)
# plt.show()
#
# # Panel 4
# cluster_mean = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
#                 [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
#                 [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.axis("off")
# for i in range(18):
#     r = 0.5 * np.random.uniform(0, 1, 20)
#     theta = 2 * np.pi * np.random.uniform(0, 1, 20)
#     x = cluster_mean[i, 0] + r * np.sin(theta)
#     y = cluster_mean[i, 1] + r * np.cos(theta)
#
#     s = shape_vec[i]
#     ec = edgecolor_vec[i]
#     fc = facecolor_vec[i]
#
#     ax.scatter(x, y, marker=s, facecolors=fc, edgecolors=ec)
#     circle = plt.Circle((cluster_mean[i, 0], cluster_mean[i, 1]), 0.5, color='black', fill=False, ls='--')
#     ax.add_patch(circle)
# plt.show()
