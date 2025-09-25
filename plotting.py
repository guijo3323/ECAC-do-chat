import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def boxplot_by_activity(values, labels, title=None, savefig=None):
    acts = np.unique(labels)
    data = [values[labels==a] for a in acts]
    plt.figure()
    plt.boxplot(data, labels=[str(a) for a in acts], showfliers=True)
    plt.xlabel("Atividade (Etiqueta)")
    plt.ylabel("Valor")
    if title: plt.title(title)
    if savefig:
        plt.savefig(savefig, bbox_inches="tight", dpi=150)
    else:
        plt.show()

def scatter_mark_outliers(x, mask, title=None, savefig=None):
    idx = np.arange(x.size)
    plt.figure()
    plt.scatter(idx[~mask], x[~mask], s=6)
    plt.scatter(idx[mask],  x[mask],  s=10)
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    if title: plt.title(title)
    if savefig:
        plt.savefig(savefig, bbox_inches="tight", dpi=150)
    else:
        plt.show()

def plot_3d(X3, mask=None, title=None, savefig=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if mask is None:
        ax.scatter(X3[:,0], X3[:,1], X3[:,2], s=6)
    else:
        ax.scatter(X3[~mask,0], X3[~mask,1], X3[~mask,2], s=6)
        ax.scatter(X3[mask,0],  X3[mask,1],  X3[mask,2],  s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title: ax.set_title(title)
    if savefig:
        fig.savefig(savefig, bbox_inches="tight", dpi=150)
    else:
        plt.show()
