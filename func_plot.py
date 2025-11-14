# Matplotlib import for plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as matplotlib
from matplotlib.pyplot import figure
from matplotlib.ticker import EngFormatter, LogLocator
from matplotlib import rc
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# Font selection --> LaTeX style
rc('font',**{'family':'serif','serif':['Computer Modern'], 'size' : 20})
rc('text', usetex=True)

plt.rcParams['lines.markersize'] = 4
#Plotting various dependencies of the growth constant

def setup_ax(ax, xlabel='', ylabel='', title=''):
    plt.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(which='major', color='#666666', linestyle='--', alpha=0.5)
    ax.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    if len(xlabel)>0:
        ax.set_xlabel(xlabel)
    if len(ylabel)>0:
        ax.set_ylabel(ylabel)
    if len(title)>0:
        ax.set_title(title, fontsize=14)
    ax.minorticks_on()
    return


