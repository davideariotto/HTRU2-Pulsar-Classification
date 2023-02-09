import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn

def plotFeatures(D, L, features_list, labels_list, mode):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if (i == j):
                #continue
                plt.figure()
                plt.hist(D[i, L == 0], density=True, alpha=0.6, edgecolor='black', linewidth=0.5)
                plt.hist(D[i, L == 1], density=True, alpha=0.6, edgecolor='black', linewidth=0.5)
                plt.legend(labels_list)
                plt.xlabel(features_list[i])
                plt.savefig("./features_analysis/"+mode+"/"+features_list[i]+".png")
            else:
                # Else use scatter plot
                #custom_scatter(i, j, features_list[i], features_list[j], D, L, labels_list, mode)
                plt.figure()
                plt.scatter(D[i, L == 0], D[j, L == 0], s=5)
                plt.scatter(D[i, L == 1], D[j, L == 1], s=5)
                plt.legend(labels_list)
                plt.ylabel(features_list[j])
                plt.xlabel(features_list[i])
                plt.savefig("./features_analysis/"+mode+"/"+features_list[i]+"-"+features_list[j]+".png")
    return            

def custom_scatter(i, j, xlabel, ylabel, D, L, classesNames, mode):
    # Function used for scatter plots. It receives the indexes i, j of the attributes
    # to plot, the labels for x, y axes, the dataset matrix D, the array L with the
    # values for the classes and the list of classes names (used for the legend)
    plt.figure()
    plt.scatter(D[i, L == 0], D[j, L == 0], color="#1e90ff", s=10)
    plt.scatter(D[i, L == 1], D[j, L == 1], color="#ff8c00", s=10)
    plt.legend(classesNames)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    #plt.savefig("./features-analysis/"+mode+"/"+xlabel+"-"+ylabel+".png", dpi=1200)
    plt.show()
    return

def heatmap(D, L, note=False):
    
    if(note == False):
        folder = "no_an"
    else:
        folder = "an"
    
    # Training set
    plt.figure()
    plt.title("Heatmap of the whole training set", pad = 12.0)
    seaborn.heatmap(np.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False, annot=note)
    plt.savefig("./heatmaps/"+folder+"/all.png", dpi=1200)
    plt.show()
    
    # False class
    plt.figure()
    plt.title("Heatmap of the false class", pad = 12.0)
    seaborn.heatmap(np.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False, annot=note)
    plt.savefig("./heatmaps/"+folder+"/false.png", dpi=1200)
    plt.show()
    
    # True class
    plt.figure()
    plt.title("Heatmap of the true class", pad = 12.0)
    seaborn.heatmap(np.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False, annot=note)
    plt.savefig("./heatmaps/"+folder+"/true.png", dpi=1200)
    plt.show()
    
    return

def plotDCF(x, y, xlabel, folder_name, text = "", base = 10, ticks = None):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale('log', base = base)
    if(ticks is not None):
        plt.xticks(ticks)
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    
    plt.ylabel("min DCF")
    plt.title(text, pad = 12.0)
    plt.show()
    plt.savefig("./"+title+".png", dpi=1200)
    #plt.savefig(folder_name, dpi=1200)
    return

def plotDCF_SVM(x, y, xlabel, folder_name, text = "", base = 10, ticks = None):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 and K=1.0', color='r')
    plt.plot(x, y[1*len(x): 2*len(x)], label='min DCF prior=0.5 and K=10.0', color='m')
    
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1 and K=1.0', color='b')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.1 and K=10.0', color='c')
    
    plt.plot(x, y[4*len(x): 5*len(x)], label='min DCF prior=0.9 and K=1.0', color='g')
    plt.plot(x, y[5*len(x): 6*len(x)], label='min DCF prior=0.9 and K=10.0', color='y')
    
    plt.xlim([min(x), max(x)])
    plt.xscale('log', base = base)
    
    if(ticks is not None):
        plt.xticks(ticks)
        
    plt.legend(["min DCF prior=0.5 and K=1.0", "min DCF prior=0.5 and K=10.0", 
                "min DCF prior=0.1 and K=1.0", "min DCF prior=0.1 and K=10.0", 
                "min DCF prior=0.9 and K=1.0", "min DCF prior=0.9 and K=10.0"])
    plt.xlabel(xlabel)
    
    plt.ylabel("min DCF")
    plt.title(text, pad = 12.0)
    plt.savefig(folder_name, dpi=1200)
    return

def plotDCF_RBF(x, y, xlabel, folder_name, text = "", base = 10, ticks = None):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='logγ = -4 and K = 0.0', color='r')
    plt.plot(x, y[1*len(x): 2*len(x)], label='logγ = -3 and K = 0.0', color='m')
    
    plt.plot(x, y[2*len(x): 3*len(x)], label='logγ = -4 and K = 1.0', color='b')
    plt.plot(x, y[3*len(x): 4*len(x)], label='logγ = -3 and K = 1.0', color='c')
      
    plt.xlim([min(x), max(x)])
    plt.xscale('log', base = base)
    
    if(ticks is not None):
        plt.xticks(ticks)
        
    plt.legend(["logγ = -4 and K = 0.0","logγ = -3 and K = 0.0",
               "logγ = -4 and K = 1.0","logγ = -3 and K = 1.0"])
    plt.xlabel(xlabel)
    
    plt.ylabel("min DCF")
    plt.title(text, pad = 12.0)
    plt.savefig(folder_name, dpi=1200)
    return

def plotFindC(x, y, xlabel, folder_name, text = ""):
    plt.figure()
    
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - c = 0 and K=0.0', color='m')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - c = 0 and K=1.0', color='y')
    
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - c = 1 and K=0.0', color='b')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 - c = 1 and K=1.0', color='g')
    
    plt.plot(x, y[4*len(x): 5*len(x)], label='min DCF prior=0.5 - c = 15 and K=0.0', color='r')
    plt.plot(x, y[5*len(x): 6*len(x)], label='min DCF prior=0.5 - c = 15 and K=1.0', color='c')

    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    
    plt.legend(["min DCF prior=0.5 - c = 0 and K=0.0", "min DCF prior=0.5 - c = 0 and K=1.0", 
                'min DCF prior=0.5 - c = 1 and K=0.0', 'min DCF prior=0.5 - c = 1 and K=1.0', 
                'min DCF prior=0.5 - c = 15 and K=0.0', 'min DCF prior=0.5 - c = 15 and K=1.0'])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.title(text, pad = 12.0)
    plt.savefig(folder_name, dpi=1200)
    
    return

def bayesErrorPlot(actualDCF, minDCF, effective_prior, model, folder_name, text = "", color = 'r'):
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.plot(effective_prior, actualDCF, label='actual DCF', color=color)
    plt.plot(effective_prior, minDCF, label='min DCF', color=color, linestyle="--")
    plt.xlim([min(effective_prior), max(effective_prior)])
    plt.legend([model + " - act DCF", model + " - min DCF"])
    plt.xlabel(r'$\log \frac{\tilde{\pi}}{1-\tilde{\pi}}$')
    plt.ylabel("DCF")
    plt.title(text, pad = 12.0)
    plt.savefig(folder_name, dpi=1200)
    return

def bayesErrorPlotTotal(actualDCF_MVG, minDCF_MVG, actualDCF_LR, minDCF_LR, actualDCF_GMM, minDCF_GMM, effective_prior, folder_name, text = ""):
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.plot(effective_prior, actualDCF_MVG, label='actual DCF', color='r')
    plt.plot(effective_prior, minDCF_MVG, label='min DCF', color='r', linestyle="--")
    
    plt.plot(effective_prior, actualDCF_LR, label='actual DCF', color='b')
    plt.plot(effective_prior, minDCF_LR, label='min DCF', color='b', linestyle="--")
    
    plt.plot(effective_prior, actualDCF_GMM, label='actual DCF', color='g')
    plt.plot(effective_prior, minDCF_GMM, label='min DCF', color='g', linestyle="--")
    
    plt.xlim([min(effective_prior), max(effective_prior)])
    plt.legend(["Tied Full-Cov - act DCF", " Tied Full-Cov - min DCF",
                "Log Reg - act DCF", "Log Reg - min DCF",
                "Full-Cov, 16-G - act DCF", "Full-Cov, 16-G - min DCF"])
    plt.xlabel(r'$\log \frac{\tilde{\pi}}{1-\tilde{\pi}}$')
    plt.ylabel("DCF")
    plt.title(text, pad = 12.0)
    plt.savefig(folder_name, dpi=1200)
    return

def plotROC(TPR, FPR, TPR1, FPR1, TPR2, FPR2, folder_name):
    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2, color='r')
    plt.plot(FPR1, TPR1, linewidth=2, color='b')
    plt.plot(FPR2, TPR2, linewidth=2, color='g')
    plt.legend(["MVG Tied Full-Cov", "Logistic regression", "Full-Cov, 16-G"])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(folder_name, dpi=1200)
    return

def plotEvaluation(x, y_training, y_test, xlabel, folder_name, text = "", base = 10, ticks = None):
    plt.figure()
    plt.plot(x, y_training[0:len(x)], label='min DCF prior=0.5', color='r', linestyle="--")
    plt.plot(x, y_training[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='b', linestyle="--")
    plt.plot(x, y_training[len(x): 2*len(x)], label='min DCF prior=0.9', color='g', linestyle="--")
    
    plt.plot(x, y_test[0:len(x)], label='min DCF prior=0.5', color='r')
    plt.plot(x, y_test[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='b')
    plt.plot(x, y_test[len(x): 2*len(x)], label='min DCF prior=0.9', color='g')
    
    plt.xlim([min(x), max(x)])
    plt.xscale('log', base = base)
    if(ticks is not None):
        plt.xticks(ticks)
    plt.legend(["min DCF prior=0.5 [Val]", "min DCF prior=0.9 [Val]", "min DCF prior=0.1 [Val]", 
                "min DCF prior=0.5 [Eval]", "min DCF prior=0.9 [Eval]", "min DCF prior=0.1 [Eval]"])
    plt.xlabel(xlabel)
    
    plt.ylabel("min DCF")
    plt.title(text, pad = 12.0)
    plt.savefig(folder_name, dpi=1200)
    return



def histrogram(y_training, y_test, folder_name ):

    data = ((y_training[0], y_test[0]), (y_training[1], y_test[1]), 
            (y_training[2], y_test[2]), (y_training[3], y_test[3]), 
            (y_training[4], y_test[4]), (y_training[5], y_test[5]), 
            (y_training[6], y_test[6]))
    
    dim = len(data[0])
    w = 0.75
    dimw = w / dim
    
    fig, ax = plt.subplots()
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        ax.bar(x + i * dimw, y, dimw, bottom=0.001)
    
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    
    plt.legend(['min DCF [Val]', 'min DCF [Eval]'], loc='upper right')
    plt.xlabel("GMM components")   
    plt.ylabel("min DCF")
    plt.savefig(folder_name, dpi=1200)
    return