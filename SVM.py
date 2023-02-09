import numpy as np
from itertools import repeat
import scipy.optimize 
import utils
import plot

def trainLinearSVM(DTR, LTR, C = 1.0, K = 1.0, pi_T = None):
    
    if(pi_T == None):
        w = modifiedDualFormulation(DTR, LTR, C, K)
    else:
        w = modifiedDualFormulationBalanced(DTR, LTR, C, K, pi_T)
    return w

def getScoresLinearSVM(w, DEV, K = 1.0):
    DEV = np.vstack([DEV, np.zeros(DEV.shape[1]) + K])
    S = np.dot(w.T, DEV)
    return S

def trainPolynomialSVM(DTR, LTR, C = 1.0, K = 1.0, c = 0, d = 2):
    x = kernelPoly(DTR, LTR, K, C, d, c)
    return x

def getScoresPolynomialSVM(x, DTR, LTR, DEV, K = 1.0, c = 0, d = 2):
    S = np.sum(np.dot((x * LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DEV) + c)**d + K**2), axis=0)
    return S

def trainRBFKernel(DTR, LTR, gamma = 1.0, K = 1.0, C = 1.0):
    x = kernelRBF(DTR, LTR, K, C, gamma)
    return x

def getScoresRBFKernel(x, DTR, LTR, DEV, gamma = 1.0, K = 1.0):
    kernelFunction = np.zeros((DTR.shape[1], DEV.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DEV.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DEV[:, j], gamma, K**2)
    S=np.sum(np.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunction), axis=0)
    return S
    


def findUnbalancedC (D, L, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30)): 
    '''
    Plot graphs for SVM unbalanced using grid search in order to configure optimal parameters for k and C
    Parameters
    ----------
    D : dataset
    L : label of the dayaset
    rangeK : int, optional
        range of k values to try. The default is [1.0, 10.0].
    rangeC : int, optional
        range for C to try. The default is np.logspace(-5, -1, num=30).
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC)
                    exit()  
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC)
                
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC)
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC)
                
            m = m - 1
        single_fold = False  
    print("\n\nFINISH PLOTS FOR SVM UNBALANCED")
    
    return

def execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC, pi_T = None):
    
    minDCF = []
    for model in utils.models:         
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                if(single_fold):
                    w = trainLinearSVM(DTR, LTR, C, k, pi_T)
                    cost = utils.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                else:
                    scores = []
                    for singleKFold in allKFolds:
                        w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k, pi_T)
                        scores.append(getScoresLinearSVM(w, singleKFold[2], k))
                    scores=np.hstack(scores)
                    cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF.append(cost)
                print("C:", C, ", cost:", cost)
    
    plot.plotDCF_SVM(rangeC, minDCF, "C", "./noPCA-singlefold.png")
    print("\n\nPlot done \n\n")
    
    return


def findBalancedC (D, L, pi_T, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30)):
    '''
    Plot graphs for SVM unbalanced using grid search in order to configure optimal parameters for k and C
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    rangeK : int, optional
        range of k values to try. The default is [1.0, 10.0].
    rangeC : int, optional
        range for C to try. The default is np.logspace(-5, -1, num=30).
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC, pi_T)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC, pi_T)
                
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC, pi_T)
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, rangeK, rangeC, pi_T)
                
            m = m - 1
        single_fold = False  
    print("\n\nFINISH PLOTS FOR SVM BALANCED")
    
    return

def computeLinearSVM(D, L, C, K=1.0, pi_T = None):
    '''
    Generate the result for the SVM table with balanced and unbalanced class and priors 0.5, 0.9 and 0.1
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    C : value of C found before
    K : value of K found before
    pi_T : if None the linear SVM is unbalanced, otherwise it's balanced with the prior we pass
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute_compute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, K, pi_T)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute_compute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, K, pi_T)
                
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute_compute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, K, pi_T)
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute_compute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, K, pi_T)
                
            m = m - 1
        single_fold = False
               
    return


def execute_compute(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, K, pi_T):
    
    for model in utils.models:         
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        if(single_fold):
            w = trainLinearSVM(DTR, LTR, C, K, pi_T)
            minDCF = utils.minimum_detection_costs(getScoresLinearSVM(w, DEV), LEV, model[0], model[1], model[2])
        else:
            scores = []
            for singleKFold in allKFolds:
                w = trainLinearSVM(singleKFold[1], singleKFold[0], C, K, pi_T)
                scores.append(getScoresLinearSVM(w, singleKFold[2]))
            scores=np.hstack(scores)
            minDCF = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
    
    return


def findPolynomialC (D, L, rangeK=[0.0, 1.0], d=2, rangec=[0, 1, 15], rangeC=np.logspace(-5, -1, num=30)):
    '''
    Plot graphs for polynomial SVM using grid search in order to configure optimal parameters for c and C
    Parameters
    ----------
    D : dataset
    L : label
    rangeK : int, optional
         range of k values to try. The default is [0.0, 1.0].
    d : int, optional
        d. The default is 2.
    rangec : int, optional
        range for c values to try. The default is [0, 1, 15].
    rangeC : TYPE, optional
        range for C to try. The default is np.logspace(-5, -1, num=30).
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, d, rangec, rangeC)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, d, rangec, rangeC)       
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, d, rangec, rangeC)  
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, d, rangec, rangeC)      
            m = m - 1
        single_fold = False
     
    return


def execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, d, rangec, rangeC):
    
    model= utils.models[0] # Take only prior 0.5, cfp 1 and cfn 1
    minDCF = []
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                if(single_fold):
                    x = trainPolynomialSVM(DTR, LTR, C, k, c, d)
                    cost = utils.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, k, c, d), LEV, model[0], model[1], model[2])
                else:
                    scores = []
                    for singleKFold in allKFolds:
                        x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, k, c, d)
                        scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], k, c, d))
                    scores=np.hstack(scores)
                    cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                    
                minDCF.append(cost)
                print("C:", C, ", cost:", cost)
                
    if(m==8):
        if(single_fold):
            plot.plotFindC(rangeC, minDCF, "C", "./SVM_poly/noPCA-singlefold.png")
        else: plot.plotFindC(rangeC, minDCF, "C", "./SVM_poly/noPCA-kfold.png")
    else:
        if(single_fold):
            file_name = "./SVM_poly/PCA",str(m),"-singlefold.png"
            plot.plotFindC(rangeC, minDCF, "C", file_name)
        else:
            file_name = "./SVM_poly/PCA",str(m),"-kfold.png"
            plot.plotFindC(rangeC, minDCF, "C", file_name)
    print("\n\nPlot done \n\n")
    
    return


def computePolynomialSVM(D, L, C, c, K = 1.0, d = 2):
    '''
    Generate the result for the polynomial SVM table with priors 0.5, 0.9 and 0.1
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    C : value of C we found before
    c : value of c we found before
    K : value of K we found before
    d : value of d we decide before
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute_poly(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, c, K, d)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute_poly(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, c, K, d)     
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute_poly(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, c, K, d) 
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute_poly(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, c, K, d)    
            m = m - 1
        single_fold = False
           
    return

def execute_poly(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, c, K, d):
       
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        if(single_fold):
            x = trainPolynomialSVM(DTR, LTR, C, K, c, d)
            minDCF = utils.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, K, c, d), LEV, model[0], model[1], model[2])
        else:
            scores = []
            for singleKFold in allKFolds:
                x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, K, c, d)
                scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], K, c, d))
            scores=np.hstack(scores)
            minDCF = utils.minimum_detection_costs(scores, evaluationLabelsPCA6, model[0], model[1], model[2])  
        
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
           
    return



def findRBFKernelC (D, L, rangeK = [0.0, 1.0], rangeC = np.logspace(-3, 3, num=30), rangeGamma = [10**(-4),10**(-3)]):
    '''
    Plot graphs for kernel RBF SVM using grid search in order to configure optimal parameters for gamma and C
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    rangeK : int, optional
        range of k values to try. The default is [0.0, 1.0].
    rangeC : int, optional
        range for C to try. The default is np.logspace(-3, 3, num=30).
    rangeGamma : int, optional
        range for gamma to try. The default is [10**(-4),10**(-3)].
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute_find_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, rangeC, rangeGamma)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute_find_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, rangeC, rangeGamma)       
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute_find_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, rangeC, rangeGamma)  
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute_find_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, rangeC, rangeGamma)      
            m = m - 1
        single_fold = False
     
    
    return


def execute_find_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, rangeK, rangeC, rangeGamma):
    
    model = utils.models[0] # Take only prior 0.5, cfp 1 and cfn 1
    minDCF = []
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                if(single_fold):
                    x = trainRBFKernel(DTR, LTR, gamma, k, C)
                    cost = utils.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, k), LEV, model[0], model[1], model[2])
                else:
                    scores = []
                    for singleKFold in allKFolds:
                        x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, k, C)
                        scores.append(getScoresRBFKernel(x, singleKFold[1], singleKFold[0], singleKFold[2], gamma, k))
                    scores=np.hstack(scores)
                    cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                      
                minDCF.append(cost)
                print("C:", C, ", cost:", cost)
                
    if(m==8):
        if(single_fold):
            plot.plotFindC(rangeC, minDCF, "C", "./RBFKernel/noPCA-singlefold.png")
        else: plot.plotFindC(rangeC, minDCF, "C", "./RBFKernel/noPCA-kfold.png")
    else:
        if(single_fold):
            file_name = "./RBFKernel/PCA",str(m),"-singlefold.png"
            plot.plotFindC(rangeC, minDCF, "C", file_name)
        else:
            file_name = "./RBFKernel/PCA",str(m),"-kfold.png"
            plot.plotFindC(rangeC, minDCF, "C", file_name)
    print("\n\nPlot done \n\n")
    
    return


def computeRBFKernel(D, L, C, gamma, K = 1.0):
    '''
    Generate the result for the kernel RBF SVM table with priors 0.5, 0.9 and 0.1
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    C : value of C we found before
    gamma : value of gamma we found before
    K : value of K we found before
    Returns
    -------
    None.
    '''
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                print("no PCA")
                if (single_fold):
                    print("single-fold")
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                    execute_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, gamma, K)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, gamma, K)        
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, gamma, K)    
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, gamma, K)      
            m = m - 1
        single_fold = False
    
    return

def execute_RBF(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, C, gamma, K):
      
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        if(single_fold):
            x = trainRBFKernel(DTR, LTR, gamma, K, C)
            minDCF = utils.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, K) , LEV, model[0], model[1], model[2])
        else:
            scores = []
            for singleKFold in allKFolds:
                x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, K, C)
                scores.append(getScoresRBFKernel(x,  singleKFold[1], singleKFold[0], singleKFold[2], gamma, K))
            scores=np.hstack(scores)
            minDCF = utils.minimum_detection_costs(scores, evaluationLabelsnoPCA, model[0], model[1], model[2])
            
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)

    return

def modifiedDualFormulation(DTR, LTR, C, K):
    # Compute the D matrix for the extended training set with K=1
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])
    
    # Compute the H matrix
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    # minimization of LD(alpha) = -JD(alpha)
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)

def modifiedDualFormulationBalanced(DTR, LTR, C, K, piT):
    # Compute the D matrix for the extended training set
   
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])

    # Compute the H matrix 
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    
    C1 = C*piT/(DTR[:,LTR == 1].shape[1]/DTR.shape[1])
    C0 = C*(1-piT)/(DTR[:,LTR == 0].shape[1]/DTR.shape[1])
    
    boxConstraint = []
    for i in range(DTR.shape[1]):
        if LTR[i]== 1:
            boxConstraint.append ((0,C1))
        elif LTR[i]== 0:
            boxConstraint.append ((0,C0))
    
    (x, f, d) = scopt.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM, np.zeros(DTR.shape[1]), args=(Hij,), bounds=boxConstraint, factr=1.0)
    return np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)

# TO DELETE !!!
def primalLossDualLossDualityGapErrorRate(DTR, C, Hij, LTR, D, K):
    #[ (0, C), (0, C), ..., (0, C)]
    boxConstraint = list(repeat((0, C), DTR.shape[1]))
    
    (x, f, d) = scopt.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM, np.zeros(DTR.shape[1]), args=(Hij,), bounds=boxConstraint, factr=1.0)
    
    # Now we can recover the primal solution
    alfa = x
    # All xi are inside D
    w = np.sum((alfa*LTR).reshape(1, DTR.shape[1])*D, axis=1)

    return w

def dualObjectiveOfModifiedSVM(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)

def kernelPoly(DTR, LTR, K, C, d, c):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d+ K**2
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    x = dualLossErrorRatePoly(DTR, C, Hij, LTR, K, d, c)
    return x

def dualLossErrorRatePoly(DTR, C, Hij, LTR, K, d, c):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x

def kernelRBF(DTR, LTR, K, C, gamma):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    x = dualLossErrorRateRBF(DTR, C, Hij, LTR, K, gamma)
    return x

def RBF(x1, x2, gamma, K):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))+K**2

def dualLossErrorRateRBF(DTR, C, Hij, LTR, K, gamma):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x

