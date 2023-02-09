import numpy as np
import plot
import utils
import gaussianClassifier as gc
import logisticRegression as lr
import SVM
import GMM
import scoresRecalibration as sr
import experimentalPart as ep


def ZNormalization(Data):
    """
    Transform the data into a distribution with 0 mean and 1 variance
    """
    mu = Data.mean(axis=1)
    sigma = D.std(axis=1)
    normalizedData = (Data - utils.mcol(mu)) / utils.mcol(sigma)
    return normalizedData, mu, sigma

if __name__ == '__main__':
    
    # Load the training data
    D,L = utils.load("data/Train.txt")
    
    # Features exploration
    plot.plotFeatures(D, L, utils.features_list, utils.classes_list, "original")
    normalizedData, normalizedMean, normalizedStandardDeviation = ZNormalization(D)
    plot.plotFeatures(normalizedData, L, utils.features_list, utils.classes_list, "normalized")
    
    # Correlation analysis
    plot.heatmap(normalizedData, L, True)
    plot.heatmap(normalizedData, L, False)
    
    
    # MVG CLASSIFIER 
    gc.computeMVGClassifier(normalizedData, L)


    # LOGISTIC REGRESSION 
    lr.findBestLambda(normalizedData, L)
    
    lambd_lr = 1e-4 # best value of lambda
    
    lr.computeLogisticRegression(normalizedData, L, lambd = lambd_lr)
    
    # SUPPORT VECTOR MACHINES
    
    # LINEAR SVM 
    SVM.findUnbalancedC(normalizedData, L, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))

    SVM.findBalancedC(normalizedData, L, 0.5, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
    SVM.findBalancedC(normalizedData, L, 0.1, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
    SVM.findBalancedC(normalizedData, L, 0.9, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
    
    # Best values of C
    C_unbalanced = 10**-2      
    C_balanced_05 = 10**-3
    C_balanced_01 = 6*10**-3
    C_balanced_09 = 7*10**-4
    K=1.0 # best value of K between 1.0 and 10.0
    
    SVM.computeLinearSVM(normalizedData, L, C=C_unbalanced, K=1.0, pi_T=None)
    
    SVM.computeLinearSVM(normalizedData, L, C_balanced_05, K=1.0, pi_T=0.5)
    SVM.computeLinearSVM(normalizedData, L, C_balanced_01, K=1.0, pi_T=0.1)
    SVM.computeLinearSVM(normalizedData, L, C_balanced_09, K=1.0, pi_T=0.9)
    
    # POLYNOMIAL SVM 
    SVM.findPolynomialC(normalizedData, L, rangeK=[0.0,1.0], d=2, rangec=[0,1,15], rangeC=np.logspace(-5, -1, num=30)) 
    
    # Best value of C and c 
    C_polynomial = 5*10**(-5) 
    c_polynomial = 15
    
    SVM.computePolynomialSVM(normalizedData, L, C = C_polynomial, c = c_polynomial, K = 1.0, d = 2)
    
    # RBF SVM
        
    SVM.findRBFKernelC(normalizedData, L, rangeK = [0.0, 1.0], rangeC = np.logspace(-3, 3, num=30), rangeGamma = [10**(-4),10**(-3)])
    
    # Best value of C and gamma 
    C_RBF = 10**(-1)
    gamma_RBG = 10**(-3)
    
    SVM.computeRBFKernel(normalizedData, L, C = C_RBF, gamma = gamma_RBG, K = 1.0)
    
    # GAUSSIAN MIXTURE MODEL
    
    GMM.findGMMComponents(normalizedData, L, maxComp = 7)
    
    # Best values of components for each model
    nComp_full = 4 # 2^4 = 16
    nComp_diag = 5 # 2^5 = 32
    nComp_tied = 6 # 2^6 = 64
    
    GMM.computeGMM(normalizedData, L, nComp_full, mode = "fc")
    GMM.computeGMM(normalizedData, L, nComp_diag, mode = "nb") 
    GMM.computeGMM(normalizedData, L, nComp_tied, mode = "tc")  
    
    # SCORES RECALIBRATION 
    sr.computeActualDCF(normalizedData, L, lambd = lambd_lr, components = nComp_full) 
    sr.computeBayesErrorPlots(normalizedData, L, lambd = lambd_lr, components = nComp_full)
    sr.calibratedBayesErrorPlots(normalizedData, L, lambd = lambd_lr, components = nComp_full)
    sr.computeCalibratedErrorPlot(normalizedData, L, lambd = lambd_lr, components = nComp_full)
    
    # EXPERIMENTAL RESULTS 
        
    DT, LT = load_data.load("data/Test.txt")
    
    # On Z-normalization we use the mean and the standard deviation of the Z-normalization done on the training set
    normalizedDataTest, _, _ = ZNormalization(DT, normalizedMean, normalizedStandardDeviation)
    
    ep.computeExperimentalResults(normalizedData, L, normalizedDataTest, LT)
    
    ep.EvaluateHyperParameterChosen(normalizedData, L, normalizedDataTest, LT)
    
    ep.computeROC(normalizedData, L, normalizedDataTest, LT)
    