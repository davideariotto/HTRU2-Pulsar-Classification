
import numpy as np
import utils
import gaussianClassifier as gc
import scipy.special as scs
import plot

def trainGaussianClassifier(D, L, components):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    # GMM(1.0, mean, covariance) = GMM(w, mu, sigma)
    GMM0_start = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D0).reshape((D0.shape[0], D0.shape[0]))))]
    GMM1_start = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D1).reshape((D1.shape[0], D1.shape[0]))))]
    
    GMM0 = LBGalgorithm (GMM0_start, D0, components, mode = "full")
    GMM1 = LBGalgorithm (GMM1_start, D1, components, mode = "full")
    
    return GMM0, GMM1

def getScoresGaussianClassifier(X, GMM0, GMM1):
    LS0 = computeLogLikelihood(X, GMM0)
    LS1 = computeLogLikelihood(X, GMM1)
        
    llr = LS1-LS0
    return llr

def trainNaiveBayes(D, L, components):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    # GMM(1.0, mean, covariance) = GMM(w, mu, sigma)
    GMM0_start = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D0)*np.eye( D0.shape[0]).reshape((D0.shape[0]),D0.shape[0])))]
    GMM1_start = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D1)*np.eye( D1.shape[0]).reshape((D1.shape[0]),D1.shape[0])))]
    
    GMM0 = LBGalgorithm (GMM0_start, D0, components, mode = "diag")
    GMM1 = LBGalgorithm (GMM1_start, D1, components, mode = "diag")
    
    return GMM0, GMM1

def getScoresNaiveBayes(X, GMM0, GMM1):
    LS0 = computeLogLikelihood(X, GMM0)
    LS1 = computeLogLikelihood(X, GMM1)
        
    llr = LS1-LS0
    return llr

def trainTiedCov(D, L, components):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    sigma0 =  np.cov(D0).reshape((D0.shape[0], D0.shape[0]))
    sigma1 =  np.cov(D1).reshape((D1.shape[0], D1.shape[0]))
    
    sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*sigma0+D[:, L == 1].shape[1]*sigma1)
    # GMM(1.0, mean, covariance) = GMM(w, mu, sigma)
    GMM0_start = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainEigenvaluesCovariance(sigma))]
    GMM1_start = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainEigenvaluesCovariance(sigma))]
    
    GMM0 = LBGalgorithm (GMM0_start, D0, components, mode = "tied")
    GMM1 = LBGalgorithm (GMM1_start, D1, components, mode = "tied")
    
    return GMM0, GMM1

def getScoresTiedCov(X, GMM0, GMM1):
    LS0 = computeLogLikelihood(X, GMM0)
    LS1 = computeLogLikelihood(X, GMM1)
        
    llr = LS1-LS0
    return llr

def findGMMComponents(D, L, maxComp=7):
    '''
    Plot graphs for GMM full covariance, diagonal covariance and tied full covariance 
    using grid search in order to configure optimal parameters for the number of components
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    maxComp : int, number of components to try  
        DESCRIPTION. The default is 7.
    Returns
    -------
    None.
    '''
    
    modes = ['fc', 'nb', 'tc'] # gaussian classifier (full covariance), naive baies, tied covariance
    allKFolds = [] 
    evaluationLabels = []
    
    for mode in modes:
        
        print("\n\nMODE = ",mode,"\n\n")
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
                        execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, maxComp, mode)   
                    else: 
                        print("k-folds")
                        allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                        execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, maxComp, mode)       
                else:
                    # PCA
                    print("PCA m = %d" %(m))
                    if (single_fold):
                        print("single-fold")
                        D_PCA = utils.PCA(D, L, m)
                        (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                        execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, maxComp, mode)  
                    else: 
                        print("k-folds")
                        D_PCA = utils.PCA(D, L, m)
                        allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                        execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, maxComp, mode)      
                m = m - 1
            single_fold = False
        
    print("\n\nFINISH PLOTS FOR GMM")
    
    return

def execute_find(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, single_fold, m, maxComp, mode):
    
    minDCF = []
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            if(single_fold):
                if(mode=='fc'):
                    GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, component)
                elif(mode=='nb'):
                    GMM0, GMM1 = trainNaiveBayes(DTR, LTR, component)
                else: # mode == 'tc'
                    GMM0, GMM1 = trainTiedCov(DTR, LTR, component)
                cost = utils.minimum_detection_costs(getScoresGaussianClassifier(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
            else:
                scores = []
                for singleKFold in allKFolds:
                    if(mode=='fc'):
                        GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], component)
                    elif(mode=='nb'):
                        GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], component)
                    else: # mode == 'tc'
                        GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], component)
                    scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
                scores=np.hstack(scores)
                cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                       
            minDCF.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    if(m==8):
        if(single_fold):
            file_name = "./GMM/noPCA-singlefold-",mode,".png"
            plot.plotDCF([2**(component) for component in range(maxComp)], minDCF, "GMM components", file_name, base = 2)
        else:
            file_name = "./GMM/noPCA-kfold-",mode,".png"
            plot.plotDCF([2**(component) for component in range(maxComp)], minDCF, "GMM components", file_name, base = 2)
    else:
        if(single_fold):
            file_name = "./GMM/PCA",str(m),"-singlefold-",mode,".png"
            plot.plotFindC(rangeC, minDCF, "C", file_name)
        else:
            file_name = "./GMM/PCA",str(m),"-kfold-",mode,".png"
            plot.plotFindC(rangeC, minDCF, "C", file_name)
    print("\n\nPlot done for ",file_name,"\n\n")
    
    return

def computeGMM(D, L, components, mode = "fc"):
    '''
    
    Parameters
    ----------
    D : dataset
    L : label of the dataset
    components : int, number of optimal components found before
    mode : define it to print value for full-cov, diag-cov or tied-full-cov
        DESCRIPTION. The default is "full".
    Returns
    -------
    None.
    '''

    print("MODE =",mode)
    
    allKFolds = [] 
    evaluationLabels = []
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for model in utils.models:
        for i in range(0,2):    # two iterations: single and k folds   
            m = 8
            while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
                if(m == 8):
                    # NO PCA
                    print("no PCA")
                    if (single_fold):
                        print("single-fold")
                        (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
                        execute_GMM(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, components, single_fold, mode, model)
                    else: 
                        print("k-folds")
                        allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                        execute_GMM(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, components, single_fold, mode, model)  
                else:
                    # PCA
                    print("PCA m = %d" %(m))
                    if (single_fold):
                        print("single-fold")
                        D_PCA = utils.PCA(D, L, m)
                        (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                        execute_GMM(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, components, single_fold, mode, model)
                    else: 
                        print("k-folds")
                        D_PCA = utils.PCA(D, L, m)
                        allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                        execute_GMM(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, components, single_fold, mode, model) 
                print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)   
                m = m - 1
            single_fold = False

    return

def execute_GMM(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, components, single_fold, mode, model):
    
    if(single_fold):
        if(mode == "fc"):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, components)        
            score = getScoresGaussianClassifier(DEV, GMM0, GMM1)
        if(mode == "nb"):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, components)
            score = getScoresNaiveBayes(DEV, GMM0, GMM1)
        if(mode == "tc"):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, components)
            score = getScoresTiedCov(DEV, GMM0, GMM1) 
        minDCF = utils.minimum_detection_costs(score, LEV, model[0], model[1], model[2])
    else:
        scores = []
        for singleKFold in allKFolds:
            if(mode == "fc"):
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            if(mode == "nb"):
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            if(mode == "tc"):
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
        scores=np.hstack(scores)
        minDCF = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
    print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF) 
    
    return


def constrainEigenvaluesCovariance(sigma, psi = 0.01):

    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, utils.mcol(s)*U.T)
    return sigma

def LBGalgorithm(GMM, X, iterations, mode = "full"):
    # estimate parameters for the initial GMM(1.0, mu, sigma)
    GMM = EMalgorithm(X, GMM, mode = mode)
    for i in range(iterations):
        # estimate new parameters after the split
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM, mode = mode)
    return GMM

def split(GMM, alpha = 0.1):
    splittedGMM = []
    # we split in 2 parts each component of the GMM
    for i in range(len(GMM)):  
        weight = GMM[i][0]
        mean = GMM[i][1]
        sigma = GMM[i][2]
        # find the leading eigenvector of sigma
        U, s, Vh = np.linalg.svd(sigma)
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((weight/2, mean + d, sigma))
        splittedGMM.append((weight/2, mean - d, sigma))
    return splittedGMM

def EMalgorithm(X, gmm, delta = 10**(-6), mode = "full"):
    flag = True
    while(flag):
        # Compute log marginal density with initial parameters
        #S = logpdf_GMM(X, gmm)
        S = joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)
        
        #logmarg = marginalDensityGMM(S, gmm)
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm))
        
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = log_likelihood_GMM(logmarg, X)
        
        # ------ E-step ----------
        # Compute the posterior probability for each component of the GMM for each sample
        posteriorProbability = np.exp(S - logmarg.reshape(1, logmarg.size))
        
        # ------ M-step ----------
        (w, mu, cov) = Mstep(X, S, posteriorProbability, mode)
        
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
       
        # Compute the new log densities and the new sub-class conditional densities
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )                                                                            
        loglikelihood2 = log_likelihood_GMM(logmarg, X)
        
        if (loglikelihood2 - loglikelihood1 < delta):
            flag = False
        if (loglikelihood2 - loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    return gmm

def Mstep(X, S, posterior, mode = "full"):
    Zg = np.sum(posterior, axis=1)  # 3
    
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        Sum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            Sum += posterior[g, i] * X[:, i]
        Fg[:, g] = Sum
    
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        Sum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            X_i = X[:, i].reshape((X.shape[0], 1))
            X_iT = X[:, i].reshape((1, X.shape[0]))
            Sum += posterior[g, i] * np.dot(X_i, X_iT)
        Sg[g] = Sum
    
    mu_new = Fg / Zg

    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu_new[:, g].reshape((X.shape[0], 1)),
                           mu_new[:, g].reshape((1, X.shape[0])))
    
    cov_new = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
   
    
    if(mode == "full"): 
        for g in range(S.shape[0]):        
            cov_new[g] = constrainEigenvaluesCovariance(cov_new[g])
    elif(mode == "diag"):
        for g in range(S.shape[0]):
            cov_new[g] = constrainEigenvaluesCovariance(cov_new[g] * np.eye(cov_new[g].shape[0]))
    elif(mode == "tied"):
        tsum = np.zeros((cov_new.shape[1], cov_new.shape[2]))
        for g in range(S.shape[0]):
            tsum += Zg[g]*cov_new[g]
        for g in range(S.shape[0]):
            cov_new[g] = constrainEigenvaluesCovariance(1/X.shape[1] * tsum)
            
    w_new = Zg/np.sum(Zg)
    
    return (w_new, mu_new, cov_new)


def logpdf_GMM(X, gmm):
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        # Compute log density
        S[i, :] = gc.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return S

def joint_log_density_GMM (S, gmm):
    
    for i in range(len(gmm)):
        # Add log of the prior of the corresponding component
        S[i, :] += np.log(gmm[i][0])
    return S

def marginal_density_GMM (S):
    return scs.logsumexp(S, axis = 0)


def log_likelihood_GMM(logmarg, X):
    return np.sum(logmarg)/X.shape[1]

def computeLogLikelihood(X, gmm):
    tempSum=np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        tempSum[i,:]= np.log(gmm[i][0]) + gc.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return scs.logsumexp(tempSum, axis=0)