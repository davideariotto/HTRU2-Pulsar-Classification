
import numpy as np
import utils


def trainGaussianClassifier(D, L):
    mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
    mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
       
    sigma0 = np.cov(D[:, L == 0])
    sigma1 = np.cov(D[:, L == 1])
    
    return mean0, sigma0, mean1, sigma1


def trainNaiveBayes(D, L):
    mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
    mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
              
    sigma0 = np.cov(D[:, L == 0])
    sigma1 = np.cov(D[:, L == 1])
    
    sigma0 = sigma0 * np.identity(sigma0.shape[0])
    sigma1 = sigma1 * np.identity(sigma1.shape[0])

    return mean0, sigma0, mean1, sigma1


def trainTiedCov(D, L):
    mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
    mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
       
    sigma0 = np.cov(D[:, L == 0])
    sigma1 = np.cov(D[:, L == 1])
    
    sigma = 1 / (D.shape[1]) * (D[:, L == 0].shape[1] * sigma0 + D[:, L == 1].shape[1] * sigma1)
    
    return mean0, sigma, mean1, sigma


def getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, evaluationSet):
        LS0 = logpdf_GAU_ND(evaluationSet, mean0, sigma0 )
        LS1 = logpdf_GAU_ND(evaluationSet, mean1, sigma1 )
        #log-likelihood ratios
        llr = LS1-LS0
        return llr

def logpdf_GAU_ND(x, mu, sigma):
    return -(x.shape[0]/2)*np.log(2*np.pi)-(0.5)*(np.linalg.slogdet(sigma)[1])- (0.5)*np.multiply((np.dot((x-mu).T, np.linalg.inv(sigma))).T,(x-mu)).sum(axis=0)


def evaluateGaussianClassifier(D, L, mode):
    
    f = ''
    if(mode == 'fc'):
        f = trainGaussianClassifier
    elif (mode == 'nb'):
        f = trainNaiveBayes
    elif (mode == 'tc'):
        f = trainTiedCov
    
    scoresSingleFold, evaluationLabelsSingleFold = utils.single_fold(D, L, f, getScoresGaussianClassifier)   
    
    dcf_single_05 = utils.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1)
    dcf_single_01 = utils.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = utils.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    # Then we implement the k-fold with k=5
    scoresKFold, evaluationLabelsKFold = utils.Kfold(D, L, trainGaussianClassifier, getScoresGaussianClassifier)
   
    dcf_kfold_05 = utils.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = utils.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = utils.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1)
    
    return dcf_single_05, dcf_single_01, dcf_single_01, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09


def computeMVGClassifier (D,L):
       
    # To understand which model is most promising, and to assess the effects of using PCA, we can adopt two methodologies: single-fold and k-fold
    # We decide to implement both of them. First, we implement the single-fold
    
    print("  Single-fold                K-fold")
    print("0.5  0.1  0.9            0.5  0.1  0.9")
    print("no PCA")
    
    ''' GAUSSIAN CLASSIFIER '''
    res1, res2, res3, res4, res5, res6 = evaluateGaussianClassifier(D, L, mode='fc')
    print("Full Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(res1, res2, res3, res4, res5, res6))
    
    
    ''' NAIVE BAYES '''
    res1, res2, res3, res4, res5, res6 = evaluateGaussianClassifier(D, L, mode='nb')
    print("Diagonal Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(res1, res2, res3, res4, res5, res6))
    
    
    ''' TIED COVARIANCE '''
    res1, res2, res3, res4, res5, res6 = evaluateGaussianClassifier(D, L, mode='tc')
    print("Tied Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(res1, res2, res3, res4, res5, res6))
    
    m = 7
    modes = ['fc', 'nb', 'tc']
    messages = ['Full Covariance', 'Diagonal Covariance', 'Tied Covariance']
    while ( m > 4 ):
        print("PCA m = %d" %(m))
        for mode, msg in zip(modes, messages):
        
            pca = utils.PCA(D, L, m)
            
            res1, res2, res3, res4, res5, res6 = evaluateGaussianClassifier(pca, L, mode=mode)
            
            print(msg)
            print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(res1, res2, res3, res4, res5, res6))
        m = m - 1
    
    
    return