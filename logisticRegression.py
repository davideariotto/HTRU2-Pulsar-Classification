

import numpy as np
import scipy.optimize
import utils
import plot

def trainLogisticRegression(DTR, LTR, l, prior):
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), args=(DTR, LTR, l, prior), approx_grad=True)
    return x,f,d

def logreg_obj(v, DTR, LTR, l, prior):
    # v has dimensions = number of features + 1, let's unpack it
    w, b = v[0:-1], v[-1]
    
    j = J_balanced(w, b, DTR, LTR, l, prior)
    return j

def J_balanced(w, b, DTR, LTR, l, prior):
    # regularization term: L2 norm of the weight vector multiplied by the regularization coefficient 
    normTerm = l/2*(np.linalg.norm(w)**2)
    
    sumTermTrueClass = 0
    sumTermFalseClass = 0
    for i in range(DTR.shape[1]):
        if LTR[i]==1:
            # subtract the bias
            sumTermTrueClass += np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b)) 
        else:
            # add the bias
            sumTermFalseClass += np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b)) 
    # Overall cost function = regularization term + ( sumTermTrueClass * prior ) + ( sumTermFalseClass * (1-prior))
    j = normTerm + (prior/DTR[:, LTR==1].shape[1])*sumTermTrueClass + ((1-prior)/DTR[:, LTR==0].shape[1])*sumTermFalseClass
    return j

def getScoresLogisticRegression(wb, D):
    """

    Args:
        wb: concatenated weight and bias
        D: data from validation set

    Returns:
        predictions
    """
    res = np.dot(wb[0:-1], D) + wb[-1]
    return res
 
def findBestLambda(D,L):
    lambdas = np.logspace(-5, 5, num=50)
    
    single_fold = True  # flag that shows if we're going to do single or k folds
    for i in range(0,2):    # two iterations: single and k folds   
        m = 8
        while (m > 5):  # iterate three times (no pca, pca with m=7 and m=6)
            if(m == 8):
                # NO PCA
                execute_task(D,L,lambdas, single_fold, m)
            else:
                # PCA
                D_PCA = utils.PCA(D, L, m)
                execute_task(D_PCA, L, lambdas, single_fold, m)
            m = m - 1
        single_fold = False    
     

def execute_task(D, L, lambdas, single_fold, m): 
    
    minDCF = []
    if(single_fold):
        (DTR, LTR), (DEV, LEV) = utils.single_fold(D, L, None, None, False)
    else:
        allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
        
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            if(single_fold):
                x,f,d = trainLogisticRegression(DTR, LTR, l, 0.5)
                cost = utils.minimum_detection_costs(getScoresLogisticRegression(x, DEV), LEV, model[0], model[1], model[2])
            else:
                scores = []
                for singleKFold in allKFolds:
                    x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], l, 0.5)
                    scores.append(getScoresLogisticRegression(x, singleKFold[2]))
                scores=np.hstack(scores)
                cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                
            minDCF.append(cost)
            print("Lambda:", l, ", cost:", cost)
    if(m==8):
        if( single_fold ):
            title = "no_pca_sf"
        else:
            title = "no_pca_kf"
    else:
        if( single_fold ):
            title = "pca_",str(m),"_sf"
        else:
            title = "pca_",str(m),"_kf"
    plot.plotDCF(lambdas, minDCF, "λ", "file_name_TO_DO", title ,ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    

def computeLogisticRegression(D, L, lambd = 1e-4):
    
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
                    execute_computation(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, lambd, single_fold)   
                else: 
                    print("k-folds")
                    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
                    execute_computation(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, lambd, single_fold)
                
            else:
                # PCA
                print("PCA m = %d" %(m))
                if (single_fold):
                    print("single-fold")
                    D_PCA = utils.PCA(D, L, m)
                    (DTR, LTR), (DEV, LEV) = utils.single_fold(D_PCA, L, None, None, False)
                    execute_computation(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, lambd, single_fold)
                else: 
                    print("k-folds")
                    D_PCA = utils.PCA(D, L, m)
                    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)
                    execute_computation(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, lambd, single_fold)
                
            m = m - 1
        single_fold = False    
           
    return



def execute_computation(DTR, LTR, DEV, LEV, allKFolds, evaluationLabels, lambd, single_fold):
    
    for model in utils.models:
        for pi_T in utils.models:
            if(single_fold):
                # Single Fold
                x, f, d = trainLogisticRegression(DTR, LTR, lambd, pi_T[0])
                mdc = utils.minimum_detection_costs(getScoresLogisticRegression(x, DEV), LEV, model[0], model[1], model[2])  
            else:
                # K-Fold
                scores = []
                for skf in allKFolds:
                    x, f, d = trainLogisticRegression(skf[1], skf[0], lambd, pi_T[0])
                    scores.append(getScoresLogisticRegression(x, skf[2]))
                scores=np.hstack(scores)
                mdc = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", mdc)
    
    return