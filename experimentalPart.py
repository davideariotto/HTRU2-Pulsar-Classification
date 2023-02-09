import numpy as np
import gaussianClassifier as gc
import logisticRegression as lr
import scoresRecalibration as sr
import SVM
import GMM
import utils

# EXPERIMENTAL RESULTS
def computeExperimentalResults(D, L, Dtest, Ltest):
    
    D_PCA = utils.PCA(D, L, 7)
    D_PCA_TEST = utils.PCA(Dtest, Ltest, 7)
    
    # no PCA
    print("no PCA")
    ER_MVG(D, L, Dtest, Ltest)
    ER_LR(D, L, Dtest, Ltest)
    ER_SVM(D, L, Dtest, Ltest)
    ER_GMM(D, L, Dtest, Ltest)
    
    # PCA m = 7
    print("PCA m = 7")
    ER_MVG(D_PCA, L, D_PCA_TEST, Ltest)
    ER_LR(D_PCA, L, D_PCA_TEST, Ltest)
    ER_SVM(D_PCA, L, D_PCA_TEST, Ltest)
    ER_GMM(D_PCA, L, D_PCA_TEST, Ltest)
    
    return


def ER_MVG(D,L, Dtest, Ltest):
    
    print("MVG Full-Cov")
    mean0, sigma0, mean1, sigma1 = gc.trainGaussianClassifier(D, L)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest), Ltest, model[0], model[1], model[2])
        print("prior:", model[0], "minDCF:", minDCF_test)
    
    print("MVG Diag-Cov")
    mean0, sigma0, mean1, sigma1 = gc.trainNaiveBayes(D, L)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest), Ltest, model[0], model[1], model[2])
        print("prior:", model[0], "minDCF:", minDCF_test)
    
    print("MVG Tied-Cov")
    mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(D, L)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest), Ltest, model[0], model[1], model[2])
        print("prior:", model[0], "minDCF:", minDCF_test)
    
    return

def ER_LR(D, L, Dtest, Ltest):
    
    lambd = 10**(-4)
    prior = 0.5
    
    print("Logistic Regression pi_T = 0.5")
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
        print("Lambda:", lambd, "pi_T", prior, "prior:", model[0], "minDCF:", minDCF_test) 
        
    prior = 0.1
    print("Logistic Regression pi_T = 0.1")
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
        print("Lambda:", lambd, "pi_T", prior, "prior:", model[0], "minDCF:", minDCF_test)
        
    prior = 0.9
    print("Logistic Regression pi_T = 0.9")
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
        print("Lambda:", lambd, "pi_T", prior, "prior:", model[0], "minDCF:", minDCF_test)
    
    return

def ER_SVM(D, L, Dtest, Ltest):
    
    K_linear = 1.0
       
    C_linear = 10**(-2)
    print("Linear SVM unbalanced")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
          
    C_linear = 10**(-3)
    prior = 0.5
    print("Linear SVM with pi_T = 0.5")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear, pi_T = prior)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
    
    C_linear = 6*10**(-3)
    prior = 0.1
    print("Linear SVM with pi_T = 0.1")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear, pi_T = prior)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
     
    C_linear = 7*10**(-4)
    prior = 0.9
    print("Linear SVM with pi_T = 0.9")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear, pi_T = prior)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
          
    C_quadratic = 5*10**(-5)
    c = 15
    d = 2
    K_quadratic = 1.0
    print("Quadratic SVM")
    x = SVM.trainPolynomialSVM(D, L, C = C_quadratic, K = K_quadratic, c = c, d = d)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(SVM.getScoresPolynomialSVM(x, D, L, Dtest, K = K_quadratic, c = c, d = d), Ltest, model[0], model[1], model[2])
        print("C:", C_quadratic, "c:", c, "d:", d, "prior:", model[0], "minDCF:", minDCF_test)
     
    C_RBF = 10**(-1)
    gamma = 10**(-3)
    K_RBF = 1.0
    print("RBF Kernel SVM")
    x = SVM.trainRBFKernel(D, L, gamma = gamma, K = K_RBF, C = C_RBF)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(SVM.getScoresRBFKernel(x, D, L, Dtest, gamma = gamma, K = K_RBF), Ltest, model[0], model[1], model[2])
        print("C:", C_RBF, "gamma:", gamma, "prior:", model[0], "minDCF:", minDCF_test)

    return

def ER_GMM(D, L, Dtest, Ltest):
    
    nComp_full = 4 # 2^4 = 16
    nComp_diag = 5 # 2^5 = 32
    nComp_tied = 6 # 2^6 = 64
    
    
    print("GMM Full-Cov", 2**(nComp_full), "components")
    GMM0, GMM1 = GMM.trainGaussianClassifier(D, L, nComp_full)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(GMM.getScoresGaussianClassifier(Dtest, GMM0, GMM1), Ltest, model[0], model[1], model[2])
        print("components:", 2**(nComp_full), "prior:", model[0], "minDCF:", minDCF_test)
        

    print("GMM Diag-Cov", 2**(nComp_diag), "components")
    GMM0, GMM1 = GMM.trainNaiveBayes(D, L, nComp_diag)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(GMM.getScoresNaiveBayes(Dtest, GMM0, GMM1), Ltest, model[0], model[1], model[2])
        print("components:", 2**(nComp_diag), "prior:", model[0], "minDCF:", minDCF_test)
        
    
    print("GMM Tied-Cov", 2**(nComp_tied), "components")
    GMM0, GMM1 = GMM.trainTiedCov(D, L, nComp_tied)
    
    for model in utils.models:
        minDCF_test = utils.minimum_detection_costs(GMM.getScoresTiedCov(Dtest, GMM0, GMM1), Ltest, model[0], model[1], model[2])
        print("components:", 2**(nComp_tied), "prior:", model[0], "minDCF:", minDCF_test)
    
    return


# EVALUATION PARAMETER
def EvaluateHyperParameterChosen(D, L, Dtest, Ltest):
    findBestLambdaEval(D, L, Dtest, Ltest)
    findBestCEval_unbalanced(D, L, Dtest, Ltest)
    
    findBestCEval_balanced(D, L, Dtest, Ltest, 0.5)
    findBestCEval_balanced(D, L, Dtest, Ltest, 0.1)
    findBestCEval_balanced(D, L, Dtest, Ltest, 0.9)
    
    findBestCompEval(D, L, Dtest, Ltest)
  
    return

def findBestLambdaEval (D, L, Dtest, Ltest):
    lambdas = np.logspace(-5, 5, num=50) 
    minDCF_val = []
    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            scores = []
            for singleKFold in allKFolds:
                x, f, d = lr.trainLogisticRegression(singleKFold[1], singleKFold[0], l, 0.5)
                scores.append(lr.getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_val.append(cost)
            print("Training Lambda:", l, ", cost:", cost)       
        
    minDCF_eval = []
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            x,f,d = lr.trainLogisticRegression(D, L, l, 0.5)
            cost = utils.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
            minDCF_eval.append(cost)
            print("Test Lambda:", l, ", cost:", cost)
                     
    plot.plotEvaluation(lambdas, minDCF_val, minDCF_eval, "λ", "./evaluationhp/logistic_regression.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    return


def findBestCEval_unbalanced (D, L, Dtest, Ltest):
    k = 1.0
    rangeC=np.logspace(-5, -1, num=30)

    print("unbalanced SVM\n")
    minDCF_val = []
    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
    
    for model in utils.models:
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for C in rangeC:
            scores = []
            
            for singleKFold in allKFolds:
                w = SVM.trainLinearSVM(singleKFold[1], singleKFold[0], C, k)
                scores.append(SVM.getScoresLinearSVM(w, singleKFold[2], k))

            scores=np.hstack(scores)
            cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_val.append(cost)
            print("Training C:", C, ", cost:", cost)
            
    minDCF_eval = []    
    for model in utils.models: 
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for C in rangeC:
                            
            w = SVM.trainLinearSVM(D, L, C, k)
    
            cost = utils.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, k), Ltest, model[0], model[1], model[2])
            
            minDCF_eval.append(cost)
            print("Test C:", C, ", cost:", cost)
                
    plot.plotEvaluation(rangeC, minDCF_val, minDCF_eval, "C", "./evaluationhp/SVM_unbalanced.png")
    return

def findBestCEval_balanced (D, L, Dtest, Ltest, pi_T):
    print("balanced SVM", pi_T)
    
    k = 1.0
    rangeC=np.logspace(-5, -1, num=30)
    minDCF_eval = []
    
    for model in utils.models:           
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])  
        for C in rangeC:                
            w = SVM.trainLinearSVM(D, L, C, k, pi_T)
            cost = utils.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, k), Ltest, model[0], model[1], model[2])
            minDCF_eval.append(cost)
            print("Test C:", C, ", cost:", cost)
    
    minDCF_val = []
    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
    
    for model in utils.models:
        
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for C in rangeC:
            scores = []
            
            for singleKFold in allKFolds:
                w = SVM.trainLinearSVM(singleKFold[1], singleKFold[0], C, k, pi_T)
                scores.append(SVM.getScoresLinearSVM(w, singleKFold[2], k))

            scores=np.hstack(scores)
            cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_val.append(cost)
            print("Training C:", C, ", cost:", cost)

    filename = "./evaluationhp/SVM_balanced-" + str(pi_T) + ".png"
    plot.plotEvaluation(rangeC, minDCF_val, minDCF_eval, "C", filename)
    return
       
def findBestCompEval(D, L, Dtest, Ltest):
    
    print("FULL \n")
    full_eval = []
    model = utils.models[0]

    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        GMM0, GMM1 = GMM.trainGaussianClassifier(D, L, component)
        cost = utils.minimum_detection_costs(GMM.getScoresGaussianClassifier(Dtest, GMM0, GMM1) , Ltest, model[0], model[1], model[2])

        full_eval.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    full_val = []
    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):   
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainGaussianClassifier(singleKFold[1], singleKFold[0], component)
            scores.append(GMM.getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
        
        scores=np.hstack(scores)
        cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        full_val.append(cost)
        print("component:", 2**(component), "cost:", cost)
            
    plot.histrogram(full_val, full_eval, "./evaluationhp/GMM_fullcov.png")
    
    print("DIAG \n")
    diag_eval = []
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    for component in range(7):
        GMM0, GMM1 = GMM.trainNaiveBayes(D, L, component)
        cost = utils.minimum_detection_costs(GMM.getScoresNaiveBayes(Dtest, GMM0, GMM1) , Ltest, model[0], model[1], model[2])

        diag_eval.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    diag_val = []
    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    for component in range(7):
        
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainNaiveBayes(singleKFold[1], singleKFold[0], component)
            scores.append(GMM.getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
        
        scores=np.hstack(scores)
        cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        diag_val.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    plot.histrogram(diag_val, diag_eval, "./evaluationhp/GMM_diagcov.png")
    
    print("TIED \n")
    tied_eval = [] 
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        GMM0, GMM1 = GMM.trainTiedCov(D, L, component)
        cost = utils.minimum_detection_costs(GMM.getScoresTiedCov(Dtest, GMM0, GMM1) , Ltest, model[0], model[1], model[2])

        tied_eval.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    tied_val = []
    allKFolds, evaluationLabels = utils.Kfold(D, L, None, None, False)
    
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainTiedCov(singleKFold[1], singleKFold[0], component)
            scores.append(GMM.getScoresTiedCov(singleKFold[2], GMM0, GMM1))
        
        scores=np.hstack(scores)
        cost = utils.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        tied_val.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    plot.histrogram(tied_val, tied_eval, "./evaluationhp/GMM_tiedcov.png")
    
    return

# ROC
def computeROC(D, L, Dtest, Ltest):
    
    lambd = 1e-4
    prior = 0.5
    
    D_PCA = utils.PCA(D, L, 7)
    D_PCA_T = utils.PCA(Dtest, Ltest, 7)
    
    TPR_tiedcov, FPR_tiedcov = computeTiedCov(D_PCA, L, D_PCA_T, Ltest, lambd, prior)
    TPR_lr, FPR_lr = computeLR(D_PCA, L, D_PCA_T, Ltest, lambd, prior)
    TPR_gmm, FPR_gmm = computeGMM(D, L, Dtest, Ltest, lambd, prior)
    
    plot.plotROC(TPR_tiedcov, FPR_tiedcov, TPR_lr, FPR_lr, TPR_gmm, FPR_gmm, "./ROC/rocplot.png")
    
    return


def computeTiedCov(D, L, Dtest, Ltest, lambd, prior = 0.5):
    
    print("Tied-Cov PCA m = 7")
    
    TPR = []
    FPR = []
    mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(D, L)
    
    tiedcov_scores = gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest)
    scores = sr.calibrateScores(tiedcov_scores, Ltest, lambd).flatten()
    sortedScores = np.sort(scores)
    
    for t in sortedScores:
        confusionMatrix = utils.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(scores, Ltest, t)
        FPRtemp, TPRtemp = utils.computeFPRTPR(prior, 1, 1, confusionMatrix)
        TPR.append(TPRtemp)
        FPR.append(FPRtemp)
        
    return (TPR, FPR)

def computeLR(D, L, Dtest, Ltest, lambd, prior = 0.5):
    
    print("LR PCA m = 7")
    
    TPR = []
    FPR = []
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    lr_scores = lr.getScoresLogisticRegression(x, Dtest)
    scores = sr.calibrateScores(lr_scores, Ltest, lambd).flatten()
    sortedScores = np.sort(scores)
    
    for t in sortedScores:
        confusionMatrix = utils.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(scores, Ltest, t)
        FPRtemp, TPRtemp = utils.computeFPRTPR(prior, 1, 1, confusionMatrix)
        TPR.append(TPRtemp)
        FPR.append(FPRtemp)
    
    return (TPR, FPR)

def computeGMM(D, L, Dtest, Ltest, lambd, prior = 0.5):
    components = 4 # 16 components 
    
    print("GMM PCA m = 7 with", 2**components, "components")
        
    TPR = []
    FPR = []
    GMM0, GMM1 = GMM.trainGaussianClassifier(D, L, components)
    
    GMM_scores = GMM.getScoresGaussianClassifier(Dtest, GMM0, GMM1)
    scores = sr.calibrateScores(GMM_scores, Ltest, lambd, prior).flatten()
    sortedScores = np.sort(scores)
    
    for t in sortedScores:
        confusionMatrix = utils.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(scores, Ltest, t)
        FPRtemp, TPRtemp = utils.computeFPRTPR(prior, 1, 1, confusionMatrix)
        TPR.append(TPRtemp)
        FPR.append(FPRtemp)
        
    return (TPR, FPR)
