import numpy as np
import utils
import gaussianClassifier as gc
import logisticRegression as lr
import GMM
import plot

def computeActualDCF(D, L, lambd = 1e-4, components = 4):    
    D_PCA = utils.PCA(D, L, 7)
    allKFolds, evaluationLabels = utils.Kfold(D_PCA, L, None, None, False)

    print("\nMVG Tied Full-Cov - PCA m = 7 \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFolds:
            mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(singleKFold[1], singleKFold[0])
            scores.append(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, singleKFold[2]))
            
        scores=np.hstack(scores)
        actualDCF = utils.compute_actual_DCF(scores, evaluationLabels, model[0], model[1], model[2])
        
        print("Application with prior:", model[0], "actualDCF =", actualDCF)
    
    print("\nLinear Logistic Regression, Lambda =", lambd, "pi_T = 0.5 - PCA m = 7 \n")
    for model in utils.models: 
        scores = []
        for singleKFold in allKFolds:
            x, f, d = lr.trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, 0.5)
            scores.append(lr.getScoresLogisticRegression(x, singleKFold[2]))

        scores=np.hstack(scores)
        actualDCF = utils.compute_actual_DCF(scores, evaluationLabels, model[0], model[1], model[2])
        
        print("Lambda = ", lambd, "pi_T = 0.5 and application with prior:", model[0], "actualDCF =", actualDCF)
        
    print("\nFull-Cov GMM with", 2**components, "components - PCA m = 7 \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
            scores.append(GMM.getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))

        scores=np.hstack(scores)
        actualDCF = utils.compute_actual_DCF(scores, evaluationLabels, model[0], model[1], model[2])
        
        print("Application with prior:", model[0], "actualDCF =", actualDCF)
        
    return

def computeBayesErrorPlots(D, L, lambd = 1e-4, components = 4):

    D_PCA = utils.PCA(D, L, 7)
    allKFolds, evaluationLabels = folds.Kfold(PCA7, L, None, None, False)
    
    pointsToUse = 21
    effective_prior = np.linspace(-4, 4, pointsToUse)
    
    eff_priors = 1/(1+np.exp(-1*effective_prior))
    
    print("\nMVG Tied-Cov - PCA m = 7 \n")
    MVGactualDCFs = []
    MVGminDCFs = []
    for point in range(pointsToUse):
        scores = []
        for singleKFold in allKFolds:
            mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(singleKFold[1], singleKFold[0])
            scores.append(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, singleKFold[2]))
            
        scores=np.hstack(scores)
        MVGactualDCFs.append(utils.compute_actual_DCF(scores, evaluationLabels, eff_priors[point], 1, 1))
        MVGminDCFs.append(utils.minimum_detection_costs(scores, evaluationLabels, eff_priors[point], 1, 1))
        print("At iteration", point, "the min DCF is", MVGminDCFs[point], "and the actual DCF is", MVGactualDCFs[point])
      
    print("\n\nPlot done for MVGTiedCov.png \n\n")  
    plot.bayesErrorPlot(MVGactualDCFs, MVGminDCFs, effective_prior, "Tied Full-Cov", "./bayesErrorPlot/MVGTiedCov.png", color='r')
    
    print("\nLinear Logistic Regression, Lambda =", lambd, "pi_T = 0.5 - PCA m = 7 \n")
    LRactualDCFs = []
    LRminDCFs = []
    for point in range(pointsToUse):
        scores = []
        for singleKFold in allKFolds:
            x, f, d = lr.trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, 0.5)
            scores.append(lr.getScoresLogisticRegression(x, singleKFold[2]))
            
        scores=np.hstack(scores)
        LRactualDCFs.append(utils.compute_actual_DCF(scores, evaluationLabelsPCA7, eff_priors[point], 1, 1))
        LRminDCFs.append(utils.minimum_detection_costs(scores, evaluationLabelsPCA7, eff_priors[point], 1, 1))
        print("At iteration", point, "the min DCF is", LRminDCFs[point], "and the actual DCF is", LRactualDCFs[point])
      
    print("\n\nPlot done for logreg.png \n\n")  
    plot.bayesErrorPlot(LRactualDCFs, LRminDCFs, effective_prior, "Log Reg", "./bayesErrorPlot/logreg.png", color='b')

    print("\nFull-Cov GMM with", 2**components, "components - PCA m = 7 \n")
    GMMactualDCFs = []
    GMMminDCFs = []
    for point in range(pointsToUse):
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
            scores.append(GMM.getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))

        scores=np.hstack(scores)
        GMMactualDCFs.append(utils.compute_actual_DCF(scores, evaluationLabels, eff_priors[point], 1, 1))
        GMMminDCFs.append(utils.minimum_detection_costs(scores, evaluationLabels, eff_priors[point], 1, 1))
        print("At iteration", point, "the min DCF is", GMMminDCFs[point], "and the actual DCF is", GMMactualDCFs[point])
     
    print("\n\nPlot done for full-covGMM.png \n\n")  
    plot.bayesErrorPlot(GMMactualDCFs, GMMminDCFs, effective_prior, "Full-Cov, 16-G", "./bayesErrorPlot/full-covGMM.png", color='g')
    
    
    print("\n\nPlot done for total.png \n\n")  
    plot.bayesErrorPlotTotal(MVGactualDCFs, MVGminDCFs, LRactualDCFs, LRminDCFs, GMMactualDCFs, GMMminDCFs, effective_prior, "./bayesErrorPlot/total.png")
        
    print("\n\nFINISH PLOTS FOR BAYES ERROR")
    return
    
def calibratedBayesErrorPlots(D, L, lambd = 1e-4, components = 4):
    D_PCA = utils.PCA(D, L, 7)
    allKFolds, evaluationLabels = folds.Kfold(D_PCA, L, None, None, False)
    
    pointsToUse = 21
    effective_prior = np.linspace(-4, 4, pointsToUse)
    
    eff_priors = 1/(1+np.exp(-1*effective_prior))
    

    print("\nMVG Tied Full-Cov - PCA m = 7 \n")
    MVGactualDCFs = []
    MVGminDCFs = []
    for point in range(pointsToUse):
        scores = []
        for singleKFold in allKFolds:
            mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(singleKFold[1], singleKFold[0])
            scores.append(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, singleKFold[2]))
            
        scores=np.hstack(scores)
        MVGminDCFs.append(utils.minimum_detection_costs(scores, evaluationLabels, eff_priors[point], 1, 1))
        
        # change the scores for the actualDCF
        calibratedScores = calibrateScores(scores, evaluationLabels, 1e-4).flatten()
        
        MVGactualDCFs.append(utils.compute_actual_DCF(calibratedScores, evaluationLabels, eff_priors[point], 1, 1))
        
        print("At iteration", point, "the min DCF is", MVGminDCFs[point], "and the actual DCF with 10^-4 is", MVGactualDCFs[point])
        
    print("\n\nPlot done for MVGTiedCovCalibrated.png \n\n")  
    plot.bayesErrorPlot(MVGactualDCFs, MVGminDCFs, effective_prior, "Tied Full-Cov", "./bayesErrorPlot/MVGTiedCovCalibrated.png", color='r')

    
    print("\nLinear Logistic Regression, Lambda =", lambd, "pi_T = 0.5 - PCA m = 7 \n")
    LRactualDCFs = []
    LRminDCFs = []
    for point in range(pointsToUse):
        scores = []
        for singleKFold in allKFolds:
            x, f, d = lr.trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, 0.5)
            scores.append(lr.getScoresLogisticRegression(x, singleKFold[2]))
            
        scores=np.hstack(scores)
        LRminDCFs.append(utils.minimum_detection_costs(scores, evaluationLabels, eff_priors[point], 1, 1))
        
        # change the scores for the actualDCF
        calibratedScores = calibrateScores(scores, evaluationLabels, 1e-4).flatten()
        
        LRactualDCFs.append(utils.compute_actual_DCF(calibratedScores, evaluationLabels, eff_priors[point], 1, 1))
        
        print("At iteration", point, "the min DCF is", LRminDCFs[point], "and the actual DCF with 10^-4 is", LRactualDCFs[point])
    
    print("\n\nPlot done for logregCalibrated.png \n\n")  
    plot.bayesErrorPlot(LRactualDCFs, LRminDCFs, effective_prior, "Log Reg", "./bayesErrorPlot/logregCalibrated.png", color = 'b')

    print("\nFull-Cov GMM with", 2**components, "components - PCA m = 7 \n")
    GMMactualDCFs = []
    GMMminDCFs = []
    for point in range(pointsToUse):
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
            scores.append(GMM.getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))

        scores=np.hstack(scores)
        GMMminDCFs.append(utils.minimum_detection_costs(scores, evaluationLabels, eff_priors[point], 1, 1))
        
        # change the scores for the actualDCF
        calibratedScores = calibrateScores(scores, evaluationLabels, 1e-4).flatten()
        
        GMMactualDCFs.append(utils.compute_actual_DCF(calibratedScores, evaluationLabels, eff_priors[point], 1, 1))
        
        print("At iteration", point, "the min DCF is", GMMminDCFs[point], "and the actual DCF with 10^-4 is", GMMactualDCFs[point])
    
    print("\n\nPlot done for full-covGMMCalibrated.png \n\n")
    plot.bayesErrorPlot(GMMactualDCFs, GMMminDCFs, effective_prior, "Full-Cov, 16-G", "./bayesErrorPlot/full-covGMMCalibrated.png", color = 'g')

    print("\n\nPlot done for totalCalibrated.png \n\n")  
    plot.bayesErrorPlotTotal(MVGactualDCFs, MVGminDCFs, LRactualDCFs, LRminDCFs, GMMactualDCFs, GMMminDCFs, effective_prior, "./bayesErrorPlot/totalCalibrated.png")
        
    print("\n\nFINISH PLOTS FOR CALIBRATED BAYES ERROR")
    return

def computeCalibratedErrorPlot(D, L, lambd = 1e-4, components = 4):
    D_PCA = utils.PCA(D, L, 7)
    allKFolds, evaluationLabels = folds.Kfold(D_PCA, L, None, None, False)
    
    print("\nMVG Tied-Cov - PCA m = 7 \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFolds:
            mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(singleKFold[1], singleKFold[0])
            scores.append(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, singleKFold[2]))
            
        scores=np.hstack(scores)
        
        calibrated_Scores = calibrateScores(scores, evaluationLabels, 1e-4).flatten()
        actualDCF = utils.compute_actual_DCF(calibrated_Scores, evaluationLabels, model[0], model[1], model[2])
        
        print("Application with prior:", model[0], "actualDCF =", actualDCF)
    
    print("\nLinear Logistic Regression, Lambda =", lambd, "pi_T = 0.5 - PCA m = 7 \n")
    for model in utils.models: 
        scores = []
        for singleKFold in allKFolds:
            x, f, d = lr.trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, 0.5)
            scores.append(lr.getScoresLogisticRegression(x, singleKFold[2]))

        scores=np.hstack(scores)
        
        calibrated_Scores = calibrateScores(scores, evaluationLabelsPCA7, 1e-4).flatten()
        actualDCF = utils.compute_actual_DCF(calibrated_Scores, evaluationLabels, model[0], model[1], model[2])
        
        print("Lambda = ", lambd, "pi_T = 0.5 and application with prior:", model[0], "actualDCF =", actualDCF)
        
    print("\nFull-Cov GMM with", 2**components, "components - PCA m = 7 \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
            scores.append(GMM.getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))

        scores=np.hstack(scores)
        
        calibrated_Scores = calibrateScores(scores, evaluationLabelsPCA7, 1e-4).flatten()
        actualDCF = utils.compute_actual_DCF(calibrated_Scores, evaluationLabels, model[0], model[1], model[2])
        
        print("Application with prior:", model[0], "actualDCF =", actualDCF)
        
    return
    
    
def calibrateScores(scores, labels, lambd, prior=0.5):
    # f(s) = as+b can be interpreted as the llr for the two class hypothesis
    # class posterior probability: as+b+log(pi/(1-pi)) = as +b'
    scores = utils.mrow(scores)
    x, _, _ = lr.trainLogisticRegression(scores, labels, lambd, prior)
    alpha = x[0]
    betafirst = x[1]
    calibratedScores = alpha * scores + betafirst - np.log(prior/(1 - prior))
    return calibratedScores