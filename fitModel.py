import numpy as np
import numpy.random as npr

import ssm
from ssm.util import one_hot, find_permutation

from sklearn.model_selection import KFold
import scipy.io
from scipy.io import loadmat
import pickle

# Function to fit hmm-glm model
def fitFunction(glmType,glmLag,intercept,subjectIndex,numKFold,numStates,numInit,observationNoise,diagonalP,wZero,superBlockMethod, species, subj):

    subjectNames = ['sam','tabitha','chloe','blanche']
    dataDirectoryNames = ['inputHumansAllNewJN/','input1/']

    saveDirectory = 'rawData/modelSaves/' + species + '/' + subj + '/'
    params = [glmType,glmLag,intercept,subjectIndex,numKFold,numStates,numInit,observationNoise,diagonalP,wZero,superBlockMethod]
    saveName = '{:s}'.format('_'.join(map(str, params)))

    dataDirectory = 'rawData/inputs/' + species + '/' + subj + '/'
    dataName = 'glm'+str(glmType)+'_lag'+str(glmLag)+'_int'+str(intercept)

    # Load training data
    with open(dataDirectory+dataName+'.pickle', 'rb') as f:
        [XData, YData] = pickle.load(f)

    # gather training and testing data
    numBlkTotal = len(XData)

    # 5 fold cross validation
    # create train and test set indices
    kf = KFold(n_splits=5,shuffle=True,random_state=1)
    kfIdx = 0
    for trainIdx, testIdx in kf.split(np.arange(numBlkTotal)):
        if kfIdx == numKFold:
            blkTrain = trainIdx
            blkTest = testIdx
            print('fKold is '+str(kfIdx))
            print('train is: '+str(blkTrain))
            print('test is: '+str(blkTest))
            break
        else:
            kfIdx += 1

    # create train set
    numTrialTrain = 0
    numTrialTest = 0
    XDataTrain = []
    YDataTrain = []
    for blk in blkTrain:
        for f in range(12):
            XDataTrain.append(XData[blk][f])
            YDataTrain.append(YData[blk][f].astype('int'))
            numTrialTrain += np.shape(YDataTrain[-1])[0]

    # create test set
    XDataTest = []
    YDataTest = []
    for blk in blkTest:
        for f in range(12):
            XDataTest.append(XData[blk][f])
            YDataTest.append(YData[blk][f].astype('int'))
            numTrialTest += np.shape(YDataTest[-1])[0]

    print(numTrialTrain,numTrialTest)
    numInput = np.shape(XDataTrain[-1])[1]

    # Fit base model (1 state)
    glmhmmOne = ssm.HMM(1, 1, numInput, observations="input_driven_obs", observation_kwargs=dict(C=2), transitions='stateinputdriven')
    fit_ll_One = glmhmmOne.fit(YDataTrain, inputs=XDataTrain, method='em', num_iters=2,tolerance=10**-4)

    # Fit state-specific model
    if numStates > 1:
        glmhmm = ssm.HMM(numStates, 1, numInput, observations="input_driven_obs", observation_kwargs=dict(C=2), transitions='stateinputdriven')
        # Initialize model parameters
        glmhmm = initializeObservation(glmhmm,glmhmmOne,observationNoise)
        glmhmm = initializeTransition(glmhmm,glmhmmOne,diagonalP,wZero)
        fit_ll = glmhmm.fit(YDataTrain, inputs=XDataTrain, method='adam', num_iters=10000)

        # Get performance
        train_ll = glmhmm.log_likelihood(YDataTrain, inputs=XDataTrain)/numTrialTrain
        test_ll = glmhmm.log_likelihood(YDataTest, inputs=XDataTest)/numTrialTest
    else:
        glmhmm = glmhmmOne
        fit_ll = fit_ll_One
        # Get performance
        train_ll = glmhmmOne.log_likelihood(YDataTrain, inputs=XDataTrain)/numTrialTrain
        test_ll = glmhmmOne.log_likelihood(YDataTest, inputs=XDataTest)/numTrialTest

    # Save model and results
    with open(saveDirectory+saveName+'.pickle', 'wb') as f:
        pickle.dump([glmhmm, fit_ll, train_ll, test_ll,numTrialTrain, numTrialTest], f)

### function to initialize observation weights of glmhmm
def initializeObservation(glmhmm,glmhmmOne,sigma):

    for k in range(glmhmm.K):
        oShape = glmhmmOne.observations.params[0].shape
        glmhmm.observations.params[k] = glmhmmOne.observations.params[0] * (1 + sigma * npr.rand(oShape[0],oShape[1]))

    return glmhmm

### function to initialize transition weights of glmhmmOne
def initializeTransition(glmhmm,glmhmmOne,diagonalP,wZero):

    # initilize log_Ps variable
    Ps = diagonalP * np.eye(glmhmm.K) + .05 * npr.rand(glmhmm.K, glmhmm.K)
    Ps /= Ps.sum(axis=1, keepdims=True)
    glmhmm.transitions.log_Ps = np.log(Ps)

    # initialize Ws variable
    if wZero == 1:
        glmhmm.transitions.Ws = np.zeros(glmhmm.transitions.Ws.shape)
    elif wZero == 2:
        for k1 in range(glmhmm.K):
            for k2 in range(glmhmm.K):
                glmhmm.transitions.Ws[k1,k2,:] = glmhmmOne.observations.params[0]

    return glmhmm


if __name__ == "__main__":

    import itertools

    glmType = [1]
    glmLag = [1, 2, 3]
    intercept = [1]
    subjectIndex = [0]
    numKFold = [0, 1, 2, 3, 4]
    numStates = [1, 2, 3, 4, 5]
    numInit = [1, 2, 3, 4, 5]
    observationNoise = [0]
    diagonalP = [0.9]
    wZero = [1]
    superBlockMethod = [0]

    subjects = ['sam','tabitha','chloe','blanche',
                'b01', 'b02', 'b03', 'b04', 'b05',
                'b06', 'b07', 'b08', 'b09', 'b10']
    species = ['monkey','monkey','monkey','monkey',
                'human', 'human', 'human', 'human', 'human',
                'humanJN', 'humanJN', 'humanJN', 'humanJN', 'humanJN']

    # Hyperparameters
    allParameters = [glmType, glmLag, intercept, subjectIndex, numKFold, numStates, numInit, observationNoise,
                     diagonalP, wZero, superBlockMethod]

    sps = np.unique(species)
    for sp in sps:  # Loop through species
        sinds = np.where(np.array(species) == sp)[0]
        # loop through and fit model for each subject
        for scnt, sind in enumerate(sinds):
            subj = subjects[sind]
            allParameters[3][0] = scnt

            # loop through hyperparameter combination
            parameterCartProd = list(itertools.product(*allParameters))
            for paramComb in parameterCartProd:
                params = paramComb + (sp, subj)
                fitFunction(*params)
