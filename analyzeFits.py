#!/usr/bin/env python
# coding: utf-8


import itertools
import numpy as np
import numpy.random as npr
# import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  

import ssm
from ssm.util import one_hot, find_permutation

import scipy.io
from scipy.io import loadmat
from scipy.io import savemat
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import log_loss
import pickle
import copy
import os
import sys

# transforms history and choice data format
# From list of list format To list of 12 (# features) x T (rule block length) array format
def getDataToSuperBlocks(historyData, choiceData, intercept):

    numBlocks = len(historyData)
    
    historySuperBlocks = []
    choiceSuperBlocks = []

    # Loop through super blocks
    # (i.e. series of trials without breaks - may span multiple rules)
    for blk in range(numBlocks):
        T = historyData[blk][0].shape[0]
        H = np.zeros([12,T])
        C = np.zeros([12,T])
        # Loop through features and transform to arrays
        for f in range(12):
            C[f,:] = np.squeeze(choiceData[blk][f].astype('int'))
            H[f,:] = np.argmax(historyData[blk][f][:,intercept:],axis=1).astype('int')
        historySuperBlocks.append(H)
        choiceSuperBlocks.append(C)
        
    return historySuperBlocks, choiceSuperBlocks


# Retrieve most likely states from model for each rule block, feature
def getViterbiSuperBlocks(glmhmm, historyData, choiceData):
    
    numBlocks = len(historyData)

    viterbiStates = []

    # Loop through super blocks
    # (i.e. series of trials without breaks - may span multiple rules)
    for blk in range(numBlocks):
        T = historyData[blk][0].shape[0]
        viterbiBlock = np.zeros([12,T])
        # Loop through features and retrieve most likely states
        for f in range(12):
            viterbiBlock[f,:] = glmhmm.most_likely_states(choiceData[blk][f].astype('int'),input=historyData[blk][f])
        viterbiStates.append(viterbiBlock)
        
    return viterbiStates


# Compute proportion of all trials
# under each state (on current trial, previous trial if next is True)
# and with each outcome history
def getHistoryFrequency(viterbi, history, lag, K, next = False):

    p_H_S = np.zeros([4**lag,K])

    numBlocks = len(history)
    
    # Loop through super blocks
    # (i.e. series of trials without breaks - may span multiple rules)
    for blk in range(numBlocks):
        T = history[blk].shape[1]
        # Loop through states
        for s in range(K):
            # Loop through outcome histories
            for h in range(p_H_S.shape[0]):
                sBlk = viterbi[blk] == s
                hBlk = history[blk] == h
                if next:
                    sBlk = sBlk[:, :-1]  # state in previous trial
                    hBlk = hBlk[:, 1:]  # history for current trial
                p_H_S[h,s] += np.count_nonzero(np.multiply(sBlk,hBlk))
    
    # counts to probabilities
    for s in range(K):
        p_H_S[:,s] = p_H_S[:,s] / np.sum(p_H_S[:,s])
        
    return p_H_S


# Compute feature choice likelihood
# under each state given the outcome history
def getChoiceLikelihood(glmhmm = None,
                        intercept = 1,
                        states = None,
                        history = None,
                        choice = None,
                        lag = 1,
                        K=1,
                        empirical = False):

    if empirical: # from empirical measurement

        pc_HS = np.zeros([2, 4 ** lag, K])

        numBlocks = len(states)

        # Loop through super blocks
        # (i.e. series of trials without breaks - may span multiple rules)
        for blk in range(numBlocks):
            # Loop through states
            for s in range(K):
                for h in range(pc_HS.shape[1]):
                    sBlk = states[blk] == s
                    hBlk = history[blk] == h
                    hsBlk = np.multiply(hBlk, sBlk)
                    pc_HS[1, h, s] += np.count_nonzero(hsBlk)
                    pc_HS[0, h, s] += np.count_nonzero(np.multiply(hsBlk, choice[blk]))

        pc_HS = np.divide(pc_HS[0, :, :], pc_HS[1, :, :])
        pc_HS[np.isnan(pc_HS)] = 0

    else: #  from (fit) model parameters
        Wk = glmhmm.observations.Wk
        # pc_HS_mdl = np.zeros([4**lag,K])

        if intercept == 1:
            x = Wk[:, 0, :1] + Wk[:, 0, 1:]
        else:
            x = Wk[:, 0, :]
        pc_HS = 1 / (1 + np.exp(x.T))

    return pc_HS


# Compute choice probability per state
def getChoiceProbability(pc_HS_mdl = None,
                         pH_S = None,
                         states=None,
                         choice=None,
                         K = 1,
                         empirical=False):

    if empirical: # from empirical measurement
        pc_S = np.zeros([2, K])

        numBlocks = len(states)

        # Loop through super blocks
        # (i.e. series of trials without breaks - may span multiple rules)
        for blk in range(numBlocks):
            # Loop through states
            for s in range(K):
                sBlk = states[blk] == s
                pc_S[1, s] += np.count_nonzero(sBlk)
                pc_S[0, s] += np.count_nonzero(np.multiply(sBlk, choice[blk]))

        pc_S = np.divide(pc_S[0, :], pc_S[1, :])
    else:
        if pc_HS_mdl.shape != pH_S.shape:
            sys.exit('choice likelihood and history frequency shapes do not match')
        else:
            # marginalize over outcome history
            pc_S = np.einsum('ik,ik->k', pc_HS_mdl, pH_S)

    return pc_S


# Get posterior choice probability per feature from model
def getPosteriorSuperBlocks(glmhmm,historyData,choiceData):
    
    numBlocks = len(historyData)

    posteriorStates = []
    posteriorStates2 = []

    # Loop through super blocks
    # (i.e. series of trials without breaks - may span multiple rules)
    for blk in range(numBlocks):
        T = historyData[blk][0].shape[0]
        posteriorBlock = np.zeros([12,T,glmhmm.K])
        posteriorBlock2 = np.zeros([12,T,glmhmm.K])
        # loop through features
        for f in range(12):
            posteriorBlock[f,:,:] = glmhmm.expected_states2(choiceData[blk][f].astype('int'),input=historyData[blk][f])[0]
            posteriorBlock2[f, :, :] = glmhmm.expected_states(choiceData[blk][f].astype('int'), input=historyData[blk][f])[0]
        posteriorStates.append(posteriorBlock)
        posteriorStates2.append(posteriorBlock2)

    return posteriorStates, posteriorStates2


# Recompute no. of trials to criterion
# based on the criteria parameters used in original human dataset
def checkMonByHumCrit(reward, trialsToCriteria):
    numBlocks = len(reward)
    byHumCrit = np.zeros((numBlocks,3))
    # Loop through blocks
    for blk in range(numBlocks):
        byHumCrit[blk,2] = trialsToCriteria[blk]
        R = reward[blk]
        trialNum = np.NaN
        # Find earliest trial where human criteria is met
        for t in range(1, R.shape[0]):
            isCriteria, t2c = getCriteria(R[:t], 'human')
            if isCriteria > 0:
                trialNum = t2c
                break
        byHumCrit[blk,0] = trialNum
        byHumCrit[blk,1] = isCriteria

    return byHumCrit


# check if learning criterion is met
# given reward sequence
def getCriteria(reward, species):
    if species == 'monkey':
        criteria = [7,15,20]
    elif species == 'human':
        criteria = [5,8,10]
    elif species == 'humanJN':
        criteria = [7,15,20]
    else:
        print('species must be monkey or human')
    
    # block length
    T = reward.shape[0]
    # trials to criteria
    t2c = -1
    # which criteria (0 - neither, 1 - window, 2 - consecutive)
    isCriteria = 0
    
    # window
    if np.sum(reward[-criteria[2]:]) >= criteria[1]:
        isCriteria = 1
        t2c = T - criteria[2]
    # consecutive
    elif np.sum(reward[-criteria[0]:]) == criteria[0]:
        isCriteria = 2
        t2c = T - criteria[0]
    
    return isCriteria, t2c


# species specific simulation (N samples)
# for number of trials to criterion
# given choice probability = p
def getPersistTrialsToCriteria(p,N,species):
    
    if species == 'monkey':
        criteria = [7,15,20]
    elif species == 'human':
        criteria = [5,8,10]
    elif species == 'humanJN':
        criteria = [7,15,20]
    else:
        print('species must be monkey or human')
    
    maxBlockLength = 100
    persistTrialsToCriteria = np.zeros([N])
    persistWhichCriteria = np.zeros([N])
    # Run simulation n times
    for n in range(N):
        # Generate reward sequence based on random choices
        R = npr.rand(maxBlockLength) <= p
        # Find first trial where a criterion is met
        for t in range(1,maxBlockLength):
            isCriteria, t2c = getCriteria(R[:t],species)
            if isCriteria > 0:
                # t not t2c
                persistTrialsToCriteria[n] = t
                persistWhichCriteria[n] = isCriteria
                break
            
    # check that criteria was met for every block
    if np.sum(persistWhichCriteria == 0) > 0:
        print('criteria not met for some blocks ' + str(np.sum(persistWhichCriteria == 0)) + ' ' + str(persistWhichCriteria.shape))
    return persistTrialsToCriteria


# Compute transition likelihood
# under each state given the outcome history
def getTransitionLikelihood(glmhmm = None,
                            intercept = 1,
                            states = None,
                            history = None,
                            lag = 1,
                            K = 1,
                            empirical = False):

    if empirical: # from empirical measurement
        M = 4 ** lag # number of unique histories
        pz_HS = np.zeros([K, M, K])

        numBlocks = len(states)

        # Loop through super blocks
        # (i.e. series of trials without breaks - may span multiple rules)
        for blk in range(numBlocks):
            T = states[blk].shape[1]
            # loop through trials to compute counts
            for t in range(1, T):
                for f in range(states[blk].shape[0]):
                    Z = int(states[blk][f, t])
                    S = int(states[blk][f, t - 1])
                    H = int(history[blk][f, t])
                    pz_HS[Z, H, S] += 1

        # Compute likelihoods
        pzH_S = np.copy(pz_HS)
        for s in range(K):
            # Joint probability of next state and outcome history, given state
            pzH_S[:, :, s] = pzH_S[:, :, s] / np.sum(pzH_S[:, :, s])
            for h in range(M):
                # Transition likelihood
                pz_HS[:, h, s] = pz_HS[:, h, s] / np.sum(pz_HS[:, h, s])

    else: #  from (fit) model parameters
        # pz_HS = np.zeros([K,4**lag,K])

        log_Ps = glmhmm.transitions.log_Ps.T[:,np.newaxis,:]
        Ws = np.moveaxis(glmhmm.transitions.Ws, 0, -1)

        if intercept == 1:
            pz_HS = log_Ps + Ws[:,:1,:] + Ws[:,1:,:]
        else:
            pz_HS = log_Ps + Ws

        pz_HS = softmax(pz_HS, axis=0)
        pzH_S = None

    return pz_HS, pzH_S


# Compute transition probability per state
def getTransitionProbability(pz_HS_mdl = None,
                            pH_S0 = None,
                            states=None,
                            K = 1,
                            empirical=False):

    if empirical: # from empirical measurement
        pz_S = np.zeros([K, K])

        numBlocks = len(states)

        # Loop through super blocks
        # (i.e. series of trials without breaks - may span multiple rules)
        for blk in range(numBlocks):
            T = states[blk].shape[1]
            # loop through trials to compute counts
            for t in range(1, T):
                for f in range(states[blk].shape[0]):
                    Z = int(states[blk][f, t])
                    S = int(states[blk][f, t - 1])
                    pz_S[Z, S] += 1

        # compute probabilities
        for s in range(K):
            pz_S[:, s] = pz_S[:, s] / np.sum(pz_S[:, s])
    else:
        # marginalize over outcome history
        pz_S = np.einsum('ijk,jk->ik', pz_HS_mdl, pH_S0)

    return pz_S

# Compute likelihood of transition from each state
# given next state the outcome history
def getEmpiricalReverseTransitionLikelihood(K, lag, states, history):
    M = 4 ** lag # number of unique histories
    pS_Hz_emp = np.zeros([K, M, K])
    pS_Hz_emp2 = np.zeros([K, M, K]) # skips self transitions
    pS_z_emp = np.zeros([K, K])
    pS_z_emp2 = np.zeros([K, K]) # skips self transitions

    numBlocks = len(states)

    # Loop through super blocks
    # (i.e. series of trials without breaks - may span multiple rules)
    for blk in range(numBlocks):
        T = states[blk].shape[1]
        # loop through trials to compute counts
        for t in range(1, T):
            # loop through features
            for f in range(states[blk].shape[0]):
                Z = int(states[blk][f, t])
                S = int(states[blk][f, t - 1])
                H = int(history[blk][f, t])
                pS_Hz_emp[Z, H, S] += 1
                pS_z_emp[Z, S] += 1
                if Z != S: # skip selp transition
                    pS_Hz_emp2[Z, H, S] += 1
                    pS_z_emp2[Z, S] += 1

    # joint probability of outcome history and previous state, given next state
    pHS_z_emp = np.copy(pS_Hz_emp)
    pHS_z_emp2 = np.copy(pS_Hz_emp2)
    # Normalize counts to compute probabilities
    for z in range(K):
        pHS_z_emp[z, :, :] = pHS_z_emp[z, :, :] / np.sum(pHS_z_emp[z, :, :])
        pHS_z_emp2[z, :, :] = pHS_z_emp2[z, :, :] / np.sum(pHS_z_emp2[z, :, :])
        pS_z_emp[z, :] = pS_z_emp[z, :] / np.sum(pS_z_emp[z, :])
        pS_z_emp2[z, :] = pS_z_emp2[z, :] / np.sum(pS_z_emp2[z, :])
        for h in range(M):
            pS_Hz_emp[z, h, :] = pS_Hz_emp[z, h, :] / np.sum(pS_Hz_emp[z, h, :])
            pS_Hz_emp2[z, h, :] = pS_Hz_emp2[z, h, :] / np.sum(pS_Hz_emp2[z, h, :])


    return [pS_Hz_emp, pS_Hz_emp2, pS_z_emp, pS_z_emp2, pHS_z_emp, pHS_z_emp2]


# Compute feature choice likelihood
# given previous state and the outcome history
def getChoiceLikelihoodPrevious(pc_HS_mdl = None,
                                pz_HS_mdl = None,
                                states = None,
                                history = None,
                                choice = None,
                                lag = 1,
                                K = 1,
                                empirical = False):

    if empirical: # from empirical measurement
        M = 4 ** lag # number of unique histories

        pc_HS0 = np.zeros([2, M, K])

        numBlocks = len(states)

        # Loop through super blocks
        # (i.e. series of trials without breaks - may span multiple rules)
        for blk in range(numBlocks):
            T = states[blk].shape[1]
            # loop through trials to compute counts
            for t in range(1, T):
                for f in range(12):
                    H = int(history[blk][f, t])
                    S0 = int(states[blk][f, t - 1])
                    C = int(choice[blk][f, t])

                    pc_HS0[1, H, S0] += 1
                    pc_HS0[0, H, S0] += C

        pc_HS0 = np.divide(pc_HS0[0, :, :], pc_HS0[1, :, :])
        pc_HS0_comp = None
    else:  #  based on (fit) model parameters
        # number of states
        K = pc_HS_mdl.shape[1]
        # number of unique histories
        M = pc_HS_mdl.shape[0]

        pc_HS0_comp = np.zeros([M, K, K, 3])
        # choice component
        pc_HS0_comp[:,:,:,0] = np.around(pc_HS_mdl[:,np.newaxis,:],3)
        # transition component
        pc_HS0_comp[:,:,:,1] = np.around(np.moveaxis(pz_HS_mdl, 0, -1),3)
        # product component
        pc_HS0_comp[:,:,:,2] = np.around(np.multiply(pc_HS0_comp[:,:,:,0],
                                                     pc_HS0_comp[:,:,:,1]),3)
        pc_HS0 = np.around(np.sum(pc_HS0_comp[:, :, :, 2], axis=2), 3)
            
    return pc_HS0, pc_HS0_comp


# Compute feature choice probability
# given previous state
def getChoiceProbabilityPrevious(pc_HS0, pH_S0):

    pc_S0 = np.einsum('ik,ik->k', pc_HS0, pH_S0)
        
    return pc_S0


# Get indices of trial sequences
# corresponding to single rule blocks
def getRuleIndex(rule):
    idx = []
    r = []
    
    currIdx = [0]
    for t in range(1,rule.shape[0]):
        # if rule has switched
        # store and restart stored trial sequence
        if rule[t] != rule[t-1]:
            idx.append(currIdx)
            r.append(rule[t-1])
            currIdx = [t]
        # if rule has not changed
        # append to ongoing trial sequence
        elif rule[t] == rule[t-1]:
            currIdx.append(t)
    # last rule
    idx.append(currIdx)
    r.append(rule[-1])

    return idx, r

# Split trial data from continuous trial blocks into
# individual rule blocks
# (i.e. blocks of continuous trials under the same rule)
def splitIntoBlocks(superBlocksData,species):
    
    ct = 0
    
    # data to be split
    blocksData = {'history' : [],
                  'viterbi' : [],
                  'reward' : [],
                  'choice' : [],
                  'posterior' : [],
                  'currRule' : [],
                  'prevRule' : [],
                  'stimulus' : [],
                  'chosenObject' : [],
                  'whichCriteria' : [],
                  'trialsToCriteria' : []}
    
    # number of super blocks
    numBlocks = len(superBlocksData['rule'])

    # Loop through continuous trial blocks
    for blk in range(numBlocks):
        # Split trial indices by rule blocks
        rule = superBlocksData['rule'][blk].astype('int')
        idx, r = getRuleIndex(rule)
        if len(idx) > 1:
            # Create rule-based trial blocks
            for i in range(1,len(idx)):
                reward = superBlocksData['choice'][blk][r[i],idx[i]]
                isCriteria, t2c = getCriteria(reward,species)
                # if criteria is met
                if isCriteria > 0:
                    blocksData['history'].append(superBlocksData['history'][blk][:,idx[i]])
                    blocksData['viterbi'].append(superBlocksData['viterbi'][blk][:,idx[i]])
                    blocksData['reward'].append(reward)
                    blocksData['choice'].append(superBlocksData['choice'][blk][:,idx[i]])
                    blocksData['posterior'].append(superBlocksData['posterior'][blk][:,idx[i],:])
                    blocksData['currRule'].append(r[i])
                    blocksData['prevRule'].append(r[i-1])
                    blocksData['stimulus'].append(superBlocksData['stimulus'][blk][:,:,idx[i]])
                    blocksData['chosenObject'].append(superBlocksData['chosenObject'][blk][idx[i]])
                    blocksData['whichCriteria'].append(isCriteria)
                    blocksData['trialsToCriteria'].append(t2c)
                else:
                    if i != (len(idx)-1):
                        print('criteria not met for superblock ' + str(blk) + ' block ' + str(i) + ', ' + str(len(idx)))
    
    return blocksData


# Categorize trials based on identity of
# feature(s) under exploration (and their relationship to the rule)
def getTrialsCategory(state,prevRule,currRule):
    
    T = state.shape[1]
    trialsCategory = np.zeros([6,T])
    
    # perseveration
    # previous rule in persist state
    prevF = state[prevRule,:] == 0
    for t in range(T):
        if prevF[t] == 1:
            trialsCategory[0,t] = 1
        elif prevF[t] == 0:
            break
    
    # perseveration length
    numP = int(np.sum(trialsCategory[0,:]))
    # if perseveration lasts the entire block
    if numP == T:
        return trialsCategory
    # if perseveration doesn't last the entire block
    else:
        # Loop through trials in current rule block
        for t in range(numP,T):
            # features in persist/preferred at trial t
            SRule = state[currRule,t]
            SNonRule = state[:,t]
            SNonRule = np.delete(SNonRule,currRule,axis=0)
            # random search
            if np.sum(SNonRule <= 1) == 0 and SRule > 1:
                trialsCategory[1,t] = 1
            # non-rule exploration
            elif np.sum(SNonRule <= 1) > 0 and SRule > 1:
                trialsCategory[2,t] = 1
            # rule favored exploration
            elif np.sum(SNonRule <= 1) > 0 and SRule <= 1:
                trialsCategory[3,t] = 1
            # rule preferred
            elif np.sum(SNonRule <= 1) == 0 and SRule == 1:
                trialsCategory[4,t] = 1
            # rule exploitation
            elif np.sum(SNonRule <= 1) == 0 and SRule == 0:
                trialsCategory[5,t] = 1
            else:
                print('trial has no category')
    
    # check that every trial is labeled
    if np.array_equal(np.sum(trialsCategory,axis=0),np.ones([T])) is False:
        print('trialsCategory is not correct')
        
    return trialsCategory

# Get number of non-contiguous blocks
# and the number of trials in each such block
# where tci is 1 - tci is a binary 1-D indicator array
# which may indicate whether a trial falls in a particular category
def getNAndEll(tci):
    
    N = 0
    ell = []
    
    ct = tci[0]
    # Loop through trials
    for t in range(1,tci.shape[0]):
        if tci[t] == 1 and tci[t-1] == 1:
            ct += 1
        # start counting if indicator transitioned 0->1
        elif tci[t] == 1 and tci[t-1] == 0:
            ct = 1
        # reset counter if indicator transitioned 1->0
        elif tci[t] == 0 and tci[t-1] == 1:
            ell.append(int(ct))
            N += 1
            ct = 0

    if ct > 0:
        N += 1
        ell.append(int(ct))
    
    return N, ell
        

# Preseverations statistic - probability of
# choosing the rule for the previous rule block
# at each trial before and after rule switch
def getPerseverationChoice(choice,prevRule,currRule,species):
    
    numBlocks = len(choice)
    if species == 'monkey':
        nBefore = 8
        nAfter = 8
    elif species == 'human':
        nBefore = 5
        nAfter = 5
    elif species == 'humanJN':
        nBefore = 5
        nAfter = 5
    else:
        print('species must be monkey or human')
    
    beforeChoice = np.zeros([nBefore])
    afterChoice = np.zeros([nAfter])
    
    for blk in range(numBlocks): # loop through blocks
        if choice[blk].shape[1] < nBefore or choice[blk].shape[1] < nAfter:
            print('block '+str(blk)+' is too short')
        else:
            # increment counts
            beforeChoice = beforeChoice + choice[blk][int(currRule[blk]),-nBefore:]
            afterChoice = afterChoice + choice[blk][int(prevRule[blk]),:nAfter]

    pChoice = np.concatenate((beforeChoice,afterChoice),axis=0)
    return pChoice/numBlocks # return probabilities (from counts)

# Preseveration statistic - probability of
# previous rule block's being in the persist state
# at each trial before and after rule switch
def getPerseverationPersist(trialsCategory,state,currRule,species):
    
    numBlocks = len(trialsCategory)
    if species == 'monkey':
        nBefore = 8
        nAfter = 8
    elif species == 'human':
        nBefore = 5
        nAfter = 5
    elif species == 'humanJN':
        nBefore = 5
        nAfter = 5
    else:
        print('species must be monkey or human')
    
    beforePersist = np.zeros([nBefore])
    afterPersist = np.zeros([nAfter])
    
    for blk in range(numBlocks): # Loop through blocks
        if state[blk].shape[1] < nBefore or state[blk].shape[1] < nAfter:
            print('block '+str(blk)+' is too short')
        else:
            # increment counts
            x = copy.deepcopy(state[blk][int(currRule[blk]),:]) == 0
            beforePersist = beforePersist + x[-nBefore:]
            afterPersist = afterPersist + trialsCategory[blk][0,:nAfter]
                
    pPersist = np.concatenate((beforePersist,afterPersist),axis=0)
    return pPersist/numBlocks # return probabilities (from counts)


# Choice probability statistics when
# negative feedback is received during perseveration
def getChoicePreviousStatePerseveration(trialsCategory, states, history, choice, prevRule):
    pc_C = np.zeros([2])
    pc_SC = np.zeros([2, 4])
    pz_S0C = np.zeros([4])

    ph_C = np.zeros([4])

    numBlocks = len(states)

    for blk in range(numBlocks):
        T = states[blk].shape[1]
        for t in range(1, T):
            # Update history count if previous rule was in persist state at t-1
            if trialsCategory[blk][0, t - 1] == 1:
                ph_C[int(history[blk][prevRule[blk], t])] += 1

            # Update counts if, in addition, direct negative feedback is received
            # for choosing the previous rule at trial t-1
            if trialsCategory[blk][0, t - 1] == 1 and history[blk][prevRule[blk], t] == 2:
                C = int(choice[blk][prevRule[blk], t])
                Z = int(states[blk][prevRule[blk], t])

                # Choice count at trial t
                pc_C[1] += 1
                pc_C[0] += C

                # Choice count by state at trial t
                pc_SC[1, Z] += 1
                pc_SC[0, Z] += C

                # State count at trial t
                pz_S0C[Z] += 1

    # Comput probabilites (from counts)
    ph_C = ph_C / np.sum(ph_C)
    pc_C = pc_C[0] / pc_C[1]
    pc_SC = np.divide(pc_SC[0, :], pc_SC[1, :])
    pc_SC[np.isnan(pc_SC)] = 0
    pz_S0C = np.divide(pz_S0C, np.sum(pz_S0C))

    choicePreviousStatePerseveration = {'ph_C': ph_C,
                                        'pc_C': pc_C,
                                        'pc_SC': pc_SC,
                                        'pz_S0C': pz_S0C}

    return choicePreviousStatePerseveration


# Transition probability statistics when
# non-rule feature transitions from explore to non-explore state
# during or at the end of the rule-favored exploration category
def getStateNonRuleExplore(trialsCategory,states,history,currRule):
    
    numBlocks = len(trialsCategory)
    
    pz_S0C = np.zeros([2])
    pH_S0C = np.zeros([4])
    pz_HS0C = np.zeros([2,4])
    
    for blk in range(numBlocks): # Loop through blocks
        T = states[blk].shape[1]
        for t in range(1,T): # Loop through trials
            # t-1 is rule favored exploration, and current rule still explored at t
            if trialsCategory[blk][3,t-1] == 1 and states[blk][currRule[blk],t] <= 1:
                for f in range(12):
                    # find non-rule features that were under exploration at t-1
                    if f != currRule[blk] and states[blk][f,t-1] <= 1:
                        H = int(history[blk][f,t])
                        # check for transition where exploration stops at t
                        Z = int(states[blk][f,t] > 1)
                        # Update counts
                        pz_S0C[Z] += 1
                        pH_S0C[H] += 1
                        pz_HS0C[Z,H] += 1

    # Compute probabilities (from counts)
    pz_S0C = np.divide(pz_S0C,np.sum(pz_S0C))
    pH_S0C = np.divide(pH_S0C,np.sum(pH_S0C))
    pzH_S0C = np.divide(pz_HS0C, np.sum(pz_HS0C))
    for h in range(4):
        pz_HS0C[:,h] = np.divide(pz_HS0C[:,h],np.sum(pz_HS0C[:,h]))

    stats = {'pz_S0C' : pz_S0C,
             'pH_S0C' : pH_S0C,
             'pz_HS0C' : pz_HS0C,
             'pzH_S0C' : pzH_S0C}
    
    return stats

# Return feature dimension (color, shape, pattern)
def getDimension(f):
    if f in [0,1,2,3]:
        return 0
    elif f in [4,5,6,7]:
        return 1
    elif f in [8,9,10,11]:
        return 2
    else:
        print(f)
        print('f is not between 0 and 11')
        
###########################

# Probability of choosing rule feature
# during rule exploitation trials
def getRulePersistChoiceProbability(trialsCategory,currRule,choice):
    
    pC = np.zeros([2])
    
    numBlocks = len(choice)
    
    for blk in range(numBlocks): # Loop through blocks
        for t in range(trialsCategory[blk].shape[1]): # Loop through trial
            if trialsCategory[blk][5,t] == 1: # look for rule exploitation trials
                # Update counts
                pC[1] += 1
                pC[0] += choice[blk][currRule[blk],t]
                
    return pC[0]/pC[1] # Compute and return probability


# State info on features of chosen object
# when a feature transitions up from preferred to persist
def procInferenceUp(states, history ,choice):
    infProcs = []

    numBlocks = len(states)
    for blk in range(numBlocks): # Loop through blocks
        T = states[blk].shape[1]
        for t in range(1, T): # Loop through trials in a block

            for f in range(12): # Loop through features
                # Filter for transition from preferred to persist
                # not chosen, and not rewarded
                if history[blk][f, t] == 0 and states[blk][f, t-1] == 1 and states[blk][f, t] == 0:
                    for f2 in range(12): # Loop through features
                        C = int(choice[blk][f2, t-1])
                        # Filter for chosen object
                        if C == 1:
                            infProcs.append([states[blk][f2, t - 1], states[blk][f2, t], int(getDimension(f)==getDimension(f2))])
    return np.array(infProcs)


# State info on features of chosen object
# when a feature transitions down from preferred to random/avoid
def procInferenceDown(states, history ,choice):
    infProcs = []

    numBlocks = len(states)
    for blk in range(numBlocks): # Loop through blocks
        T = states[blk].shape[1]
        for t in range(1, T): # Loop through trials in a block

            for f in range(12): # Loop through features
                # Filter for transition from preferred to random/avoid
                # not chosen, and rewarded
                if history[blk][f, t] == 1 and states[blk][f, t-1] == 1 and states[blk][f, t] > 1:
                    for f2 in range(12): # Loop through features
                        C = int(choice[blk][f2, t-1])
                        # Filter for chosen object
                        if C == 1:
                            infProcs.append([states[blk][f2, t - 1], states[blk][f2, t], int(getDimension(f)==getDimension(f2))])
    return np.array(infProcs)



def checkAvoidAdjacent(states):
    D0 = 0
    D1 = 0
    numBlocks = len(states)
    for blk in range(numBlocks): # Loop through blocks
        T = states[blk].shape[1]
        for t in range(T): # Loop through trials in a block
            for f in range(12): # Loop through features
                # filter for features that are being avoided
                if states[blk][f, t] == 3:
                    isExp = False
                    for f2 in range(12): # Loop through features
                        # Filter feature in the same dimension that is under exploration
                        if getDimension(f) == getDimension(f2) and states[blk][f2, t] < 2:
                            isExp = True
                    # Update counts
                    D0 += int(isExp)
                    D1 += 1
    # if D0 == 0 and D1 == 0:
    #     rv = 1
    # else:
    #     rv = D0/D1
    return D0/D1 #rv # Return probability (based on counts)


# Probability of choosing at least 1 feature under exploration
def checkRandomnessDuringExploration(states, choice):

    I0 = 0
    I1 = 0

    numBlocks = len(states)
    for blk in range(numBlocks): # Loop through blocks
        T = states[blk].shape[1]
        for t in range(T): # Loop through trials in a block
            isExp = False
            chExp = False
            for f2 in range(12): # Loop through features
                # Filter for chosen features that are under exploration
                C = int(choice[blk][f2, t])
                if C == 1 and states[blk][f2, t] < 2:
                    chExp = True
                # Filter for features that are under exploration
                if states[blk][f2, t] < 2:
                    isExp = True
            # Update counts
            I0 += int(isExp and chExp)
            I1 += int(isExp)

    # Compute probability (from counts)
    if I0 == 0 and I1 == 0:
        I2 = np.nan
    else:
        I2 = I0/I1


    return I2

# Get stats on history dependent transition of non-rule
# feature for trials in category catInd
def getCatEndReasStats(trialsCategory, states, currRule, choice, catInd):

    def updateTransProbStat(transProb_inner):
        for f in range(12):  # Loop through features
            if f != currRule[blk]:  # Filter out rule feature
                # Find trials where non-rule feature f is
                # under exploration AND
                # trials are in category catInd
                sf = np.argwhere(states[blk][f, :] <= 1)  # [0]
                ntr = np.intersect1d(sf, ell)
                if ntr.shape[0] > 0:
                    for tr in ntr:  # loop through intersecting trials
                        # if f and current rule are chosen (C+ for f)
                        if choice[blk][f, tr] == 1 and choice[blk][currRule[blk], tr] == 1:
                            ind = 3
                        # if f is chosen but current rule is not   (C- for f)
                        elif choice[blk][f, tr] == 1 and choice[blk][currRule[blk], tr] == 0:
                            ind = 2
                        # if f is not chosen but current rule is (NC+ for f)
                        elif choice[blk][f, tr] == 0 and choice[blk][currRule[blk], tr] == 1:
                            ind = 1
                        # if neither f not current rule is chosen (NC- for f)
                        else:  # choice[blk][f,tr] == 0 and choice[blk][currRule[blk],tr] == 0:
                            ind = 0

                        if (tr + 1) < states[blk].shape[1]:
                            # if f is still under exploration on next trial
                            if states[blk][f, tr + 1] <= 1:
                                transProb_inner[ind, 0] += 1
                            else:
                                transProb_inner[ind, 1] += 1
        return transProb_inner

    numBlocks = len(states)
    transProb = np.zeros([4,2])

    for blk in range(numBlocks): # Loop through blocks
        # Filter for trials in category (catInd)
        tci = trialsCategory[blk][catInd, :] == 1

        ct = tci[0]
        ell = []
        if tci[0] == 1:
            ell = [0]
        for t in range(1, tci.shape[0]): # Loop through trials
            if tci[t] == 1 and tci[t - 1] == 1: # still in same category
                ct += 1
                ell.append(t)
            elif tci[t] == 1 and tci[t - 1] == 0: # started being in the category
                ct = 1
                ell.append(t)
            elif tci[t] == 0 and tci[t - 1] == 1: # stopped being in the category
                ct = 0
                # now have a continuous sequence of trial
                # in category catInd
                ell = np.array(ell)
                transProb = updateTransProbStat(transProb)
                ell = []

        if ct > 0:
            ell = np.array(ell)
            transProb = updateTransProbStat(transProb)

    return transProb


# Stats on frequency of feature exploration
def getExplorationStats(blocksData, trialsCategory):
    numBlocks = len(blocksData['history'])

    # number of non-rule features that enter preferred/persist, per block
    numNonRuleExplore = np.zeros([numBlocks])
    ellNonRuleExplore = []
    ellNonRuleExplore2 = []
    ellAllExplore = []
    # number of simultaneously explored features (including rule)
    simultaneousExplore = []

    for blk in range(numBlocks):
        SNonRule = copy.deepcopy(blocksData['viterbi'][blk])
        STmp = copy.deepcopy(blocksData['viterbi'][blk])

        simultaneousExplore.append(np.sum(SNonRule <= 1,axis=0))
        idxP = np.where(trialsCategory[blk][0,:] == 1)[0]
        if idxP.shape[0] > 0: # Filter out perseveration trials
            SNonRule[blocksData['prevRule'][blk],idxP] = 999
            STmp[blocksData['prevRule'][blk], idxP] = 999
        SNonRule = np.delete(SNonRule,blocksData['currRule'][blk],axis=0)
        SNonRule = SNonRule <= 1
        STmp = STmp <= 1

        # Filter on rule-preferred exploration trials
        idxP2 = np.where(trialsCategory[blk][3,:] == 1)[0]

        # Number of unique non-rule features explored per block
        # excluding perseveration
        numNonRuleExplore[blk] = np.sum(np.sum(SNonRule,axis=1) > 0)

        # No. consecutive rule-favor exploration trials
        # during which individual non-rule features are under exploration
        ellBlk = []
        if idxP2.shape[0] > 0:
            for f in range(SNonRule.shape[0]):
                N, ell = getNAndEll(SNonRule[f,idxP2])
                ellBlk.append(ell)
            ellNonRuleExplore2.append(ellBlk)

        # No. consecutive trials
        # during which individual non-rule features are under exploration
        ellBlk = []
        for f in range(SNonRule.shape[0]):
            N, ell = getNAndEll(SNonRule[f,:])
            ellBlk.append(ell)
        ellNonRuleExplore.append(ellBlk)

        # No. consecutive trials
        # during which individual features (rule and non-rule) are under exploration
        ellBlk = []
        for f in range(STmp.shape[0]):
            N, ell = getNAndEll(STmp[f,:])
            ellBlk.append(ell)
        ellAllExplore.append(ellBlk)

    return numNonRuleExplore, ellNonRuleExplore, ellNonRuleExplore2, ellAllExplore, simultaneousExplore

# Setup data for object prediction
# and update superBlocksData with additional data
# from history files
def setupObjectPrediction(superBlocksData, species, subjectNames, subjectIndex, glmLag, choiceStats, name):
    # Load history data
    if species == 'monkey':
        historyFn = 'rawData/history/monkey/' + subjectNames[subjectIndex] + '_' + str(glmLag) + '_super.pickle'
        with open(historyFn, 'rb') as f:
            [_, _, _, ruleSuperBlocks, stimulusSuperBlocks, chosenObjectSuperBlocks] = pickle.load(f)
    elif species == 'human':
        historyFn = 'rawData/history/human/' + subjectNames[subjectIndex] + '_' + str(glmLag) + '_super.pickle'
        with open(historyFn, 'rb') as f:
            [_, _, ruleSuperBlocks, chosenObjectSuperBlocks, stimulusSuperBlocks] = pickle.load(f)
    elif species == 'humanJN':
        historyFn = 'rawData/history/humanJN/' + subjectNames[subjectIndex] + '_' + str(glmLag) + '_super.pickle'
        with open(historyFn, 'rb') as f:
            [_, _, _, ruleSuperBlocks, stimulusSuperBlocks, chosenObjectSuperBlocks] = pickle.load(f)
    elif species == 'agent':
        historyFn = 'rawData/history/agent/' + subjectNames[subjectIndex] + '_' + str(glmLag) + '_super.pickle'
        with open(historyFn, 'rb') as f:
            [_, _, ruleSuperBlocks, chosenObjectSuperBlocks, stimulusSuperBlocks] = pickle.load(f)
    # Update superblock data dict
    superBlocksData['stimulus'] = stimulusSuperBlocks
    superBlocksData['chosenObject'] = chosenObjectSuperBlocks

    # set up additional superblock data variables
    if len(ruleSuperBlocks) == len(superBlocksData['choice']):
        rule = []
        stimulus = []
        chosenObject = []
        for blk in range(len(ruleSuperBlocks)):
            blockLength = superBlocksData['choice'][blk].shape[1]

            if species != 'agent':
                rule.append(ruleSuperBlocks[blk][glmLag:])
                stimulus.append(stimulusSuperBlocks[blk][:, :, glmLag:])
                chosenObject.append(chosenObjectSuperBlocks[blk][glmLag:])
            else:
                rule.append(ruleSuperBlocks[blk])
                stimulus.append(stimulusSuperBlocks[blk])
                chosenObject.append(chosenObjectSuperBlocks[blk])

            # check that size of choice, history, rule, chosenObject and stimulus match up
            if (blockLength == rule[-1].shape[0] == stimulus[-1].shape[2] == chosenObject[-1].shape[0]) is False:
                print('block lengths do not match')

        superBlocksData['rule'] = rule
        superBlocksData['stimulus'] = stimulus
        superBlocksData['chosenObject'] = chosenObject
    else:
        print('number of super blocks do not match')
        print('choice:' + str(len(superBlocksData['choice'])) + ', rule:' + str(len(ruleSuperBlocks)))

    # Setup data for object prediction
    featureChoiceLikelihood = []
    featureChoiceLikelihood_pos = []
    objectChoice = []

    numBlocks = len(superBlocksData['choice'])

    for blk in range(numBlocks):
        T = superBlocksData['choice'][blk].shape[1]
        fO = np.zeros([4, 3, T])
        fO_pos = np.zeros([4, 3, T])
        cO = np.zeros([4, T])
        for t in range(T):
            for i in range(4):
                idxF = np.where(superBlocksData['stimulus'][blk][:, i, t] == 1)[0]
                for j in range(len(idxF)):
                    f = idxF[j]
                    H = int(superBlocksData['history'][blk][f, t])
                    S = int(superBlocksData['viterbi'][blk][f, t])
                    fO[i, j, t] = choiceStats['pc_HS_mdl'][H, S]
                    fO_pos[i, j, t] = np.sum(choiceStats['pc_HS_mdl'][H, :]*superBlocksData['posterior2'][blk][f,t,:])
            cO[int(superBlocksData['chosenObject'][blk][t]), t] = 1
        featureChoiceLikelihood.append(fO)
        featureChoiceLikelihood_pos.append(fO_pos)
        objectChoice.append(cO)

    superBlocksData['featureChoiceLikelihood'] = featureChoiceLikelihood
    superBlocksData['featureChoiceLikelihood_pos'] = featureChoiceLikelihood_pos
    superBlocksData['objectChoice'] = objectChoice

    # Save for object prediction
    with open('rawData/forObjectPred/'+species+'/' + name + '_objPredFromFeatPred.pickle', 'wb') as f:
            pickle.dump([np.concatenate(featureChoiceLikelihood, axis=2), np.concatenate(featureChoiceLikelihood_pos, axis=2), np.concatenate(objectChoice, axis=1), np.concatenate(superBlocksData['rule'], axis=0)], f)

    return superBlocksData

# Run all category analysis
def getCategoryAnalysis(blocksData,species):
    
    numBlocks = len(blocksData['history'])
    
    blockLength = np.zeros([numBlocks])
    trialsCategory = []
    blocksCategory = np.zeros([6,numBlocks])
    LCategory = np.zeros([6,numBlocks])
    rewRateCat = np.zeros([6, numBlocks]) + np.NaN

    # First get number of trials (and related stats, e.g. reward rate)
    # for each category in each block
    for blk in range(numBlocks):
        # block length
        blockLength[blk] = int(blocksData['viterbi'][blk].shape[1])
        # trials category
        tc = getTrialsCategory(blocksData['viterbi'][blk],blocksData['prevRule'][blk],blocksData['currRule'][blk])
        trialsCategory.append(tc)
        ellBlk = []
        for i in range(6):
            blocksCategory[i,blk] = np.sum(tc[i,:]) > 0
            LCategory[i,blk] = np.sum(tc[i,:])
            if blocksCategory[i,blk] == 1:
                rewRateCat[i,blk] = np.mean(blocksData['reward'][blk][tc[i,:]==1])

            N, ell = getNAndEll(tc[i,:])
            ellBlk.append(ell)

        # sum of ellCategory should equal block length
        if np.sum(np.concatenate(ellBlk)) != blockLength[blk]:
            print('sum of category segments does not equal block length')

    ############### Exploration stats
    
    # proportion of trials for which at least one trial is being explored
    exploreTrialsProportion = 1 - np.sum(LCategory[1,:])/np.sum(blockLength)

    # Get stats on frequency of feature exploration
    numNonRuleExplore, ellNonRuleExplore, ellNonRuleExplore2, ellAllExplore, simultaneousExplore = getExplorationStats(blocksData, trialsCategory)


    ############### Category evolution across trials within a block
    numBlockBins = 20
    probabilityCategory = np.zeros([6,numBlockBins])

    for blk in range(numBlocks):
        for t in range(int(blockLength[blk])):
            f = np.where(trialsCategory[blk][:,t] == 1)[0]
            nt = int(np.floor(t/blockLength[blk]*numBlockBins))
            probabilityCategory[f,nt] += 1

    # normalize per trial
    for i in range(numBlockBins):
        probabilityCategory[:,i] = np.divide(probabilityCategory[:,i],np.sum(probabilityCategory[:,i]))

    ############### Category-specific stats

    meanLCategory = np.zeros([6])
    varLCategory = np.zeros([6])
    propLCategory = np.zeros([6])
    covvarLCategory = np.zeros([6])
    
    for i in range(6):
        meanLCategory[i] = np.around(np.mean(LCategory[i,:]),2)
        varLCategory[i] = np.around(np.var(LCategory[i,:]),2)
        propLCategory[i] = np.around(meanLCategory[i]/np.mean(blockLength),3)
        blVar = np.around(np.var(blockLength),3)
        covvarLCategory[i] = np.around(np.cov(LCategory[i,:],blockLength)[0,1] / blVar,3)
        
    ############### Perseveration stats
    
    perseverationChoice = getPerseverationChoice(blocksData['choice'],blocksData['prevRule'],blocksData['currRule'],species)
    perseverationPersist = getPerseverationPersist(copy.deepcopy(trialsCategory),blocksData['viterbi'],blocksData['currRule'],species)
    choicePreviousStatePerseveration = getChoicePreviousStatePerseveration(trialsCategory,blocksData['viterbi'],blocksData['history'],blocksData['choice'],blocksData['prevRule'])

    ###############

    stateNonRuleExplore = getStateNonRuleExplore(trialsCategory,blocksData['viterbi'],blocksData['history'],blocksData['currRule'])

    ############### Rule exploitation stats
    
    rulePersistChoiceProbability = getRulePersistChoiceProbability(trialsCategory,blocksData['currRule'],blocksData['choice'])
    rulePersistTrialsToCriteria = getPersistTrialsToCriteria(rulePersistChoiceProbability,1000,species)
    
    ############### Other stats

    infProcsUp = procInferenceUp(blocksData['viterbi'],blocksData['history'],blocksData['choice'])
    infProcsDown = procInferenceDown(blocksData['viterbi'],blocksData['history'],blocksData['choice'])

    avoidAdjacent = checkAvoidAdjacent(blocksData['viterbi'])
    randomnessDurExp = checkRandomnessDuringExploration(blocksData['viterbi'], blocksData['choice'])

    monByHumCrit = checkMonByHumCrit(blocksData['reward'], blocksData['trialsToCriteria'])

    ############### Rule-favored exploration stats + other stats
    catEndReasStats = getCatEndReasStats(trialsCategory, blocksData['viterbi'], blocksData['currRule'], blocksData['choice'],  3)
    catEndReasStats2 = getCatEndReasStats(trialsCategory, blocksData['viterbi'], blocksData['currRule'], blocksData['choice'],  2)


    categoryAnalysis = {'blockLength' : blockLength,
                        'trialsCategory' : trialsCategory,
                        'blocksCategory' : blocksCategory,
                        'LCategory' : LCategory,
                        'exploreTrialsProportion' : exploreTrialsProportion,
                        'numNonRuleExplore' : numNonRuleExplore,
                        'ellNonRuleExplore' : ellNonRuleExplore,
                        'ellNonRuleExplore2' : ellNonRuleExplore2,
                        'ellAllExplore' : ellAllExplore,
                        'simultaneousExplore' : simultaneousExplore,
                        'probabilityCategory' : probabilityCategory,
                        'meanLCategory' : meanLCategory,
                        'varLCategory' : varLCategory,
                        'propLCategory' : propLCategory,
                        'covvarLCategory' : covvarLCategory,
                        'perseverationChoice' : perseverationChoice,
                        'perseverationPersist' : perseverationPersist,
                        'choicePreviousStatePerseveration' : choicePreviousStatePerseveration,
                        'stateNonRuleExplore' : stateNonRuleExplore,
                        'rulePersistTrialsToCriteria' : rulePersistTrialsToCriteria,
                        'rulePersistChoiceProbability': rulePersistChoiceProbability,
                        'rewRateCat' : rewRateCat,
                        'infProcsUp' : infProcsUp,
                        'infProcsDown' : infProcsDown,
                        'avoidAdjacent' : avoidAdjacent,
                        'randomnessDurExp' : randomnessDurExp,
                        'monByHumCrit' : monByHumCrit,
                        'catEndReasStats' : catEndReasStats,
                        'catEndReasStats2' : catEndReasStats2
                        }
    return categoryAnalysis


def runAnalysis(index):
    print(os.getcwd())

    ############################ SETUP PARAMETERS ############################
    # Set parameters for the individual
    speciesList = ['monkey', 'monkey', 'monkey', 'monkey', 'human', 'human', 'human', 'human', 'human', 'humanJN', 'humanJN', 'humanJN', 'humanJN', 'humanJN']#,'agent']
    paramList = [(1, 1, 1, 0, 2, 4, 1, 0, 0.9, 1, 0), # sam
                 (1, 1, 1, 1, 0, 4, 1, 0, 0.9, 1, 0), # tabitha
                 (1, 1, 1, 2, 0, 4, 1, 0, 0.9, 1, 0), # chloe
                 (1, 1, 1, 3, 2, 4, 3, 0, 0.9, 1, 0), # blanche
                 (1, 1, 1, 0, 4, 4, 10, 0, 0.9, 1, 0), # b01
                 (1, 1, 1, 1, 1, 4, 4, 0, 0.9, 1, 0),  # b02
                 (1, 1, 1, 2, 1, 4, 5, 0, 0.9, 1, 0),  # b03
                 (1, 1, 1, 3, 4, 4, 3, 0, 0.9, 1, 0),  # b04
                 (1, 1, 1, 4, 0, 4, 7, 0, 0.9, 1, 0),  # b05
                 (1, 1, 1, 0, 2, 4, 3, 0, 0.9, 1, 0),  # b06
                 (1, 1, 1, 1, 0, 4, 5, 0, 0.9, 1, 0),  # b07
                 (1, 1, 1, 2, 4, 4, 2, 0, 0.9, 1, 0),  # b08
                 (1, 1, 1, 3, 0, 4, 4, 0, 0.9, 1, 0),  # b09
                 (1, 1, 1, 4, 4, 4, 4, 0, 0.9, 1, 0)]   # b10
    paramVals = paramList[index]
    species = speciesList[index]

    glmType = paramVals[0]
    glmLag = paramVals[1]
    intercept = paramVals[2]
    subjectIndex = paramVals[3]
    numKFold = paramVals[4]
    numStates = paramVals[5]
    numInit = paramVals[6]
    observationNoise = paramVals[7]
    diagonalP = paramVals[8]
    wZero = paramVals[9]
    superBlockMethod = paramVals[10]

    ############################ LOAD MODEL AND DATA ############################

    # load model and data (history and choice)
    if species == 'monkey':
        subjectNames = ['sam', 'tabitha', 'chloe', 'blanche']
        name = subjectNames[subjectIndex]
        dataDirectory = 'rawData/inputs/monkey/' + name + '/'
        saveDirectory = 'rawData/modelSaves/monkey/'
        sp0 = 'monkey'

    elif species == 'human':
        subjectNames = ['b01', 'b02', 'b03', 'b04', 'b05']
        name = subjectNames[subjectIndex]
        dataDirectory = 'rawData/inputs/human/' + name + '/'
        saveDirectory = 'rawData/modelSaves/human/'
        sp0 = 'human'

    elif species == 'humanJN':
        subjectNames = ['b06', 'b07', 'b08', 'b09', 'b10']
        name = subjectNames[subjectIndex]
        dataDirectory = 'rawData/inputs/humanJN/' + name + '/' #+ name + '_'
        saveDirectory = 'rawData/modelSaves/humanJN/' #+ name + '_'
        sp0 = 'humanJN'

    elif species == 'agent':
        subjectNames = ['WSLS']
        name = subjectNames[subjectIndex]
        dataDirectory = 'rawData/inputs/agent/' + name + '/'
        saveDirectory = 'rawData/modelSaves/agent/'
        sp0 = 'human'

    # Load model
    dataName = 'glm' + str(glmType) + '_lag' + str(glmLag) + '_int' + str(intercept)
    params = [glmType, glmLag, intercept, subjectIndex, numKFold, numStates, numInit, observationNoise, diagonalP,
              wZero, superBlockMethod]
    saveName = '{:s}'.format('_'.join(map(str, params)))
    # print(os.getcwd(), flush=True)
    with open(saveDirectory + saveName + '.pickle', 'rb') as f:
        try:
            [glmhmm, fit_ll, train_ll, test_ll, trCnt, teCnt] = pickle.load(f)
        except:
            [glmhmm, fit_ll, train_ll, test_ll] = pickle.load(f)

    # Load data
    with open(dataDirectory + dataName + '.pickle', 'rb') as f:
        [historyData, choiceData] = pickle.load(f)

    # Setup superblocks of continuous trial sequences
    historySuperBlocks, choiceSuperBlocks = getDataToSuperBlocks(historyData, choiceData, intercept)
    # Get viterbi states for these superblocks
    viterbiSuperBlocks = getViterbiSuperBlocks(glmhmm, historyData, choiceData)

    # Compute choice probability per state
    pH_S = getHistoryFrequency(viterbiSuperBlocks, historySuperBlocks, glmLag, glmhmm.K)
    pc_HS_mdl = getChoiceLikelihood(glmhmm=glmhmm, intercept=intercept)
    pc_S_mdl = getChoiceProbability(pc_HS_mdl = pc_HS_mdl, pH_S = pH_S)

    # Sort states by choice probability, and re-order states in ssm accordingly
    perm = np.argsort(pc_S_mdl)[::-1]
    glmhmm.permute(perm)

    # Get viterbi states again for superblocks after they have been re-ordered
    viterbiSuperBlocks = getViterbiSuperBlocks(glmhmm, historyData, choiceData)
    # state posterior probabilities
    posteriorSuperBlocks, posteriorSuperBlocks2  = getPosteriorSuperBlocks(glmhmm, historyData, choiceData)

    # history frequency
    pH_S = getHistoryFrequency(viterbiSuperBlocks, historySuperBlocks, glmLag, glmhmm.K)


    ############################ CHOICE STATS ############################
    # model choice likelihood
    pc_HS_mdl = getChoiceLikelihood(glmhmm=glmhmm, intercept=intercept)
    # model choice probability
    pc_S_mdl = getChoiceProbability(pc_HS_mdl=pc_HS_mdl, pH_S=pH_S)
    # empirical choice likelihood
    pc_HS_emp = getChoiceLikelihood(states = viterbiSuperBlocks,
                                    history = historySuperBlocks,
                                    choice = choiceSuperBlocks,
                                    lag = glmLag,
                                    K= glmhmm.K,
                                    empirical = True)
    # empirical choice probability
    pc_S_emp = getChoiceProbability(states = viterbiSuperBlocks,
                                             choice = choiceSuperBlocks,
                                             K = glmhmm.K,
                                             empirical=True)


    # choice statistics dict for saving
    choiceStats = {'pH_S': pH_S,
                   'pc_HS_mdl': pc_HS_mdl,
                   'pc_S_mdl': pc_S_mdl,
                   'pc_HS_emp': pc_HS_emp,
                   'pc_S_emp': pc_S_emp}


    # setup superblock data dict
    # more will be added to it, and it will be saved at the end
    superBlocksData = {'history': historySuperBlocks,
                       'viterbi': viterbiSuperBlocks,
                       'choice': choiceSuperBlocks,
                       'posterior': posteriorSuperBlocks2,
                       'posterior2': posteriorSuperBlocks
                       }


    # outcome frequency per state
    pH_S0 = getHistoryFrequency(superBlocksData['viterbi'], superBlocksData['history'], glmLag, glmhmm.K, next=True)


    ############################ TRANSITION STATS ############################
    # model transition likelihood
    pz_HS_mdl,_ = getTransitionLikelihood(glmhmm = glmhmm, intercept = intercept)
    # model transition probability
    pz_S_mdl = getTransitionProbability(pz_HS_mdl = pz_HS_mdl, pH_S0 = pH_S0)
    # empirical transition likelihood
    pz_HS_emp, pzH_S_emp = getTransitionLikelihood(states = superBlocksData['viterbi'],
                                                    history = superBlocksData['history'],
                                                    lag = glmLag,
                                                    K = glmhmm.K,
                                                    empirical = True)
    # empirical "reverse" transition likelihood
    revTrans = getEmpiricalReverseTransitionLikelihood(glmhmm.K, glmLag, superBlocksData['viterbi'],
                                                 superBlocksData['history'])
    # empirical transition probability
    pz_S_emp = getTransitionProbability(states=superBlocksData['viterbi'],
                                        K = glmhmm.K,
                                        empirical=True)

    # transition statistics dict for saving
    transitionStats = {'pH_S0': pH_S0,
                       'pz_HS_mdl': pz_HS_mdl,
                       'pz_S_mdl': pz_S_mdl,
                       'pz_HS_emp': pz_HS_emp,
                       'pzH_S_emp' : pzH_S_emp,
                       'pz_S_emp': pz_S_emp,
                       'pS_Hz_emp' : revTrans}


    ############################ CHOICE STATS CONDITIONED ON PREV STATE ############################
    # model choice likelihood, given previous state
    pc_HS0_mdl, pc_HS0_mdl_comp = getChoiceLikelihoodPrevious(pc_HS_mdl = choiceStats['pc_HS_mdl'],
                                                              pz_HS_mdl = transitionStats['pz_HS_mdl'])
    # empirical choice likelihood, given previous state (empirical components)
    pc_HS0_emp, pc_HS0_emp_comp = getChoiceLikelihoodPrevious(pc_HS_mdl = choiceStats['pc_HS_emp'],
                                                              pz_HS_mdl = transitionStats['pz_HS_emp'])
    # empirical choice likelihood, given previous state (completely empirical)
    pc_HS0_empemp, _ = getChoiceLikelihoodPrevious(states = superBlocksData['viterbi'],
                                                   history = superBlocksData['history'],
                                                   choice = superBlocksData['choice'],
                                                   lag = glmLag,
                                                   K = glmhmm.K,
                                                   empirical = True)
    # model choice probability, given previous state
    pc_S0_mdl = getChoiceProbabilityPrevious(pc_HS0_mdl, transitionStats['pH_S0'])
    # empirical choice probability, given previous state
    pc_S0_emp = getChoiceProbabilityPrevious(pc_HS0_emp, transitionStats['pH_S0'])

    # choice statistics conditioned on previous state, dict for saving
    choicePreviousStateStats = {'pc_HS0_mdl': pc_HS0_mdl,
                                'pc_HS0_mdl_comp': pc_HS0_mdl_comp,
                                'pc_HS0_emp': pc_HS0_emp,
                                'pc_HS0_emp_comp': pc_HS0_emp_comp,
                                'pc_HS0_empemp': pc_HS0_empemp,
                                'pc_S0_mdl': pc_S0_mdl,
                                'pc_S0_emp': pc_S0_emp}

    # Setup for object prediction
    superBlocksData = setupObjectPrediction(superBlocksData,
                                            species,
                                            subjectNames,
                                            subjectIndex,
                                            glmLag,
                                            choiceStats,
                                            name)
    # Perform all category analysis
    blocksData = splitIntoBlocks(superBlocksData, sp0)
    categoryAnalysis = getCategoryAnalysis(blocksData, sp0)

    # Save all results
    analysis = {'choiceStats': choiceStats,
                'transitionStats': transitionStats,
                'choicePreviousStateStats': choicePreviousStateStats,
                'categoryAnalysis': categoryAnalysis}

    data = {'superBlocksData': superBlocksData,
            'blocksData': blocksData}

    with open('data/' + name +'.pickle', 'wb') as f:
        pickle.dump([analysis], f)

    with open('data/' + name + '.pickle', 'wb') as f:
        pickle.dump([data], f)


if __name__ == "__main__":
    for sub in range(14):
        runAnalysis(sub)
        print('Done: ' + str(sub), flush=True)