import numpy as np
import matplotlib.pyplot as plt

def makeplot(x, y1, label1, y2, label2):
    fig, ax1 = plt.subplots() #figsize=(8,4)

    color = 'tab:red'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(label1, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label2,color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2,'.',  color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim([42,45])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return

def makeplottau(x, y1, label1, y2, label2):
    fig, ax1 = plt.subplots() #figsize=(8,4)

    color = 'tab:red'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(label1, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label2,color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2,'.',  color=color)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim([42,45])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return

def PlotIndex(VarName, Ind, solTchem, xlabel='position [m]'):
    fig, ax1 = plt.subplots()#figsize=(8,4)

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(VarName, color=color)
    ax1.plot(solTchem[:,Header.index('t')], solTchem[:,Header.index(VarName)], color=color)

    # ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Reaction Indx',color=color)  # we already handled the x-label with ax1
    ax2.plot(solTchem[:,Header.index('t')], Ind,'.',  color=color)

    ax2.tick_params(axis='y', labelcolor=color)
#     plt.savefig('SlowIndexVar'+str(IndVar)+'Max.pdf')
    return

def getPartTopIndex(St, IndVar, M, Top=4, threshold=1e-2):
    NtimeStep, Nvar, NtotalReactions = np.shape(St)
    IndxList = []
    for i in range(NtimeStep):
      # check only if mode is exhausted 
      if IndVar < M[i]:
        S = St[i,:,:]
        IndxList += [getParticipationindex(S,IndVar,threshold)]

    Ind = []
    for li in IndxList:
      for i in range(Top):
        if len(li) >= (i+1):
          Ind +=[li[i]]
    return deleteDuplicates(Ind)

def getTop3(St, IndVar, threshold=1e-2):
    NtimeStep, Nvar, NtotalReactions = np.shape(St)

    IndxList = []
    for i in range(NtimeStep):
      S = St[i,:,:]
      IndxList += [getParticipationindex(S,IndVar,threshold=1e-2)]

    MaxPInd = []
    secondInd = []
    thirdInd = []

    for In in IndxList:
      if len(In) >= (1):
        MaxPInd += [In[0]]
      else:
        MaxPInd += [np.nan]

      if len(In) >= (2):
        secondInd += [In[1]]
      else:
        secondInd += [np.nan]

      if len(In) >= (3):
        thirdInd += [In[2]]
      else:
        thirdInd += [np.nan]

    return MaxPInd, secondInd, thirdInd

def getTop3M(St, IndVar, M,  threshold=1e-2):
    NtimeStep, Nvar, NtotalReactions = np.shape(St)

    IndxList = []
    for i in range(NtimeStep):
      S = St[i,:,:]
      IndxList += [getParticipationindex(S,IndVar,threshold=1e-2)]

    MaxPInd = []
    secondInd = []
    thirdInd = []

    for i, In in enumerate(IndxList):
      if IndVar < M[i]: ## check if mode is exhausted 
        if len(In) >= (1):
            MaxPInd += [In[0]]
        else:
          MaxPInd += [np.nan]
        
        if len(In) >= (2):
          secondInd += [In[1]]
        else:
          secondInd += [np.nan]
        
        if len(In) >= (3):
          thirdInd += [In[2]]
        else:
          thirdInd += [np.nan]
      else:
          MaxPInd += [np.nan]
          secondInd += [np.nan]
          thirdInd += [np.nan]
 
    return MaxPInd, secondInd, thirdInd

def getTopIndex(St, IndVar, Top=4, threshold=1e-2):
    NtimeStep, Nvar, NtotalReactions = np.shape(St)
    IndxList = []
    for i in range(NtimeStep):
      S = St[i,:,:]
      IndxList += [getParticipationindex(S,IndVar,threshold=1e-2)]

    MaxPInd = []
    for In in IndxList:
      if len(In) >= (1):
        MaxPInd += [In[0]]

    Ind = []
    for li in IndxList:
      for i in range(Top):
        if len(li) >= (i+1):
          Ind +=[li[i]]
    return MaxPInd, deleteDuplicates(Ind)

def getReactionsIndx(Index,threshold=1e-2):
    fast_ord = np.argsort(abs(Index))
    indList = []
    for i in range(1,len(fast_ord)):
        indx = fast_ord[-i]
        if (abs(Index[indx]) > threshold):
            indList += [indx]
            #print(Index[indx], indx)
    return indList

def getParticipationindex(Parndx,VarIndx,threshold=1e-3):
    return getReactionsIndx(Parndx[VarIndx,:],threshold)

def makePlot(VarName,  Ind, solTchem, Header,  xlabel='position [m]'):
    fig, ax1 = plt.subplots()#figsize=(8,4)

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(VarName, color=color)
    ax1.plot(solTchem[:,Header.index('t')], solTchem[:,Header.index(VarName)], color=color)

    # ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Reaction Indx',color=color)  # we already handled the x-label with ax1
    ax2.plot(solTchem[:,Header.index('t')], Ind,'.',  color=color)

    ax2.tick_params(axis='y', labelcolor=color)
    return 

def deleteDuplicates(x):
    return list(dict.fromkeys(x))

def makePlotIndx(SInd, VarName, Ind, solTchem, Header, nameRHS, logNamesReactions, xlabel='position [m]', ylabel= 'Slow Importance Indx', loc_x=1.1,loc_y=1):
    IndVar = nameRHS.index(VarName)
    fig, ax1 = plt.subplots()#figsize=(8,4)
    
    color = 'k'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(VarName, color=color)
    ax1.plot(solTchem[:,Header.index('t')], solTchem[:,Header.index(VarName)],'--.', color=color)

    # ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(ylabel,color=color)  # we already handled the x-label with ax1
    
    for i in Ind:
        ax2.plot(solTchem[:,Header.index('t')], SInd[:,IndVar,i],'-',label=logNamesReactions[i])
    ax2.legend(loc='best', bbox_to_anchor=(loc_x,loc_y) )    

    ax2.tick_params(axis='y', labelcolor=color)
    return

def makePlotPIndx(SInd, VarName, IndVar, Ind, solTchem, Header, logNamesReactions, xlabel='position [m]',loc_x=1.1,loc_y=1, figname = ''):
#     IndVar = nameRHS.index(VarName)
    fig, ax1 = plt.subplots()#figsize=(8,4)
    
    color = 'k'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(VarName, color=color)
    ax1.plot(solTchem[:,Header.index('t')], solTchem[:,Header.index(VarName)],'--.', color=color)

    # ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Participation Indx: Mode '+str(IndVar),color=color)  # we already handled the x-label with ax1
    
    for i in Ind:
        ax2.plot(solTchem[:,Header.index('t')], SInd[:,IndVar,i],'-',label=logNamesReactions[i])
    ax2.legend(loc='best', bbox_to_anchor=(loc_x,loc_y) )    

    ax2.tick_params(axis='y', labelcolor=color)
    return
