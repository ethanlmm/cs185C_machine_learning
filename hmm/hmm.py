
import math
import random
import numpy as np
import functools
import json
import os
class hmm():
    def __init__(self,name,state,symbol):
        self.name=name
        self.state=state
        self.symbol=symbol
        sum_A=0
        sum_B=0
        sum_PI=0
        self.PI={}
        self.A={}
        self.B={}

        for i in state:
            init_PI=1.0/len(state)+0.01*random.uniform(-1,1)*(1.0 / len(state))
            sum_PI+=init_PI
            self.PI[i]=init_PI
            self.A[i]={}
            for j in state:
                init_A = 1.0 / len(state)+ 0.01*random.uniform(-1,1)*(1.0 / len(state))
                sum_A += init_A
                self.A[i][j]=init_A
            for j in state:
                self.A[i][j] /= sum_A
            sum_A=0

            self.B[i]={}
            for k in symbol:
                init_B=1.0/len(symbol)+ 0.01*random.uniform(-1,1)*(1.0 / len(state))
                sum_B += init_B
                self.B[i][k]=init_B
            for k in symbol:
                self.B[i][k] /= sum_B
            sum_B=0
        for i in state:
            self.PI[i]/=sum_PI


    def forward(self,observation):
        T=len(observation)
        a = [{} for t in range(T)]
        c = [0 for x in range(T)]

        c[0] = 0
        for i in self.state:
            a[0][i] = self.PI[i] * self.B[i][observation[0]]
            c[0] = c[0] + a[0][i]

        c[0] = 1 / c[0]
        for i in self.state:
            a[0][i] = c[0] * a[0][i]
        for t in range(1, T):
            c[t] = 0
            for i in self.state:
                a[t][i] = 0
                for j in self.state:
                    a[t][i] = a[t][i] + a[t - 1][j] * self.A[j][i]
                a[t][i] = a[t][i] * self.B[i][observation[t]]
                c[t] = c[t] + a[t][i]
            c[t] = 1 / c[t]

            for i in self.state:
                a[t][i] = c[t] * a[t][i]

        return (a,c)

    def backward(self,observation,c):
        T = len(observation)
        b = [{} for t in range(T)]

        for i in self.state:
            b[T - 1][i] = c[T - 1]
        for t in range(T - 2, -1, -1):
            for i in self.state:
                b[t][i] = 0
                for j in self.state:
                    b[t][i] = b[t][i] + self.A[i][j] * self.B[j][observation[t + 1]] * b[t + 1][j]
                b[t][i] = c[t] * b[t][i]
        return b

    def fitOnce(self,observation):

        T = len(observation)
        gamma = [{} for t in range(T)]
        zi = [{} for t in range(T - 1)]
        for t in range(T - 1):
            for i in self.state:
                zi[t][i] = {}
        a,c=self.forward(observation)
        b=self.backward(observation,c)

        for t in range(T - 1):
            for i in self.state:
                gamma[t][i] = 0
                for j in self.state:
                    zi[t][i][j] = a[t][i] * self.A[i][j] * self.B[j][observation[t + 1]] * b[t + 1][j]
                    gamma[t][i] = gamma[t][i] + zi[t][i][j]
        for i in self.state:
            gamma[T - 1][i] = a[T - 1][i]
        for i in self.state:
            self.PI[i] = gamma[0][i]
        for i in self.state:
            denom = 0
            for t in range(T - 1):
                denom = denom + gamma[t][i]
            for j in self.state:
                numer = 0
                for t in range(T - 1):
                    numer = numer + zi[t][i][j]
                self.A[i][j] = numer / denom

        for i in self.state:
            denom = 0
            for t in range(T):
                denom = denom + gamma[t][i]
            for j in self.symbol:
                numer = 0
                for t in range(T):
                    if (observation[t] == j):
                        numer = numer + gamma[t][i]
                self.B[i][j] = numer / denom
        return c



    def fit(self,maxIters,observations):
        obs=observations
        oldLogProb = float("-inf")
        for i in range(maxIters):
            np.random.shuffle(obs)
            observation=functools.reduce(lambda x,y:x+y,obs)
            #length = int(round(0.5*len(observation)))

            c_score= self.fitOnce(observation)

            self.logProb = 0
            for c in c_score:
                self.logProb = self.logProb + math.log(c)
            self.logProb = -self.logProb

            print("score:"+str(self.logProb))
            if oldLogProb< self.logProb:
                oldLogProb = self.logProb
            else:
                break


    def score(self,observation):

        a,c_list=self.forward(observation)
        logProb = 0
        for c in c_list:
            logProb = logProb + math.log(c)

        return -logProb

    def save_model(self,path):
        model={}
        model["PI"]=self.PI
        model["A"]=self.A
        model["B"]=self.B
        model["state"]=self.state
        model["symbol"]=self.symbol
        with open(os.path.abspath(path+"/"+self.name+"_"+str(round(self.logProb))+".txt"),'w') as f:
            json.dump(model,f)

    def load_model(self,path,name):
        with open(os.path.abspath(path+"/"+name),'r') as f:
            model=json.load(f)
            self.PI=model["PI"]
            self.A=model["A"]
            self.B=model["B"]
            self.state=model["state"]
            self.symbol=model["symbol"]

    def score_list(self,observations):
        result=[]
        for observation in observations:
            result.append(self.score(observation))
        return result