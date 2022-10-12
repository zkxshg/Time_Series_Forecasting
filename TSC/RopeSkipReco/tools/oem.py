""" ONLINE ELASTIC MEASURES
"""
import numpy as np
'''
Online Elastic Measure
'''
class OEM:
    '''
    lX:    length of the memory for X series
    lY:    length of the memory for Y series
    w:     weight
    rateX: rate at which X time series is generated
    rateY: rate at which Y time series is generated
    dist:  type of distance
    '''
    def __init__(self, w, lX=10, lY=None, rateX=1, rateY=None, dist='euclidean'):

        self.dist = dist

        self.lX = lX
        if lY is None:
            self.lY = lX
        else:
        	self.lY = lY

        self.rateX = rateX
        if rateY is None:
            self.rateY = rateX
        else:
        	self.rateY = rateY

        self.w = w

        self.m = 0
        self.n = 0

    '''
    Add two (nonempty) chunks of the time series X, Y

    X, Y: last measures of the time series (vector, numpy.array)
    Warning: the time series have to be non-empty
    (at least composed by a single measure)

      -----------
    X | XS | XY |
      -----------
    R | RS | RY |
      -----------
        S    Y
    '''
    def addChunk(self,X,Y):
        m,n = X.size,Y.size

        # Solve XS
        dtwXSv, dtwXSh = self._solveXS(X,self.S,self.dtwS,iniI=self.m,iniJ=self.n-self.lY)
        # Solve RY
        dtwRYh, dtwRYv = self._solveRY(self.R,Y,self.dtwR,iniI=self.m-self.lX,iniJ=self.n)
        # Solve XY
        dtwXYh, dtwXYv = self._solveXY(X,Y,dtwXSv,self.dtwS[-1],dtwRYh,iniI=self.m,iniJ=self.n)

        # Save the statistics, size O(self.lX + self.lY)
        if(m < self.lX):
            self.R = np.concatenate((self.R[-(self.lX-m):], X))
            self.dtwR = np.concatenate((dtwRYv[-(self.lX-m):],dtwXYv))
        else:
            self.R = X[m-self.lX:m]
            self.dtwR = dtwXYv[m-self.lX:m]

        if(n < self.lY):
            self.S = np.concatenate((self.S[-(self.lY-n):], Y))
            self.dtwS = np.concatenate((dtwXSh[-(self.lY-n):],dtwXYh))
        else:
            self.S = Y[n-self.lY:n]
            self.dtwS = dtwXYh[n-self.lY:n]

        # Save the starting index of the next chunk
        m += self.m
        n += self.n
        self.m = np.max([0,m-n])
        self.n = np.max([0,n-m])

        # return dtwXYv[-1]
        return min(np.concatenate((self.dtwR, self.dtwS)))

    def addPoint(self,x,y):
        m, n = len(self.R), len(self.S)

        X = np.array([x])
        Y = np.array([y])
        # Solve XS
        dtwXSv, dtwXSh = self._solveXS(X,self.S,self.dtwS,iniI=self.m,iniJ=self.n-self.lY)
        # Solve RY
        dtwRYh, dtwRYv = self._solveRY(self.R,Y,self.dtwR,iniI=self.m-self.lX,iniJ=self.n)
        # Solve XY (last row and last column)
        dtwXYh, dtwXYv = self._solveXY(X,Y,dtwXSv,self.dtwS[-1],dtwRYh,iniI=self.m,iniJ=self.n)

        if m < self.lX:
            self.R = np.concatenate((self.R, X))
            self.dtwR = np.concatenate((self.dtwR,dtwXYv))
        else:
            self.R = np.concatenate((self.R[1:], X))
            self.dtwR = np.concatenate((self.dtwR[1:],dtwXYv))

        if n < self.lY:
            self.S = np.concatenate((self.S, Y))
            self.dtwS = np.concatenate((self.dtwS,dtwXYh))
        else:
            self.S = np.concatenate((self.S[1:], Y))
            self.dtwS = np.concatenate((self.dtwS[1:],dtwXYh))

        # return dtwXYv[-1]
        return min(np.concatenate((self.dtwR, self.dtwS)))

    '''
    X,S: to (partial) time series
     -----------
    X| XS | XY |
     -----------
    R|prev| RY |
     -----------
       S    Y

    dtwS: partial solutions to DTW(R,S)
    iniI: Index of the first point of X in the complete time series
    iniJ: Index of the first point of S in the complete time series

    * Warning *: X and S have to be nonempty (partial) series
    Falta: tener en cuenta rateX y rateY
    '''
    def _solveXS(self,X,S,dtwS,iniI=0,iniJ=0):

    	# Compute point-wise distance
    	if self.dist == 'euclidean':
    		S_tmp, X_tmp = np.meshgrid(S,X)
        	XS = np.sqrt((X_tmp - S_tmp)**2)

        m,s = X.size,S.size

        # Solve first row
        XS[0,0] += (self.w if iniI <= iniJ else 1)*dtwS[0]
        for j in range(1,s):
            XS[0,j] += np.min([
            	(self.w if (iniJ+j) <= iniI else 1)*XS[0,j-1],
            	self.w*dtwS[j-1],
            	(self.w if iniI <= (iniJ+j) else 1)*dtwS[j]])
        # Solve first column
        for i in range(1,m):
            XS[i,0] += XS[i-1,0]

        # Solve the rest
        for i in range(1,m):
            for j in range(1,s):
                XS[i,j] += np.min([
                	(self.w if (iniI+i <= iniJ+j) else 1)*XS[i-1,j],
                	self.w*XS[i-1,j-1],
                	(self.w if (iniJ+j <= iniI+i-1) else 1)*XS[i,j-1]])

        return XS[:,-1], XS[-1,:]

    def _solveRY(self,R,Y,dtwR,iniI=0,iniJ=0):

    	if self.dist == 'euclidean':
    		Y_tmp, R_tmp = np.meshgrid(Y,R)
        	RY = np.sqrt((R_tmp - Y_tmp)**2)

        r,n = R.size,Y.size

        # First first column
        RY[0,0] += (self.w if iniJ <= iniI else 1)*dtwR[0]
        for i in range(1,r):
            RY[i,0] += np.min([
            	(self.w if (iniI+i) <= iniJ else 1)*RY[i-1,0],
                self.w*dtwR[i-1],
                (self.w if iniJ <= iniI+i else 1)*dtwR[i]])

        # Solve first row
        for j in range(1,n):
            RY[0,j] += RY[0,j-1]

        # Solve the rest
        for i in range(1,r):
            for j in range(1,n):
                RY[i,j] += np.min([
                	(self.w if (iniI+i) <= (iniJ+j) else 1)*RY[i-1,j],
                	self.w*RY[i-1,j-1],
                	(self.w if (iniJ+j) <= (iniI+i) else 1)*RY[i,j-1]])
        return RY[-1,:], RY[:,-1]

    '''
    X,S: (partial) time series
     -----------
    X| XS | XY |
     -----------
    R| RS | RY |
     -----------
       S    Y

    dtwRS: solution to DTW(R,S)
    dtwXS: partial solutions to DTW(X,S), DTW(X,S)[:,-1]
    dtwRY: partial solutions to DTW(R,Y), DTW(R,Y)[-1,:]
    iniI: Index of the first point of X in the complete time series
    iniJ: Index of the first point of Y in the complete time series

    * Warning *: X and Y have to be nonempty (partial) series
    Falta: tener en cuenta rateX y rateY
    '''
    def _solveXY(self,X,Y,dtwXS,dtwRS,dtwRY,iniI=0,iniJ=0):

    	# Compute point-wise distance
    	if self.dist == 'euclidean':
    		Y_tmp, X_tmp = np.meshgrid(Y,X)
        	XY = np.sqrt((X_tmp - Y_tmp)**2)

        m,n = X.size,Y.size

        # Solve the first point
        XY[0,0] += np.min([(self.w if iniJ <= iniI else 1)*dtwXS[0],
                          self.w*dtwRS,
                          (self.w if iniI <= iniJ else 1)*dtwRY[0]])
        # Solve first row
        for j in range(1,n):
        	XY[0,j] += np.min([
        		(self.w if (iniJ+j) <= iniI else 1)*XY[0,j-1],
        		self.w*dtwRY[j-1],
        		(self.w if iniI <= (iniJ+j) else 1)*dtwRY[j]])

        # First first column
        for i in range(1,m):
        	XY[i,0] += np.min([
        		(self.w if (iniI+i) <= iniJ else 1)*XY[i-1,0],
        		self.w*dtwXS[i-1],
        		(self.w if iniJ <= iniI+i else 1)*dtwXS[i]])

        # Solve the rest
        for i in range(1,m):
            for j in range(1,n):
                XY[i,j] += np.min([(self.w if (iniI+i) <= (iniJ+j) else 1)*XY[i-1,j],
                                  self.w*XY[i-1,j-1],
                                  (self.w if (iniJ+j) <= (iniI+i) else 1)*XY[i,j-1]])
        return XY[-1,:], XY[:,-1]

        '''
    R,S: to (partial) time series
     -----------
    X| XS | XY |
     -----------
    R|prev| RY |
     -----------
       S    Y

    * Warning *: X and S have to be nonempty (partial) series
    Falta: tener en cuenta rateX y rateY
    '''
    def _solveRS(self,R,S,iniI=0,iniJ=0):

    	# Compute point-wise distance
    	if self.dist == 'euclidean':
    		S_tmp, R_tmp = np.meshgrid(S,R)
        	RS = np.sqrt((R_tmp - S_tmp)**2)

        r,s = R.size,S.size

        # Solve first row
        for j in range(1,s):
            RS[0,j] += (self.w if (iniJ+j) <= iniI else 1)*RS[0,j-1]

        # First first column
        for i in range(1,r):
            RS[i,0] += (self.w if (iniI+i) <= iniJ else 1)*RS[i-1,0]

        # Solve the rest
        for i in range(1,r):
        	for j in range(1,s):
        		RS[i,j] += np.min([
        			(self.w if (iniI+i <= iniJ+j) else 1)*RS[i-1,j],
        			self.w*RS[i-1,j-1],
        			(self.w if (iniJ+j <= iniI+i-1) else 1)*RS[i,j-1]])

        # Save the statistics, size O(self.lX + self.lY)
        if self.lX >= r:
        	self.R = R
        	self.dtwR = RS[:,-1]
        else:
        	self.R = R[-self.lX:]
        	self.dtwR = RS[-self.lX:,-1]

        if self.lY >= s:
        	self.S = S
        	self.dtwS = RS[-1,:]
        else:
        	self.S = S[-self.lY:]
        	self.dtwS = RS[-1,-self.lY:]

     	return RS[-1,-1]


'''
Heuristic for setting the weight of OEM
'''
def getW(s=100,p=0.001):
    return np.power(p,1.0/s)
