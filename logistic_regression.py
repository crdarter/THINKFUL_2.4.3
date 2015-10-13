import pandas as pd
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = cleanInterestRate
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
loansData['Loan.Length'] = cleanLoanLength
loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]
loansData['constant'] = 1
IR_TF = loansData['Interest.Rate'].map(lambda x: int(x >= .12))
loansData['IR_TF'] = IR_TF
loansData.to_csv('loansData_clean.csv', header=True, index=False)
ind_vars = ['Loan.Length', 'FICO.Score', 'constant']


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

import pandas as pd
df = pd.read_csv("loansData_clean.csv")
ind_vars = ['Loan.Length', 'FICO.Score', 'constant']

logit = sm.Logit(df['IR_TF'], df[ind_vars])
result = logit.fit()

coeff = result.params
print coeff
import math 
import pylab

FicoScore = 720
LoanAmount = 10000
logistic_function = 1/(1 + math.exp(coeff[2] + coeff[1]*(FicoScore) - coeff[0]*(LoanAmount)))

print logistic_function

x = np.linspace(0,950,1000) 
y = 1/(1 + np.exp(coeff[2] + coeff[1]*(x) - coeff[0]*(10000)))
pylab.plot(x,y)
pylab.show()
