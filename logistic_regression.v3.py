import numpy as np
import pandas as pd
import statsmodels.api as sm
import pdb
import math
loansDataClean = pd.read_csv('loansData_clean.csv')
IR_TF = loansDataClean['Interest.Rate'].map(lambda x: x<0.12)
def T_F(dt):
	if (dt):
		return 1
	else:
		return 0
loansDataClean['IR_TF'] = loansDataClean['Interest.Rate']
loansDataClean['IR_TF'] = IR_TF
loansDataClean['IR_01'] = loansDataClean['IR_TF'].apply(T_F)
def one(n):
	return 1
loansDataClean['Constant_Int'] = loansDataClean['IR_TF'].apply(one)
Ind_Vars = ['FICO.Score', 'Amount.Requested', 'Constant_Int']
logit = sm.Logit(loansDataClean['IR_01'], loansDataClean[Ind_Vars])
result = logit.fit()
coeff = result.params
def calc_int_rate(params, fico_score, loan_amount):
	interest_rate = params['Constant_Int'] + params['FICO.Score'] * fico_score + params['Amount.Requested'] * loan_amount
	return abs(interest_rate)
def logistic_function_prob(interest_rate):
	p = 1/(1 + math.exp(-(interest_rate)))
	return p
interest_rate = calc_int_rate(coeff, 735, 20000)
logistic_function_prob(interest_rate)