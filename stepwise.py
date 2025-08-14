from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import SVG, HTML
import copy as copy


_model_params_doc = """
    Parameters
    ----------
    data: DataFrame
        Raw Data,
    target: array-like
        The dependent variable, dim = n*1
    xnames: array-like
        The independnet variable
    weight: array-like
        Each observation in the input data set is weighted by the value of the WEIGHT variable. By default, weight is np.ones(n)
    method: ['forward', 'backward', 'stepwise']
        The default selection method is 'stepwise'
    maxiter: int
        maxiter = 25 (default)
    mindiff: float
        mindiff = 1e-8 (default)
    """
_models_Result_docs = """
    params: array
        Parameters' Estimates
    AIC: float
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
    BIC: float
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
    SC: float
        Schwarz criterion. `-LogL + p*(log(nobs))`
    std_error: Array
        The standard errors of the coefficients.(bse)
    Chi_Square: float
        Wald Chi-square : (logit_res.params[0]/logit_res.bse[0])**2
    Chisqprob: float
        P-value from Chi_square test statistic 
    llf: float
        Value of the loglikelihood, as (LogL)
    """    
# Newton-Raphson Iteration
 
class Logistic(object):
    __doc__ = """
    The Logistic Regression Model.
    %(Params_doc)s
    %(Result_doc)s
    Notes
    ----
    """ % {'Params_doc' : _model_params_doc, 
           'Result_doc': _models_Result_docs}
    
    def __init__(self, data, xnames, target, **kwargs):
        data = data[xnames + [target]]
        self.nrows = len(data)
        self.ncols = len(xnames)
        data['const'] = np.ones(self.nrows)

        if 'weight' in kwargs.keys():
            data['weight'] = data[weight]
        else :
            data['weight'] = np.ones(self.nrows)
        self.data = data
        self.const = True  # default self.cont = True, include intercept in model  
        self.xcols = xnames
        self.target = target

        if 'method' in kwargs.keys():
            self.method = kwargs['method']
        else:
            self.method = 'stepwise'

        if 'mindiff' in kwargs.keys():
            self.mindiff = kwargs['mindiff']
        else:
            self.mindiff = 1e-8 # default mindiff
      
        if 'maxiter' in kwargs.keys():
            self.maxiter = kwargs['maxiter']
        else:
            self.maxiter = 25   # default maxiter

    def forward(self, **kwargs):
        if 'slentry' in kwargs.keys():
            self.slentry = float(kwargs['slentry'])
        else:
            self.slentry = 0.05  # default
        xenter = ['const']
        xwait = self.xcols
        for i in range(self.ncols):
            wald_score = list()
            wald_pvalue = list() 
            for var in xwait:
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[xenter + [var]], 
                            freq_weights=self.data['weight'],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                # print(logitmodel.summary())
                wald_test = logitmodel.wald_test(var)
                wald_score.append(wald_test.statistic[0][0])
                wald_pvalue.append(wald_test.pvalue)
            checkenter = checkio(xwait, wald_score, wald_pvalue)
            checkenter.enter()
            min_pvalue = min(wald_pvalue)
            var_enter = xwait[wald_pvalue.index(min_pvalue)]
            if min_pvalue < self.slentry:
                print("STEP %s: Var %s Entered " % (i, var_enter))
                xenter.append(var_enter)
                xwait.remove(var_enter)
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[xenter], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
            else:
                print("         No (additional) Variables met the %s significance level for enter into the model"%(self.slentry))
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[xenter], 
                            freq_weights=self.data['weight'],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
                break
        return logitmodel


    def backward(self, **kwargs):
        if 'slstay' in kwargs.keys():
            self.slstay = float(kwargs['slstay'])
        else:
            self.slstay  = 0.05  # default

        if 'xvars' in kwargs.keys():
            xcols = kwargs['xvars']
        else:
            xcols = self.xcols

        if 'step' in kwargs.keys():
            step = kwargs['step']
        else:
            step = 0

        xstay = copy.copy(xcols)
        logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[['const'] + xstay], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
        print(logitmodel.summary())
        while True:
            wald_test_res = logitmodel.wald_test_terms()
            wald_test_res = wald_test_res.summary_frame().sort_values('P>chi2', ascending=False)
            # check_stay = checkio(xwait, wald_score, wald_pvalue)
            # check_stay.remove()
            # print(wald_test_res)
            wald_test_res.drop("const", inplace=True)
            max_pvalue = wald_test_res['P>chi2'][0]
            var_remove = wald_test_res.index[0]

            if max_pvalue > self.slstay:
                print(" **** BackWard ****" )
                print(" STEP %s: Var %s Removed " % (step, var_remove))
                xstay.remove(var_remove)
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[['const'] + xstay], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
                step += 1
            else:
                print("**** BackWard ****" )
                print(" STEP %s: No (additional) Variables met the %s significance level for remove into the model" % (step, self.slstay))
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[['const'] + xstay], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
                break
            if len(xstay) == 1:
                break         
        return logitmodel  
            

    def stepwise(self, **kwargs):
        if 'slentry' in kwargs.keys():
            self.slentry = float(kwargs['slentry'])
        else:
            self.slentry = 0.05  # default
        if 'slstay' in kwargs.keys():
            self.slstay = float(kwargs['slstay'])
        else:
            self.slstay  = 0.05  # default

        if "early_stop_step_num" in kwargs.keys():
            self.early_stop_step_num = float(kwargs['early_stop_step_num'])
        else:
            self.early_stop_step_num = float("inf")
            
        const = ['const']
        xenter = list()
        xwait = self.xcols
        step = 0
        for i in range(self.ncols):
            if step == 0:
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data['const'], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
            else:
                logitmodel = self.backward(xvars=xenter, slentry=self.slentry, slstay=self.slstay)
                wald_test_res = logitmodel.wald_test_terms()
                wald_test_res = wald_test_res.summary_frame()
                xenter = [var for var in wald_test_res.index if var != 'const']
                if xin not in xenter:
                    print("Note: Model building terminates because the last effect entered is removed by the Wald statistic criterion.")
                    break
            wald_score  = list()
            wald_pvalue = list()
            for xvar in xwait:
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[const + xenter + [xvar]], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                # print(logitmodel.summary())
                wald_test = logitmodel.wald_test(xvar)
                wald_score.append(wald_test.statistic[0][0])
                wald_pvalue.append(wald_test.pvalue)
            min_pvalue = min(wald_pvalue)
            checkenter = checkio(xwait, wald_score, wald_pvalue)
            checkenter.enter()
            if min_pvalue < self.slentry:    
                xin = xwait[wald_pvalue.index(min_pvalue)]
                xenter = xenter+[xin]
                xwait.remove(xin)
                print("** step %s: %s entered:\n"%(step, xin))
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[const + xenter], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
            else:
                logitmodel = sm.GLM(endog=self.data[self.target], 
                            exog=self.data[const + xenter], 
                            freq_weights=self.data["weight"],
                            family = sm.families.Binomial(link = sm.families.links.logit())).fit(disp=0)
                print(logitmodel.summary())
                print("step %s: None of the remaining variables outside the model meet the entry criterion, and the stepwise selection is terminated. " % step)             
                break
            step += 1
            ## xmk: early stop
            if step >= self.early_stop_step_num:
                break
        return logitmodel

             
class checkio(object):
    def __init__(self, xwait, score, pvalue):
        self.xwait = xwait
        self.score = score
        self.pvalue= pvalue
    def enter(self):
        print("              Analysis of Variables Eligible for Entry  ")
        print("==============================================================================")
        print("  %5s\t \t%5s\t   \t  \t%5s\t" % ("variable", "Wald Chi-square", "Pr>ChiSq"))
        for i,v in enumerate(self.xwait):
            print("  %5s\t             \t%10s\t     \t%10s\t" % (v, self.score[i], self.pvalue[i]))
        print(" ") 
    def remove(self):
        print("              Analysis of Variables Eligible for Remove  ")
        print("==============================================================================")
        print("  %5s\t \t%5s\t   \t  \t%5s\t" % ("variable", "Wald Chi-square", "Pr>ChiSq"))
        for i,v in enumerate(self.xwait):
            print("  %5s\t             \t%10s\t     \t%10s\t" % (v, self.score[i], self.pvalue[i]))
        print(" ")
