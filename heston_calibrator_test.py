# -*- coding: utf-8 -*-
"""
Created on Thu May  4 19:42:46 2023

@author: XYZW
"""
import QuantLib as ql
from collections import namedtuple
import numpy as np
import pandas as pd
#%%
def create_heston_helpers(S0,data,rf,dvd,init_sol):
    """
    Inputs:
        init_sol: v0,kappa,theta,sigma,rho
    """
    cal = ql.TARGET()
    dc = ql.Actual360()
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(S0))
    theta, kappa,sigma,rho,v0 = init_sol
    today = ql.Date().todaysDate()
    riskFreeCurve = ql.FlatForward(today, rf, dc)
    dividendCurve = ql.FlatForward(today, dvd, dc)
    riskFreeHandle = ql.YieldTermStructureHandle(riskFreeCurve)
    dividendHandle = ql.YieldTermStructureHandle(dividendCurve)
    helpers = [ql.HestonModelHelper(d.expiry, cal, spot, d.strike,
                ql.QuoteHandle(ql.SimpleQuote(d.vol)), riskFreeHandle, 
                dividendHandle) for d in data]
    process = ql.HestonProcess(riskFreeHandle,dividendHandle,spotHandle,
                               v0,kappa,theta,sigma,rho)
    
    hestonModel = ql.HestonModel(process)
    for d in helpers:
        d.setPricingEngine(ql.AnalyticHestonEngine(hestonModel))
    return helpers

def heston_calibrator(spot,strikes,tenors,vols,rf,dvd,init_sol,tol = 1.0e-6, 
                      no_iter = 100,stat_iter = 50):
    """
    Vols: Market vols
    
    Given a spot price: float/int
    
    A list of strikes, tenors, vols 
    
    + risk free rate: rf
    
    dvd: dividend rate
    
    init_sol = initial guess
    
    Returns:
        params: 5 parameters of the heston Model.
    """
    cal = ql.TARGET()
    dc = ql.Actual360()
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(spot))
    
    volQuotes = [ql.QuoteHandle(ql.SimpleQuote(vols[i])) for i in range(len(vols))]
    theta, kappa,sigma,rho,v0 = init_sol
    today = ql.Date().todaysDate()
    
    riskFreeCurve = ql.FlatForward(today, rf, dc)
    dividendCurve = ql.FlatForward(today, dvd, dc)
    riskFreeHandle = ql.YieldTermStructureHandle(riskFreeCurve)
    dividendHandle = ql.YieldTermStructureHandle(dividendCurve)
    
    helpers = [ql.HestonModelHelper(tenors[i], cal, spot, strikes[i],
                volQuotes[i], riskFreeHandle, dividendHandle) 
               for i in range(len(vols))]
    lm = ql.LevenbergMarquardt(tol,tol,tol)
    process = ql.HestonProcess(riskFreeHandle,dividendHandle,spotHandle,
                               v0,kappa,theta,sigma,rho)
    
    hestonModel = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(hestonModel)
    
    for i in range(len(helpers)):
        helpers[i].setPricingEngine(engine)
    
    hestonModel.calibrate(helpers,lm, ql.EndCriteria(no_iter,stat_iter,1.0e-8,
                                               1.0e-8,1.0e-8))
    params = list(hestonModel.params())#theta, kappa,sigma,rho,v0
    return params

def heston_calibrator2(spot,strikes,tenors,mkt_prices,rf,dvd,init_sol,tol = 1.0e-6, 
                      no_iter = 100,stat_iter = 50):
    cal = ql.TARGET()
    dc = ql.Actual360()
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(spot))
    
    volQuotes = [ql.QuoteHandle(ql.SimpleQuote(vols[i])) for i in range(len(vols))]
    theta, kappa,sigma,rho,v0 = init_sol
    today = ql.Date().todaysDate()
    
    riskFreeCurve = ql.FlatForward(today, rf, dc)
    dividendCurve = ql.FlatForward(today, dvd, dc)
    riskFreeHandle = ql.YieldTermStructureHandle(riskFreeCurve)
    dividendHandle = ql.YieldTermStructureHandle(dividendCurve)
    
    helpers = [ql.HestonModelHelper(tenors[i], cal, spot, strikes[i],
                volQuotes[i], riskFreeHandle, dividendHandle) 
               for i in range(len(vols))]
    lm = ql.LevenbergMarquardt(tol,tol,tol)
    process = ql.HestonProcess(riskFreeHandle,dividendHandle,spotHandle,
                               v0,kappa,theta,sigma,rho)
    
    hestonModel = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(hestonModel)
    
    for i in range(len(helpers)):
        helpers[i].setPricingEngine(engine)
    
    hestonModel.calibrate(helpers,lm, ql.EndCriteria(no_iter,stat_iter,1.0e-8,
                                               1.0e-8,1.0e-8))
    params = list(hestonModel.params())#theta, kappa,sigma,rho,v0
    return params

def create_heston_report(spot,strikes,tenors,vols,rf,dvd,v0,kappa,theta,sigma,rho):
    init_sol = [theta,kappa,sigma,rho,v0]
    data = [CalibrationData(strikes[i],tenors[i],vols[i]) 
            for i in range(len(strikes))]
    params = heston_calibrator(spot,strikes,tenors,vols,rf,dvd,init_sol)
    helpers_heston = create_heston_helpers(spot,data,rf,dvd,params)
    price_errors = [helpers_heston[i].modelValue()/helpers_heston[i].marketValue()-1
                    for i in range(len(helpers_heston))]
    market_prices = [helpers_heston[i].marketValue() 
                     for i in range(len(helpers_heston))]
    model_prices = [helpers_heston[i].modelValue() 
                     for i in range(len(helpers_heston))]
    model_vols = [helpers_heston[i].impliedVolatility(model_prices[i], 
                            1.0e-6, 50, 0.01, 1.00) for i in range(len(helpers_heston))]
    vol_errors = [abs(vols[i]-model_vols[i])/vols[i] for i in range(len(vols))]
    df = pd.DataFrame(data = np.array([market_prices,model_prices,vols,model_vols,price_errors,
                              vol_errors]).T,index = list(range(len(model_vols))),
                              columns = ['Market Prices','Model prices',
                                         'Market vols','Model vols',
                                         'Price errors','Vol errors'])
    return df

#%%
from collections import namedtuple
today = ql.Date().todaysDate()
CalibrationData = namedtuple("CalibrationData","strike,expiry,vol")
dc = ql.Actual360()
v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;

rf,dvd = 0.04,0.00
init_sol = [theta, kappa,sigma,rho,v0]
spot = 100;strikes = [100,105,110,90,95]; tenors = [ql.Period("3M")]*5;

vols = [0.2,0.22,0.25,0.3,0.32];
data = [CalibrationData(strikes[i],tenors[i],vols[i])  for i in range(len(strikes))]
params = heston_calibrator(spot,strikes,tenors,vols,0.04,0.00,init_sol)
print("Params: theta, kappa, sigma, rho, v0:",list(params))

#%%
helpers_heston = create_heston_helpers(spot,data,rf,dvd,init_sol)
helpers_heston2 = create_heston_helpers(spot,data,rf,dvd,params)
#%%
price_errors = [helpers_heston[i].modelValue()/helpers_heston[i].marketValue()-1
                for i in range(len(helpers_heston))]
price_errors2 = [helpers_heston2[i].modelValue()/helpers_heston2[i].marketValue()-1
                for i in range(len(helpers_heston))]

market_prices = [helpers_heston[i].marketValue() 
                 for i in range(len(helpers_heston2))]
model_prices = [helpers_heston[i].modelValue() 
                 for i in range(len(helpers_heston2))]
model_prices2 = [helpers_heston2[i].modelValue() 
                 for i in range(len(helpers_heston))]
model_vols = [helpers_heston[i].impliedVolatility(model_prices[i], 
                        1.0e-6, 50, 0.01, 1.00) for i in range(len(helpers_heston))]

model_vols2 = [helpers_heston2[i].impliedVolatility(model_prices2[i], 
                        1.0e-6, 50, 0.01, 1.00) for i in range(len(helpers_heston))]

vol_errors = [abs(vols[i]-model_vols[i])/vols[i] for i in range(len(vols))]
print("Model volatilities",model_vols)
print("Market volatilities",vols)
#%%
print("Volatility errors",vol_errors)
print("\n Model prices before calibration",model_prices)
print("\n Model price after calibration",model_prices2)
print("\n Model vols after calibration",model_vols2)
#%%
df_Heston = create_heston_report(spot,strikes,tenors,vols,rf,dvd,v0,
                                 kappa,theta,sigma,rho)
strikes2 = [100,105,110,90,95,95]
tenors2 = [ql.Period("3M")]*5+[ql.Period("6M")]
vols2 = vols+[0.35]
df_Heston2 =  create_heston_report(spot,strikes2,tenors2,vols2,rf,dvd,v0,
                                 kappa,theta,sigma,rho)
print(np.sum(df_Heston['Model prices']/df_Heston['Market Prices']-1)**2)
print(np.sum(df_Heston2['Model prices']/df_Heston2['Market Prices']-1)**2)


