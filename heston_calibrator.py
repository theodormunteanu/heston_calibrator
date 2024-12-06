# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:42:24 2024

@author: XYZW
"""

import QuantLib as ql
import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\equity exotics')
import price_BS as opt_BS
from collections import namedtuple
CalibrationData = namedtuple("CalibrationData","strike,expiry,vol")
import scipy.optimize as opt
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
    helpers = [ql.HestonModelHelper(d.expiry, cal, S0, d.strike,
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
    Spot: Current underlying value
    
    Strikes: list of values
    
    Vols: Market vols (vols)
    
    RETURNS:
        theta, kappa, sigma, rho, v0 (in this order)
        
        Theta: long term average of the variance
        
        Kappa: Speed of return to the average
        
        rho: correlation between the brownian motions
        
        v0: initial variance
        
        sigma: vol of vol
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
    
    if np.all([isinstance(tenors[i],(float,int)) for i in range(len(tenors))]):
        tenors = [ql.Period(int(tenors[i]*365),ql.Days) for i in range(len(tenors))]
    
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
    params = list(hestonModel.params())
    "Params = theta, kappa, sigma,rho,v0"
    return params

def heston_calibrator2(spot,strikes,tenors,mkt_prices,rf,dvd,init_sol,tol = 1.0e-6, 
                      no_iter = 100,stat_iter = 50):
    """
    Parameters:
        S0: float. 
        
        strikes = list of numbers. 
    """
    cal = ql.TARGET()
    dc = ql.Actual360()
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(spot))
    if np.all([isinstance(tenors[i],(float,int)) for i in range(len(tenors))]):
        Ts = tenors
        tenors = [ql.Period(int(tenors[i]*365),ql.Days) for i in range(len(tenors))]
    
    
    vols = [opt_BS.implied_vol(spot,strikes[i],rf,Ts[i],mkt_prices[i]) 
            for i in range(len(strikes))]
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

def heston_pricer(S0,K,rf,T,v0,kappa,theta,sigma,rho,dvd = 0,ret = 'value',
                  today = ql.Date.todaysDate()):
    """
    v0: initial variance rate
    
    theta: long term variance rate
    
    rho: correlation between the returns of stock prices and volatilities
    
    sigma: vol of vol.
    
    kappa: speed of reversion. 
    """
    tenor = ql.Period(int(T*365),ql.Days)
    calendar = ql.TARGET()
    dc = ql.Actual360()
    
    riskFreeCurve = ql.FlatForward(today, rf, dc)
    dividendCurve = ql.FlatForward(today, dvd, dc)
    
    riskFreeHandle = ql.YieldTermStructureHandle(riskFreeCurve)
    dividendHandle = ql.YieldTermStructureHandle(dividendCurve)
    
    volQuote = ql.QuoteHandle(ql.SimpleQuote(sigma))
    heston_helper = ql.HestonModelHelper(tenor, calendar, S0, K, volQuote, 
                         riskFreeHandle, dividendHandle)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    
    heston_process = ql.HestonProcess(riskFreeHandle,dividendHandle,spot_handle,
                                      v0,kappa,theta,sigma,rho)
    
    heston_model = ql.HestonModel(heston_process)
    hestonEngine = ql.AnalyticHestonEngine(heston_model)
    heston_helper.setPricingEngine(hestonEngine)
    if ret=='value':
        return heston_helper.modelValue()
    else:
        return heston_helper
    
def heston_calibrator3(S0,rf,strikes,expiries,mkt_prices,init_sol,dvd = 0.0):
    heston_func = lambda x: np.linalg.norm(np.array([heston_pricer(S0,strikes[i],rf,expiries[i],*x)
                                    for i in range(len(strikes))])-mkt_prices,2)
    return opt.minimize(heston_func,init_sol,bounds = [(0.01,1)]*4+[(-1,1)])

def heston_pricing_vanilla(S0,ref_date,v0,kappa,theta,sigma,rho,rf,dvd,maturity,K,
                           option_type = 'call',exercise_type = 'european',
                           ex_dates = 0):
    
    initialValue = ql.QuoteHandle(ql.SimpleQuote(S0))
    
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(ref_date, rf, 
                            ql.Actual365Fixed()))
    
    dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(ref_date, dvd, 
                                                ql.Actual365Fixed()))
    
    hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, 
                                     v0, kappa, theta, sigma, rho)
    
    hestonModel = ql.HestonModel(hestonProcess)
    
    hestonEngine = ql.FdHestonVanillaEngine(hestonModel)
    
    if option_type in ['Call','call']:
        option_type = ql.Option.Call
    else:
        option_type = ql.Option.Put
    
    payoff = ql.PlainVanillaPayoff(option_type, K)
    
    if isinstance(maturity,(int,float)):
        maturity = ref_date + ql.Period(int(maturity*365),ql.Days)
    
    print("Maturity",maturity)
    
    if exercise_type in ['European','european','eur']:
        exercise = ql.EuropeanExercise(maturity)
    elif exercise_type in ['American','american','amer']:
        exercise = ql.AmericanExercise(ref_date,maturity)
    elif exercise_type in ['Bermudan','bermudan','berm']:
        exercise = ql.BermudanExercise(ex_dates)

    option = ql.VanillaOption(payoff,exercise)
    
    option.setPricingEngine(hestonEngine)
    
    return option.NPV()


def create_heston_report(spot,strikes,tenors,vols,rf,dvd,v0,kappa,theta,sigma,rho):
    init_sol = [theta,kappa,sigma,rho,v0]
    data = [CalibrationData(strikes[i],tenors[i],vols[i]) 
            for i in range(len(strikes))]
    params = heston_calibrator(spot,strikes,tenors,vols,rf,dvd,init_sol)
    
    helpers_heston = create_heston_helpers(spot,data,rf,dvd,params)
    
    price_errors = [helpers_heston[i].modelValue()/helpers_heston[i].marketValue()-1
                    for i in range(len(helpers_heston))]
    
    market_prices = [helpers_heston[i].marketValue() for i in range(len(helpers_heston))]
    model_prices = [helpers_heston[i].modelValue() for i in range(len(helpers_heston))]
    
    model_vols = [helpers_heston[i].impliedVolatility(model_prices[i], 
                            1.0e-6, 50, 0.01, 1.00) for i in range(len(helpers_heston))]
    
    vol_errors = [abs(vols[i]-model_vols[i])/vols[i] for i in range(len(vols))]
    
    df = pd.DataFrame(data = np.array([market_prices,model_prices,vols,model_vols,price_errors,
                              vol_errors]).T,index = list(range(len(model_vols))),
                              columns = ['Market Prices','Model prices',
                                         'Market vols','Model vols',
                                         'Price errors','Vol errors'])
    return df
