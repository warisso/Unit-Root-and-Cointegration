# Esquema para realizar el test de raices unitarias y cointegracion en series de tiempo
# Basado en los datos de productividad laboral (y) y salarios reales (w) de Mexico
# Brida, J.,  Adrian Risso, W., & Carrera, E. (2010). Real wages as determinant of labour productivity in the Mexican tourism sector.
# European journal of tourism research, 3(1), 67-76.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

#descargar los datos a la variable
wy = pd.read_csv("wage_mex.csv")
print(wy.head())

#  Year            y            w
#0  1970  51366.50616  17144.18671
#1  1971  52061.13198  16120.72129
#2  1972  55608.35466  18885.16814
#3  1973  58652.38231  17771.37159
#4  1974  60089.06334  19822.96331

# creo series individuales y las paso a logaritmo neperiano
import numpy as np
y=np.log(wy['y'])
w=np.log(wy['w'])

#Graficos de autocorrelacion
plt.rcParams.update({'figure.figsize':(8,6), 'figure.dpi':100})
plot_acf(w)
plot_acf(y)
plt.show()

#TEST DE RAICES UNITARIAS
###### Test de Raices unitarias Augmented Dickey Fuller

from statsmodels.tsa.stattools import adfuller
y_result = adfuller(y, regression='c')  #Modelo con constante
w_result = adfuller(w, regression='c')
print('The ADF Statistic of y: %f' % y_result[0])
print('The p value of y: %f' % y_result[1])
print('The ADF Statistic of w: %f' % w_result[0])
print('The p value of w: %f' % w_result[1])

#adf, pvalue, usedlag, nobs, critic values , icbest y restore,
#(-0.9121608163243172, 0.7839413109494487, 1, 33, {'1%': -3.6461350877925254, '5%': -2.954126991123355, '10%': -2.6159676124885216}, -68.53971151319041)

y_result = adfuller(y, regression='ct') # Modelo con constante y con tendencia
w_result = adfuller(w, regression='ct')
print('The ADF Statistic of y: %f' % y_result[0])
print('The p value of y: %f' % y_result[1])
print('The ADF Statistic of w: %f' % w_result[0])
print('The p value of w: %f' % w_result[1])

y_result = adfuller(y, regression='n') # Modelo con sin constante y sin tendencia
w_result = adfuller(w, regression='n')
print('The ADF Statistic of y: %f' % y_result[0])
print('The p value of y: %f' % y_result[1])
print('The ADF Statistic of w: %f' % w_result[0])
print('The p value of w: %f' % w_result[1])


# Tomar primera diferencia de y y w. Lo demas se puede repetir
dy =y.diff().iloc[1:]
dw =w.diff().iloc[1:]
dy_result = adfuller(dy, regression='c') 
dw_result = adfuller(dw, regression='c')
print('The ADF Statistic of dy: %f' % dy_result[0])
print('The p value of y: %f' % dy_result[1])
print('The ADF Statistic of dw: %f' % dw_result[0])
print('The p value of w: %f' % dw_result[1])


######## test estacionariedad KPSS ###########################
from statsmodels.tsa.stattools import kpss
y_resultk = kpss(y) 
w_resultk = kpss(w)
print('The KPSS Statistic of y: %f' % y_resultk[0])
print('The p value of y: %f' % y_resultk[1])
print('The KPSS Statistic of w: %f' % w_resultk[0])
print('The p value of w: %f' % w_resultk[1])


# COINTEGRACION

df=pd.concat([y, w], axis=1)

import matplotlib.pyplot as plt
plt.scatter(df.y, df.w)
plt.show()

# Test de cointegracion de Johansen
from statsmodels.tsa.vector_ar.vecm import coint_johansen
jres = coint_johansen(df,  det_order=0, k_ar_diff=2)  

from statsmodels.tsa.vector_ar.vecm import coint_johansen
jres = coint_johansen(df,  det_order=-1, k_ar_diff=2) #det_order es el modelo y k_ar_diff coincide con los rezagos aplicados
# det_order = -1 es 1) en Eviews no trends no intercepts
# det_order = 0 es 3) en eviews intercepts
# det_order = 1 es linear trend, no esta el equivalente en eviews

print(f'El estadistico de la traza es None: {jres.trace_stat[0]} y el valor critico al 95% es {jres.trace_stat_crit_vals[0,1]}')
print(f'El estadistico de la traza es At most one: {jres.trace_stat[1]} y el valor critico al 95% es {jres.trace_stat_crit_vals[1,1]}')
print(f'El maximo valor propio None: {jres.max_eig_stat[0]} y el valor critico al 95% es {jres.max_eig_stat_crit_vals[0,1]}')
print(f'El maximo valor propio At most one: {jres.max_eig_stat[1]} y el valor critico al 95% es {jres.max_eig_stat_crit_vals[1,1]}')

#alternativamente para que se imprima en formato tabla
from tabulate import tabulate
datos =[['H0', 'Estadistico Traza', 'Valor Critico'], ['None',jres.trace_stat[0],jres.trace_stat_crit_vals[0,1]], ['At Most one',jres.trace_stat[1],jres.trace_stat_crit_vals[1,1]]]
print(tabulate(datos))

#VECTOR DE COINTEGRACION VECM
from statsmodels.tsa.vector_ar.vecm import *
VECM_res=VECM(df,k_ar_diff=2,coint_rank=1,deterministic='ci')
VECM_fit=VECM_res.fit()
print(VECM_fit.summary())

# Det. terms outside the coint. relation & lagged endog. parameters for equation y
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# L1.y           0.1871      0.160      1.167      0.243      -0.127       0.501   print(VECM_fit.gamma[0])
# L1.w          -0.2062      0.099     -2.082      0.037      -0.400      -0.012   [ 0.18706973 -0.20616512 -0.03849102 -0.07598704]
# L2.y          -0.0385      0.155     -0.249      0.804      -0.342       0.265
# L2.w          -0.0760      0.106     -0.717      0.473      -0.284       0.132
# Det. terms outside the coint. relation & lagged endog. parameters for equation w
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# L1.y           0.3992      0.267      1.496      0.135      -0.124       0.922   print(VECM_fit.gamma[1])
# L1.w           0.2553      0.165      1.550      0.121      -0.068       0.578   [ 0.39919809  0.25526375 -0.23562264  0.32504608]
# L2.y          -0.2356      0.257     -0.915      0.360      -0.740       0.269
# L2.w           0.3250      0.176      1.844      0.065      -0.020       0.671
#                  Loading coefficients (alpha) for equation y                  
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]    # factor alfa de ajuste anual al equilibrio
# ------------------------------------------------------------------------------
# ec1           -0.1898      0.060     -3.183      0.001      -0.307      -0.073   print(VECM_fit.alpha[0])
#                  Loading coefficients (alpha) for equation w                     [-0.18982038]
# ==============================================================================  
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ec1            0.1477      0.099      1.488      0.137      -0.047       0.342   print(VECM_fit.alpha[1])
#           Cointegration relations for loading-coefficients-column 1              [0.14765709]
# ==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------    print(VECM_fit.beta)
# beta.1         1.0000          0          0      0.000       1.000       1.000    [[ 1.        ]
# beta.2        -0.8238      0.145     -5.689      0.000      -1.108      -0.540     [-0.82377623]]
# const         -3.0374      1.396     -2.176      0.030      -5.773      -0.302    print(VECM_fit.const_coint)
# ==============================================================================    [[-3.03742434]]

#PREDICCION
print(VECM_fit.predict(steps=3,alpha=0.95))

#(array([[10.8967323 ,  9.4310412 ],                                       #prediccion 3 años y y w con intervalo de confianza
#       [10.87545844,  9.43139173],
#       [10.85783678,  9.43898052]]), array([[10.89451499,  9.42735244],
#       [10.87236538,  9.42539448],
#       [10.85431859,  9.43090576]]), array([[10.89894961,  9.43472996],
#       [10.8785515 ,  9.43738898],
#       [10.86135497,  9.44705527]]))

#ESTIMACION DE MODELO VAR

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
model = VAR(df)
res_var = model.fit(2)  #2 rezagos  
res_var.summary()

#   Summary of Regression Results   
# ==================================
# Model:                         VAR
# Method:                        OLS
# Date:           Thu, 05, May, 2022
# Time:                     14:17:18
# --------------------------------------------------------------------
# No. of Equations:         2.00000    BIC:                   -11.1968
# Nobs:                     33.0000    HQIC:                  -11.4977
# Log likelihood:           108.579    FPE:                8.75796e-06
# AIC:                     -11.6503    Det(Omega_mle):     6.60486e-06
#--------------------------------------------------------------------
# Results for equation y
# ========================================================================
#            coefficient       std. error           t-stat            prob
# ------------------------------------------------------------------------
# const         1.012737         0.552916            1.832           0.067
# L1.y          1.042556         0.179313            5.814           0.000
# L1.w         -0.104401         0.100541           -1.038           0.299
# L2.y         -0.252579         0.165628           -1.525           0.127
# L2.w          0.238241         0.104348            2.283           0.022
# ========================================================================
#
# Results for equation w
# ========================================================================
# ...
#             y         w
# y    1.000000  0.263504
# w    0.263504  1.000000

#GRANGER CAUSALITY

from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(df[['y','w']], maxlag=[2])
grangercausalitytests(df[['w','y']], maxlag=[2])

# Granger Causality
# number of lags (no zero) 2
# ssr based F test:         F=5.3851  , p=0.0105  , df_denom=28, df_num=2   #Aqui se ve que el W causa al Y
# ssr based chi2 test:   chi2=12.6933 , p=0.0018  , df=2
# likelihood ratio test: chi2=10.7397 , p=0.0047  , df=2
# parameter F test:         F=5.3851  , p=0.0105  , df_denom=28, df_num=2

# Granger Causality
# number of lags (no zero) 2
# ssr based F test:         F=1.2298  , p=0.3077  , df_denom=28, df_num=2  #Aqui se ve que el Y no causa al Y (resultados iguales a EVIEWS)
# ssr based chi2 test:   chi2=2.8987  , p=0.2347  , df=2
# likelihood ratio test: chi2=2.7784  , p=0.2493  , df=2
# parameter F test:         F=1.2298  , p=0.3077  , df_denom=28, df_num=2
# {2: ({'ssr_ftest': (1.2297516280579526, 0.3076755663419266, 28.0, 2),
#    'ssr_chi2test': (2.898700266136603, 0.2347227771106069, 2),
#    'lrtest': (2.778386459853607, 0.24927633221403223, 2),
#    'params_ftest': (1.2297516280584648, 0.30767556634178195, 28.0, 2.0)},
#   [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x25a1a6899d0>,
#    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x25a1a689d60>,
#    array([[0., 0., 1., 0., 0.],
#           [0., 0., 0., 1., 0.]])])}

