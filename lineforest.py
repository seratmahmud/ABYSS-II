import sys
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from astropy.table import Table,Column
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.io import fits

import tensorflow as tf
tf.autograph.set_verbosity(0)

from specutils import Spectrum1D
from specutils.manipulation import SplineInterpolatedResampler

from PyAstronomy import pyasl
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Routine for estimating line properties in optical (BOSS or LAMOST) spectra',formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("tableIn",help="Input table containing paths to individual spectra, and their respective RVs")
parser.add_argument("--tableOut",help="Filename of the output table",default='predictions.fits')
parser.add_argument("--path",help="Column name containing paths",default='path')
parser.add_argument("--rv",help="Column name containing RV information for each spectrum (in km/s)",default='rv')
parser.add_argument("--instrument",help="Name of the spectrograph (BOSS or LAMOST currently supported, others require a different loader)",default='BOSS')

def unnormalize(predictions):
    predictions[:,0]=10**predictions[:,0]
    predictions[:,1]=10**predictions[:,1]
    a=np.where(predictions[:,2]<0)[0]
    predictions[a,0]=-predictions[a,0]
    return np.round(predictions.astype(float),4)
    
def readspec(path,instrument='BOSS'):

    if instrument=="BOSS":
        t=Table.read(path)
        flux = t['FLUX']
        wavelength=10**t['LOGLAM']
        err=1/np.sqrt(t['IVAR'])
    elif instrument=="LAMOST":
        hdul = fits.open(path)
        spec = hdul[1].data
        hdul.close()
        flux=spec[0][0]
        wavelength=spec[0][2]
        err=1/np.sqrt(spec[0][1])
    else:
        print('Instrument has to be either BOSS or LAMOST, or a new loader needs to be supplied')

    
    flux.flatten()
    wavelength.flatten()
    err.flatten()
    a=np.where(err>np.median(err)*5)[0]
    err[a]=np.median(err)*5
    a = np.where((flux<=0) | (np.isfinite(flux)==False))[0]
    flux[a] = 1

    return wavelength,flux,err

modelh = tf.keras.models.load_model('models/hlines.model',compile=False)
modelz = tf.keras.models.load_model('models/zlines.model',compile=False)

models=[modelh,modelz]

steps=128
reps=100


arr=[]
arr.append({'lines':6562.8		,'names':'Halpha','minmax':200,'model':0,'small':0})
arr.append({'lines':4861.3		,'names':'Hbeta','minmax':200,'model':0,'small':0})
arr.append({'lines':4340.5		,'names':'Hgamma','minmax':200,'model':0,'small':0})
arr.append({'lines':4101.7		,'names':'Hdelta','minmax':200,'model':0,'small':0})
arr.append({'lines':3970.1		,'names':'Hepsilon','minmax':200,'model':0,'small':0})
arr.append({'lines':3889.064	,'names':'H8','minmax':200,'model':0,'small':1})
arr.append({'lines':3835.391	,'names':'H9','minmax':200,'model':0,'small':1})
arr.append({'lines':3797.904	,'names':'H10','minmax':200,'model':0,'small':1})
arr.append({'lines':3770.637	,'names':'H11','minmax':200,'model':0,'small':1})
arr.append({'lines':3750.158	,'names':'H12','minmax':50,'model':1,'small':1})
arr.append({'lines':3734.369	,'names':'H13','minmax':50,'model':1,'small':1})
arr.append({'lines':3721.945	,'names':'H14','minmax':50,'model':1,'small':1})
arr.append({'lines':3711.977	,'names':'H15','minmax':50,'model':1,'small':1})
arr.append({'lines':3703.859	,'names':'H16','minmax':50,'model':1,'small':1})
arr.append({'lines':3697.157	,'names':'H17','minmax':50,'model':1,'small':1})
arr.append({'lines':10049.4889	,'names':'Pa7','minmax':200,'model':0,'small':0})
arr.append({'lines':9546.0808	,'names':'Pa8','minmax':200,'model':0,'small':0})
arr.append({'lines':9229.1200	,'names':'Pa9','minmax':200,'model':0,'small':0})
arr.append({'lines':9014.909	,'names':'Pa10','minmax':200,'model':0,'small':1})
arr.append({'lines':8862.782	,'names':'Pa11','minmax':200,'model':0,'small':1})
arr.append({'lines':8750.472	,'names':'Pa12','minmax':200,'model':0,'small':1})
arr.append({'lines':8665.019	,'names':'Pa13','minmax':200,'model':0,'small':1})
arr.append({'lines':8598.392	,'names':'Pa14','minmax':200,'model':0,'small':1})
arr.append({'lines':8545.383	,'names':'Pa15','minmax':200,'model':0,'small':1})
arr.append({'lines':8502.483	,'names':'Pa16','minmax':200,'model':0,'small':1})
arr.append({'lines':8467.254	,'names':'Pa17','minmax':200,'model':0,'small':1})
arr.append({'lines':8662.140	,'names':'CaII8662','minmax':50,'model':1,'small':0})
arr.append({'lines':8542.089	,'names':'CaII8542','minmax':50,'model':1,'small':0})
arr.append({'lines':8498.018	,'names':'CaII8498','minmax':50,'model':1,'small':0})
arr.append({'lines':3933.6614	,'names':'CaK3933','minmax':200,'model':0,'small':0})
arr.append({'lines':3968.4673	,'names':'CaH3968','minmax':200,'model':0,'small':0})
arr.append({'lines':6678.151	,'names':'HeI6678','minmax':50,'model':1,'small':0})
arr.append({'lines':5875.621	,'names':'HeI5875','minmax':50,'model':1,'small':0})
arr.append({'lines':5015.678	,'names':'HeI5015','minmax':50,'model':1,'small':0})
arr.append({'lines':4471.479	,'names':'HeI4471','minmax':50,'model':1,'small':0})
arr.append({'lines':4685.7		,'names':'HeII4685','minmax':50,'model':1,'small':0})
arr.append({'lines':6583.450	,'names':'NII6583','minmax':50,'model':1,'small':0})
arr.append({'lines':6548.050	,'names':'NII6548','minmax':50,'model':1,'small':0})
arr.append({'lines':6716.440	,'names':'SII6716','minmax':50,'model':1,'small':0})
arr.append({'lines':6730.816	,'names':'SII6730','minmax':50,'model':1,'small':0})
arr.append({'lines':5018.434	,'names':'FeII5018','minmax':50,'model':1,'small':1})
arr.append({'lines':5169.030	,'names':'FeII5169','minmax':50,'model':1,'small':1})
arr.append({'lines':5197.577	,'names':'FeII5197','minmax':50,'model':1,'small':1})
arr.append({'lines':6432.680	,'names':'FeII6432','minmax':50,'model':1,'small':1})
arr.append({'lines':5577.339	,'names':'OI5577','minmax':50,'model':1,'small':1})
arr.append({'lines':6300.304	,'names':'OI6300','minmax':50,'model':1,'small':1})
arr.append({'lines':6363.777	,'names':'OI6363','minmax':50,'model':1,'small':1})
arr.append({'lines':3727.42		,'names':'OII3727','minmax':50,'model':1,'small':1})
arr.append({'lines':4958.911	,'names':'OIII4959','minmax':50,'model':1,'small':1})
arr.append({'lines':5006.843	,'names':'OIII5006','minmax':50,'model':1,'small':1})
arr.append({'lines':4363.85		,'names':'OIII4363','minmax':50,'model':1,'small':1})
arr.append({'lines':6707.760	,'names':'LiI','minmax':50,'model':1,'small':0})

z=Table(arr)
z['lines']= pyasl.airtovac2(z['lines'])

spline = SplineInterpolatedResampler()


lin = []
for i in range(len(z)):
    lin.extend(np.linspace(-z['minmax'][i],z['minmax'][i],steps) + z['lines'][i])


def main(args):
    specall = Table.read(args.tableIn)
                          
    l=len(specall)
    for i in range(len(z)):
        specall[f"{z['names'][i]}_eqw"] = np.nan
        specall[f"{z['names'][i]}_abs"] = np.nan
        specall[f"{z['names'][i]}_detection"] = Column(dtype=float,shape=(2,),length=l)
        specall[f"{z['names'][i]}_eqw_std"] = Column(dtype=float,shape=(3,),length=l)*np.nan
        specall[f"{z['names'][i]}_abs_std"] = Column(dtype=float,shape=(3,),length=l)*np.nan

    for i in tqdm(range(len(specall))):
        loc=specall[args.path][i]
        try:
            wavelength,flux,err=readspec(loc,instrument=args.instrument)
            
            specs = Spectrum1D(spectral_axis=np.sort(np.array(wavelength).flatten()*(1 - specall[args.rv][i]/3e5))*u.AA,
                              flux=(np.array(np.log10(flux)).flatten()*u.Jy),
                              uncertainty=StdDevUncertainty(err/flux/np.log(10)))
            for j in range(len(z)):
                try:
                    spec = spline(specs, (np.linspace(-z['minmax'][j],z['minmax'][j],steps) + z['lines'][j])*u.AA)
                    window = spec.flux.value.reshape((1,(steps) , 1))
                    window = np.tile(window,(reps+1,1,1))
                    scatter= spec.uncertainty.array.reshape((1,(steps) , 1))
                    scatter= np.tile(scatter,(reps+1,1,1))*np.random.normal(size=(reps+1,steps,1),loc=0,scale=1)
                    scatter[0]=0
                    window=window+scatter
                    
                            
                    predictions = unnormalize(np.array(models[z["model"][j]](window[0:1,:,:])))
                    if np.abs(predictions[0,2])>0.5: 
                        specall[f"{z['names'][j]}_eqw"][i] = predictions[0,0]
                        specall[f"{z['names'][j]}_abs"][i] = predictions[0,1]
                        specall[f"{z['names'][j]}_detection"][i][0] = predictions[0,2]
                        predictions = unnormalize(np.array(models[z["model"][j]](window[1:,:,:])))  
                        a=np.where(np.abs(predictions[1:,2])>0.5)[0]
                        specall[f"{z['names'][j]}_detection"][i][1] = np.round(len(a)/reps,2)
                        if len(a)>2:
                            specall[f"{z['names'][j]}_eqw_std"][i] = np.round(np.percentile(predictions[1:,0][a],[16,50,84]),4)
                            specall[f"{z['names'][j]}_abs_std"][i] = np.round(np.percentile(predictions[1:,1][a],[16,50,84]),4)
                except:
                    stop
        except:
            stop
    for j in range(len(z)):
        a=np.where(specall[f"{z['names'][j]}_detection"][:,1]<0.3)[0]
        specall[f"{z['names'][j]}_eqw"][a]=np.nan
        specall[f"{z['names'][j]}_abs"][a]=np.nan
        specall[f"{z['names'][j]}_eqw_std"][a]=specall[f"{z['names'][j]}_eqw_std"][a]+np.nan
        specall[f"{z['names'][j]}_abs_std"][a]=specall[f"{z['names'][j]}_abs_std"][a]+np.nan

    specall.write(args.tableOut, format='fits', overwrite = True)
    print(specall)

if __name__ == "__main__":
    args=parser.parse_args()
    main(args)
