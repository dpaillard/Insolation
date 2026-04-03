#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on      Thu Jun 17 15:20:13 2021
    
@author:    Didier Paillard (LSCE)
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import elliprd, elliprf, elliprj, ellipeinc, ellipkinc, ellipe
import inso.astro as astro
import scipy.integrate as integrate


#
#  solve the Kepler equation: x - e*sinx = v
#     e = eccentricity in [0,1[
#     v = mean anomaly (time from perihelion) in radian, in [0,2π]
#
def solveKepler(e,v):
    return optimize.newton(lambda x: x-e*np.sin(x)-v, v, fprime=lambda x: 1 - e*np.cos(x), fprime2=lambda x: e*np.sin(x))

#
#  computes the true anomaly (angular position from perihelion, in radian)
#     from the mean anomaly  (time from perihelion = 2πt/T, in radian)
#
def trueAnomalie(e,meanA):
    (n,vm) = np.divmod(meanA+np.pi, 2*np.pi)
    vm -= np.pi                                # vm in [-π, π[
    sg = np.sign(vm)
    cosE = np.cos(solveKepler(e,vm))
    v = sg * np.arccos((cosE-e)/(1-e*cosE))
    return v + meanA - vm

#
#  computes the mean anomaly  (time from perihelion = 2πt/T, in radian)
#     from the true anomaly (angular position from perihelion, in radian)
#
def meanAnomalie(e,trueA):
    (n,v) = np.divmod(trueA+np.pi, 2*np.pi)
    v -= np.pi                                # v in [-π, π[
    sqte = np.sqrt(1-e*e)
    E = 2*np.arctan(np.tan(v/2)*(1-e)/sqte)+trueA-v
    return E - e*sqte*np.sin(v)/(1+e*np.cos(v))

#
#  computes the true longitude from the mean longitude (=time from refL)
#      the default reference longitude refL is 0 (vernal point, used for perL)
#      the true anomaly of the reference point is: refA = refL-perL+np.pi = position of spring equinox (or other ref) versus perihelion
#      the mean anomaly of our point is meanL+meanAnomalie(e,refA) = time since reference point
#
def trueLongitude(meanL,e,perL,refL=0):
    return trueAnomalie(e,meanL+meanAnomalie(e,refL-perL+np.pi))+perL-np.pi


###################
#
#  length of season, in fraction of year in [0,1], function of true longitudes ->  *365.2425 to get it in days.
#
def length_of_season(lon1,lon2,e,per):
    d_ano = meanAnomalie(e,lon2-per+np.pi)- meanAnomalie(e,lon1-per+np.pi)
    return d_ano/2/np.pi


#
#  elliptic integral of the 2nd kind = integral from 0 to phi of sqrt(Max(0,1-m (sinx)^2))dx
#  Note:
#    the scipy implementation 'ellipeinc' does not work for m>1, therefore the use of Carlson
#    integrals 'elliprf' and 'elliprd'
#  Note:
#    the integrand sqrt( 1-m (sinx)^2 ) becomes complex (imaginary) when 1-m (sinx)^2 < 0
#    (this corresponds to 'polar night' cases) therefore the Max(0,...) cut-off
#    To have a nice continuous function for all values of 'phi' it is necessary to add periodically
#    the "complete elliptic" part ellipE(π/2,m):
#      n = (phi+π/2) div π
#      ellipE(phi,m) = 2 n ellipE(m) + ellipE(phi - nπ, m)
#  This corresponds to the Mathematica function Re[EllipticE[phi,m]]
#
def ellipE(phi,m):
    (n,xphi) = np.divmod(phi+np.pi/2, np.pi)
    xphi -= np.pi/2           #  phi == xphi + n*pi   &&   -pi/2 <= xphi < pi/2
    s = 1 - 2*(n%2)           #  s = (-1)^n
    sg = np.sign(xphi)
    
    def ellipE_(z,m):
        if z==0 and m==1:
            return 1
        else:
            return elliprf(z+m-1,z,z+m)-m*elliprd(z+m-1,z,z+m)/3
    
    if xphi==0:   # probably a test like 'abs(xphi)<verysmall' would be safer...
        e = 0
    else:
        c = np.square(1/np.sin(xphi))
        z = max(0,c-m)
        #e = sg*(elliprf(z+m-1,z,z+m)-m*elliprd(z+m-1,z,z+m)/3)
        e = sg * ellipE_(z,m)
    
    #e = sg*(elliprf(c-1,c-m,c)-m*elliprd(c-1,c-m,c)/3)
    #e = (sg/np.sqrt(c))*(elliprf(1-1/c,1-m/c,1)-(m/c)*elliprd(1-1/c,1-m/c,1)/3)
    #sinphi = np.sin(xphi)
    #sinphi2 = np.square(sinphi)
    #e = sg*sinphi*(elliprf(1-sinphi2,1-m*sinphi2,1)-m*sinphi2*elliprd(1-sinphi2,1-m*sinphi2,1)/3)
    #print('e = ',sg*sinphi*(elliprf(1-sinphi2,1-m*sinphi2,1)-m*sinphi2*elliprd(1-sinphi2,1-m*sinphi2,1)/3))
    if n==0:
        k = 0
    else:
        z = max(0,1-m)
        #k = (elliprf(z+m-1,z,z+m)-m*elliprd(z+m-1,z,z+m)/3)
        k = ellipE_(z,m)
    return 2*n*k + e

def ellipP(s2,phi,m):     # = ellipF(phi,m)-(1-s2)*ellipPi(s2,phi,m)
    (n,xphi) = np.divmod(phi+np.pi/2, np.pi)
    xphi -= np.pi/2           #  phi == xphi + n*pi   &&   -pi/2 <= xphi < pi/2
    s = 1 - 2*(n%2)           #  s = (-1)^n
    sg = np.sign(xphi)
    
    def ellipP_(z,m):
        if z==0 and m==1:
            sq = np.sqrt(s2)
            return sq*np.arctanh(sq)
        else:
            return s2*elliprf(z+m-1,z,z+m)-(1-s2)*s2*elliprj(z+m-1,z,z+m,z+m-s2)/3

    if xphi==0:   # probably a test like 'abs(xphi)<verysmall' would be safer...
        e = 0
    else:
        c = np.square(1/np.sin(xphi))
        z = max(0,c-m)
        #e = sg*(s2*elliprf(z+m-1,z,z+m)-(1-s2)*s2*elliprj(z+m-1,z,z+m,z+m-s2)/3)
        e = sg*ellipP_(z,m)
    if n==0:
        k = 0
    else:
        z = max(0,1-m)
        #k = s2*elliprf(z+m-1,z,z+m)-(1-s2)*s2*elliprj(z+m-1,z,z+m,z+m-s2)/3
        k = ellipP_(z,m)
    return 2*n*k + e


#  useful functions
#    p = sin_phi*sin_lon*sin_obl
#    s = np.maximum( 0, 1 - sin_phi*sin_phi - sin_lon*sin_lon*sin_obl*sin_obl )
#    ac = arccos( -p/sqrt(s+p*p) ) = hour (in radians) of sunrise/sunset
#
def inso_ac(a,b):
    s = np.maximum(0,1-a*a-b*b)
    p = a*b
    sp2 = s+p*p
    sq = np.where(sp2 == 0, 1, np.sqrt(s+p*p))   # 1 is not used...
    ac = np.where(sp2 == 0, np.pi/2, np.arccos( -p/sq ))
    return s, p, ac

#
#  dimensionless dayly inso at distance 'semi-major axis' of the Sun
#
def inso_g(a,b):
    s,p,ac = inso_ac( a, b )
    return (np.sqrt(s) + p*ac)/np.pi

def inso_mean_cosz(lon,phi,eps):   # = π g/ac
    sineps = np.sin(eps)
    sinphi = np.sin(phi)
    sinlon = np.sin(lon)
    s,p,ac = inso_ac( sineps*sinlon, sinphi )
    if s == 0 and p < 0:
        return 0
    return p + np.sqrt(s)/ac    # ac==0 iff s==0 && p < 0

#
#  h(a,b) = integral(g(a,x)dx) from 0 to b
#
def inso_h(a,b):
    s,p,ac = inso_ac( a,b )  #np.arccos( -a*b/np.sqrt(s+a*b*a*b) )
    #print("s,p,ac = ",s,' ',p,' ',ac)
    sb2 = s+b*b
    #print("acb = ",np.arccos( -b/np.sqrt(sb2) ))
    if sb2>0 :
        return (np.arccos( -b/np.sqrt(sb2) ) + b*np.sqrt(s) - a*(1-b*b)*ac)/np.pi/2 + (a-1)/4
    else:
        return b/4
#
#  G(phi,ob,x) = integral(ArcTan(tan(phi)/sqrt(1-m*Sin(x)^2))dx) from 0 to x
#
def inso_G(phi,ob,x):
    # for -π/2 <= x <= π/2
    def inso_G0(phi,ob,x):
        if abs(phi)==np.pi/2:
            return phi*x
        if phi==0:
            return 0
        if x<0:
            return -inso_G0(phi,ob,-x)
        # for 0 < x < π/2
        k = np.sin(ob)/np.cos(phi)
        k2=k*k
        t = np.tan(phi)
        if k*np.sin(x)>1:
            xm = np.arcsin(1/k)
        else:
            xm = x
        r = integrate.quad(lambda x: np.arctan(np.sqrt(1-k2*np.square(np.sin(x)))/t), 0, xm)
        return np.sign(phi)*x*np.pi/2 - r[0]
    
    (n,xx) = np.divmod(x+np.pi/2, np.pi)
    return inso_G0(phi,ob,xx-np.pi/2) + (2*n)*inso_G0(phi,ob,np.pi/2)

def inso_irrad(lon,phi,eps):
    sineps = np.sin(eps)
    sinphi = np.sin(phi)
    sinlon = np.sin(lon)
    s,p,ac = inso_ac( sineps*sinlon, sinphi )
    sp2 = s+p*p
    r1 = np.where(sp2>0, sinphi*sineps*(np.pi/2 - np.cos(lon)*ac), 0)
    cosphi = np.cos(phi)
    m = np.where(cosphi>0, np.square(sineps/cosphi), 0)
    r2 = np.where(cosphi>0, sinphi*sinphi*ellipP(sineps*sineps,lon,m)/cosphi + cosphi*ellipE(lon,m), 0)
    return r1+r2

def inso_irrad_lat(lon,phi,eps):
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    sineps = np.sin(eps)
    sinlon = np.sin(lon)
    s,p,ac = inso_ac( sineps*sinlon, sinphi )
    m = np.where(cosphi>0, np.square(sineps/cosphi), 0)
    #print("m = ",m)
    r1 = cosphi*ellipE(lon,m)
    #print("r1 = ",r1)
    r2 = sinphi*ellipP(sineps*sineps,lon,m)+cosphi*sineps*(np.pi/2 - np.cos(lon)*ac)
    #print("elleP = ",ellipP(sineps*sineps,lon,m))
    #print("r2 = ",r2)
    #r0 = cosphi*sineps*(np.pi/2 - np.cos(lon)*ac)
    #print("r0 = ",r0)
    g = inso_G(phi,eps,lon)
    #print("g = ",g)
    return 0.5*(g + sinphi*r1 - cosphi*r2 + (np.pi/2)*np.sin(eps)*(1-np.cos(lon)))

###################
#
#  length of day, in fraction of day, function of true longitude, latitude and obliquity (in radians)  ->  *24 to get it in hours
#
def inso_length_of_day(lon,phi,eps):
    sineps = np.sin(eps)
    sinphi = np.sin(phi)
    sinlon = np.sin(lon)
    s,p,ac = inso_ac( sineps*sinlon, sinphi )
    return ac/np.pi

###################
#
#   dimensionless (instantaneous) insolation - should be multiplied by solar constant
#   inputs
#       h: hour angle (radians) with noon = 0
#       lon: true longitude (radians)
#       phi: latitude (radians)
#       eps: obliquity (radians)
#       e: eccentricity
#       per: true longitude of perihelion (=climatic precession) (radians)
#
def inso_radians(h,lon,phi,eps,e,per):
    sindelta = np.sin(lon)*np.sin(eps)
    cosdelta = np.sqrt(1-sindelta*sindelta)
    g = np.sin(phi)*sindelta + np.cos(h)*np.cos(phi)*cosdelta
    if g<=0:
        return 0
    else:
        ar = (1-e*np.cos(lon-per))/(1-e*e)    # ratio a/r
        return ar*ar*g

###################
#
#   dimensionless (integrated over H) insolation - should be multiplied by solar constant and by duration of day (86400 s) to get J/m2
#
def inso_primitive(h,s,p,h0):
    ht = np.sign(h) * np.minimum(abs(h),h0)
    g = (p*ht + np.sin(ht)*np.sqrt( s+p*p ))/(2*np.pi)
    return g

def inso_primitive2(a,b,h):
    u = abs(h)
    aa = abs(a)
    a1 = 1 - a*a
    r1 = np.sqrt(a1)
    su = np.sign(u-np.pi/2)
    sinu = np.sin(u)
    cos2u = 1 - sinu*sinu
    dd = a1*cos2u + a*a
    d = np.where( dd>0, np.sqrt( cos2u/dd ), 0 )
    c = d*r1
    asc = r1*sinu*np.arcsin(c)
    i0 = np.where( u>np.pi/2, np.pi/2 * (1-aa), asc - aa*u + np.arccos(d) )
    
    sa = np.where( a>0, 1, -1 )   # is never equal to zero !
    bb = sa * b
    if bb<=su*c:
        s,p,ac = inso_ac( aa,bb )
        i1 = np.arccos( -bb/np.sqrt(s+b*b) ) + bb*np.sqrt(s) - aa*(1-b*b)*ac
    else:
        s = np.maximum(0,a1-b*b)
        t = a1 - s
        rt = np.sqrt(t)
        i1 = sa*np.sign(b)*sinu*(r1*(rt*np.sqrt(1-t)+np.arcsin(rt)) + aa*(b*b-t)) - aa*u*(1-b*b) - su*asc + np.arccos(-su*d)

    return np.sign(h)*sa*(i1-i0)/2

###################
#
#   dimensionless integrated insolation between h1 and h2 - should be multiplied by solar constant and by duration of day (86400 s) to get J/m2
#   inso_radians_integ(-np.pi,np.pi,lon,phi,eps,e,per) == inso_dayly_radians(lon,phi,eps,e,per)
#
def inso_radians_integ(h1,h2,lon,phi,eps,e,per):
    sindelta = np.sin(lon)*np.sin(eps)
    s,p,h0 = inso_ac( np.sin(phi), sindelta )
    g1 = inso_primitive(h1,s,p,h0)
    g2 = inso_primitive(h2,s,p,h0)
    ar = (1-e*np.cos(lon-per))/(1-e*e)    # ratio a/r
    return ar*ar*(g2-g1)

def inso_radians_integ2(h1,h2,lon,phi1,phi2,eps,e,per):
    sindelta = np.sin(lon)*np.sin(eps)
    sin1 = np.sin(phi1)
    sin2 = np.sin(phi2)
    g11 = inso_primitive2(sindelta,sin1,h1)
    g12 = inso_primitive2(sindelta,sin1,h2)
    g21 = inso_primitive2(sindelta,sin2,h1)
    g22 = inso_primitive2(sindelta,sin2,h2)
    ar = (1-e*np.cos(lon-per))/(1-e*e)    # ratio a/r
    return ar*ar*((g22-g21)-(g12-g11))

###################
#
#   dimensionless dayly insolation - should be multiplied by solar constant
#   inputs
#       lon: true longitude (radians)
#       phi: latitude (radians)
#       eps: obliquity (radians)
#       e: eccentricity
#       per: true longitude of perihelion (=climatic precession) (radians)
#
def inso_dayly_radians(lon,phi,eps,e,per):
    sineps = np.sin(eps)
    sinphi = np.sin(phi)
    sinlon = np.sin(lon)
    g = inso_g( sineps*sinlon, sinphi )
    ar = (1-e*np.cos(lon-per))/(1-e*e)    # ratio a/r
    return ar*ar*g

#
#   idem with
#       tr: time of the year (in radians) from reference refL (0 = spring equinox)
#
def inso_dayly_time_radians(tr,phi,eps,e,per,refL=0):
    return inso_dayly_radians(trueLongitude(tr,e,per,refL),phi,eps,e,per)

###################
#
#   integrated (dimensionless) insolation between 2 true longitudes - should be multiplied by solar constant
#
def inso_mean_radians(lon1,lon2,phi,eps,e,per):
    d_irrad = inso_irrad(lon2,phi,eps)-inso_irrad(lon1,phi,eps)
    d_ano = meanAnomalie(e,lon2-per+np.pi)- meanAnomalie(e,lon1-per+np.pi)
    pisq = np.pi*np.sqrt(1-e*e)
    return d_irrad/d_ano/pisq
#
#   idem between 2 mean longitudes (= time of the year)
#       tr: time of the year (in radians) from reference refL (0 = spring equinox)
#
def inso_mean_time_radians(tr1,tr2,phi,eps,e,per,refL=0):
    return inso_mean_radians(trueLongitude(tr1,e,per,refL),trueLongitude(tr2,e,per,refL),phi,eps,e,per)
#
#   ... the corresponding "Milankovitch" caloric seasons
#          = mean between 1/4 of a year before (-π/2) and after (+π/2) the soltice (summer or winter) (π/2 or 3π/2 vs vernal point)
#
def inso_caloric_summer_NH(phi,eps,e,per):
    return inso_mean_time_radians(-np.pi/2,np.pi/2,phi,eps,e,per,np.pi/2)

def inso_caloric_winter_NH(phi,eps,e,per):
    return inso_mean_time_radians(-np.pi/2,np.pi/2,phi,eps,e,per,3*np.pi/2)


###################
#
#   integrated (dimensionless) dayly insolation between 2 latitudes - should be multiplied by solar constant
#
def inso_mean_lat_radians(lon,phi1,phi2,eps,e,per):
    sineps = np.sin(eps)
    sinphi1 = np.sin(phi1)
    sinphi2 = np.sin(phi2)
    sinlon = np.sin(lon)
    g = (inso_h(sineps*sinlon,sinphi2) - inso_h(sineps*sinlon,sinphi1))/(sinphi2 - sinphi1)
    ar = (1-e*np.cos(lon-per))/(1-e*e)    # ratio a/r
    return ar*ar*g



###################
#
#   integrated (dimensionless) insolation between 2 true longitudes and 2 latitudes - should be multiplied by solar constant
#
def inso_mean_lon_lat_radians(lon1,lon2,phi1,phi2,eps,e,per):
    d_irrad = (inso_irrad_lat(lon2,phi2,eps)-inso_irrad_lat(lon1,phi2,eps))-(inso_irrad_lat(lon2,phi1,eps)-inso_irrad_lat(lon1,phi1,eps))
    d_ano = meanAnomalie(e,lon2-per+np.pi)- meanAnomalie(e,lon1-per+np.pi)
    pisq = np.pi*np.sqrt(1-e*e)*(np.sin(phi2)-np.sin(phi1))
    return max(0,d_irrad/d_ano/pisq)



#######################     END     #######################


#
#   some examples of use when running: $ python inso.py
#


if __name__ == '__main__':
    
    #astro.astrofiles_path = "/Users/didier/ Avec sauvegarde/Didier/Conceptual models/Inso & Models/insolation/astrofiles/"
    
    deg_to_rad = np.pi/180.
    
    #  define timescale : between 0 and 1000 kyrBP, included... (the time unit in astro.py is 1000 years, from past to future)
    t0,t1 = (-1000,0.)
    t=np.arange(t0,t1+1,1)                     # -1000 to 0 included !
    
    #  choose astronomical solution
    astro_params = astro.AstroLaskar2004()
    #astro_params = astro.AstroBerger1978()
    
    #   check if in range
    if (not(astro_params.in_range(t0)) or not(astro_params.in_range(t1))):
        print("resquested timescale is beyond the available range")
        exit()

    #   get the astronomical parameters on the requested time scale
    ecc = astro_params.eccentricity(t)
    pre = astro_params.precession_angle(t)     #in radians
    obl = astro_params.obliquity(t)            #in radians

#   print obliquity value at time t1 = 0 (today)
    print("obliquity = ",obl[-1]/deg_to_rad)
    
    #   print obliquity
    obliq = obl/deg_to_rad
    plt.plot(t,obliq)
    plt.title("obliquity over the past 1 Myr")
    plt.ylabel("obliquity (°)")
    plt.xlabel("time")
    plt.show()
    
    #  choose solar constant, latitude and true longitude (simplified calendar?)
    solar_constante = 1365
    latitude = 65*deg_to_rad
    trueLon = 90*deg_to_rad
    
    #   print dayly inso, summer solstice 65°N, at time t1 = 0 (today)
    print("dayly inso, summer solstice 65°N = ",solar_constante*inso_dayly_radians(trueLon,latitude,obl[-1],ecc[-1],pre[-1]))
    #inso65 = np.empty(len(t))
    #for i in range(len(t)):
    #    inso65[i] = solar_constante*inso_dayly_radians(trueLon,latitude,obl[i],ecc[i],pre[i])
    inso65 = solar_constante*inso_dayly_radians(trueLon,latitude,obl,ecc,pre)    # inso_dayly_radians() accepts 'np.array' as inputs
    plt.plot(t,inso65)
    plt.title("dayly inso summer solstice 65°N over the past 1 Myr")
    plt.ylabel("insolation (W/m2)")
    plt.xlabel("time")
    plt.show()

    print("dayly inso 65°N spring + 1/4 year = ",solar_constante*inso_dayly_radians(trueLongitude(np.pi/2,ecc[-1],pre[-1]),latitude,obl[-1],ecc[-1],pre[-1]))
    
    latitude2 = 75*deg_to_rad
    print("dayly inso, summer solstice 65-75°N = ",solar_constante*inso_mean_lat_radians(trueLon,latitude,latitude2,obl[-1],ecc[-1],pre[-1]))
    Minso65 = np.empty(len(t))
    for i in range(len(t)):
        Minso65[i] = solar_constante*inso_mean_lat_radians(trueLon,latitude,latitude2,obl[i],ecc[i],pre[i])
    #Minso65 = solar_constante*inso_mean_lat_radians(trueLon,latitude,latitude2,obl,ecc,pre)      # inso_mean_lat_radians() does not accept 'np.array' as inputs
    plt.plot(t,Minso65)
    plt.title("mean inso summer solstice 65-75°N over the past 1 Myr")
    plt.ylabel("insolation (W/m2)")
    plt.xlabel("time")
    plt.show()

    trueLon = 0
    trueLon2 = 180*deg_to_rad
    print("mean inso 65°N (0-180°)= ",solar_constante*inso_mean_radians(trueLon,trueLon2,latitude,obl[-1],ecc[-1],pre[-1]))
    mean_inso65 = np.empty(len(t))
    for i in range(len(t)):
        mean_inso65[i] = solar_constante*inso_mean_radians(trueLon,trueLon2,latitude,obl[i],ecc[i],pre[i])
    #mean_inso65 = solar_constante*inso_mean_radians(trueLon,trueLon2,latitude,obl,ecc,pre)      # ... does not accept 'np.array' as inputs
    plt.plot(t,mean_inso65)
    plt.title("mean inso summer (0-180°) 65°N over the past 1 Myr")
    plt.ylabel("insolation (W/m2)")
    plt.xlabel("time")
    plt.show()
    
    print("caloric summer inso 65°N = ",solar_constante*inso_caloric_summer_NH(latitude,obl[-1],ecc[-1],pre[-1]))
    caloricSummer = np.empty(len(t))
    for i in range(len(t)):
        caloricSummer[i] = solar_constante*inso_caloric_summer_NH(latitude,obl[i],ecc[i],pre[i])
    plt.plot(t,caloricSummer)
    plt.title("mean inso 'caloric summer' 65°N over the past 1 Myr")
    plt.ylabel("insolation (W/m2)")
    plt.xlabel("time")
    plt.show()




