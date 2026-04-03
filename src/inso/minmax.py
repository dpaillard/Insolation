#######################################
#
#   minima and maxima of daily insolation
#
#######################################

import numpy as np
import matplotlib.pyplot as plt
import inso.astro as astro
import inso as inso
from scipy import optimize
from operator import itemgetter
import random


deg_to_rad = np.pi/180.



def mod_2pi_minus_pi_over_2(x):
    return ((x+np.pi/2)%(2*np.pi))-np.pi/2   # x is in [-π/2,3*π/2[

def my_plot(f,a,b,n=100,label='',color=""):
    x = np.linspace(a,b,n)
    y = np.empty(len(x))
    for i in range(len(x)):
        y[i] = f(x[i])
    if color=='':
        plt.plot(x,y,label=label)
    else:
        plt.plot(x,y,label=label,color=color)


#####################################################
#
#   minmaxPhi: latitude of minmax as a fonction of true longitude
#   NB: the interesting function is the inverse: for a given latitude, what is the (true) longitude.
#           .... which is a lot more tricky
#
def one_minus_hcoth(h):   #  0 <= h < π
    if h<=0:
        return 0
    if h==np.pi/2:
        return 1
    return 1-h/np.tan(h)

def inv_one_minus_hcoth(y):   #  0 <= y
    if y<=0:
        return 0
    if y>1e6:
        return np.pi
    a = 2*np.arctan(np.square(np.square(y))/15)/np.pi
    guess = np.sqrt(3*y)*(1-a)+np.pi*(1-1/y)*a
    return abs(optimize.newton(lambda x: one_minus_hcoth(x)-y, guess, fprime=lambda x: (x-np.sin(x)*np.cos(x))/np.square(np.sin(x))))

def minmaxPhi(lon,eps,e,per):
    sineps = np.sin(eps)
    sinlon = np.sin(lon)
    coslon = np.cos(lon)
    if abs(sinlon)<1.0e-12:   #  np.sin(np.pi)==1.2246467991473532e-16
        return np.arctan(4*e*np.sin(per)/sineps/np.pi/(1-e*coslon*np.cos(per)))
    sinDelta = sineps*sinlon
    cos2Delta = 1 - np.square(sinDelta)
    tanDelta = sinDelta/np.sqrt(cos2Delta)
#if coslon == 0:   #pas nécessaire ??
#   return np.arctan(-1/tanDelta)
    z = max(0,1+2*e*np.tan(lon)*np.sin(lon-per)/(1-e*np.cos(lon-per)))
#print("z = ",z)
    if (z==0):
        return np.arctan(1./tanDelta)
    else:
        gg = 1/cos2Delta/z
        #print("gg = ",gg)
        #print("tanDelta = ",tanDelta)
        #print("h = ",inv_one_minus_hcoth(max(0,gg)))
        return np.arctan(-np.cos(inv_one_minus_hcoth(max(0,gg)))/tanDelta)

def DminmaxPhi_Dlon(lon,eps,e,per):   # derivative of minmaxPhi versus lon, to find lon_min, lon_max
    c = np.cos(eps)
    sin2lon = np.square(np.sin(lon))
    coslon = np.cos(lon)
    coslonper = np.cos(lon-per)
    tanlon = np.tan(lon)
    cot2lon = np.square(1/tanlon)
    c2 = c*c
    d = (c2+cot2lon)*(1 + 2*e*np.sin(lon-per)*tanlon/(1-e*coslonper))
    r = c2 + cot2lon*np.square(np.tan(np.pi/2 - inv_one_minus_hcoth( 1/sin2lon/d )))
    r = r + 2*coslon*sin2lon*(c2+cot2lon)*(1+6*e*e-4*e*coslonper-3*e*e*coslonper*coslonper) \
        /(1-e*coslonper)/(-2*coslon+3*e*np.cos(2*lon-per)-e*np.cos(per))
    return r

#####################################################
#
#   longitude boundaries for minmaxPhi()
#
def xc_yc(e):    #   0 <= e < 1   -   yc = omega(xc)  &   π-yc = omega(-xc)
    if e<1/3:
        return (0,-np.pi/2,np.pi/2,np.pi/2,3*np.pi/2)
    e2 = e*e
    sq = np.sqrt(7+18*e2)
    xc = np.arccos((-2+sq)/(3*e))
    yc = np.pi/2 - xc + np.arctan(2*np.sqrt(-11-9*e2+4*sq)/(5-sq))
    return (xc,yc-np.pi,np.pi-yc,yc,2*np.pi-yc)
'''
    def omega(e,x):
    return np.arctan(2*e*np.sin(x)/(1-e*np.cos(x)))-x+np.pi/2
    def inv_omega(e,y):
    (xc,yc,ycb)=xc_yc(e)
    x0 = ((y+np.pi/2)*xc+np.pi*(yc-y))/(yc+np.pi/2)
    if e<1/3 and y>3*e*np.pi/2:
    x1 = (1-e)*(np.pi-2*y)/(2-6*e)
    else:
    x1 = (np.pi*(1+5*e)-2*y*(1+e))/(2+6*e)
    z = optimize.brentq(lambda x: y-omega(e,x), x0, x1)   # the root is between x0 and x1
    return z
    '''
#def lon_boundaries(e,per):
#    (xc,omega2,omega1)=xc_yc(e)

def inv_fx(e,xx):
    x = ((xx+np.pi)%(2*np.pi))-np.pi    # x is in [-π,π[
    sx = np.sin(x)
    cx = np.cos(x)
    poly_coeff = [sx*(1+e),2*cx*(1+3*e),-10*e*sx,2*cx*(1-3*e),-sx*(1-e)]
    r4 = np.round(np.roots(poly_coeff),12)      # cut off possible small imaginary values
    real_r4 = np.real(r4[np.isreal(r4)])        # real roots
    y4 = 2*np.arctan(real_r4)
    #print("y4 = ",y4)
    #print("y4 = ",(y4<x+np.pi/2) & (y4>x-np.pi/2))
    #return y4[(y4<x+np.pi/2) & (y4>x-np.pi/2)]
    return y4

def lambda_min(e,om):
    #omega = ((om+np.pi/2)%(2*np.pi))-np.pi/2   # x is in [-π/2,3*π/2[
    omega = mod_2pi_minus_pi_over_2(om)
    yc = xc_yc(e)[1:]
    two_roots = (om>=yc[0] and om<=yc[1]) or (om>=yc[2] and om<=yc[3])
    if two_roots:
        if omega<=np.pi/2:
            return (-np.pi/2,np.pi/2)
        else:
            return (np.pi/2,3*np.pi/2)
    else:
        lbd = np.sort(mod_2pi_minus_pi_over_2(om+inv_fx(e,np.pi/2-om)))
        if omega<=np.pi/2:
            return (-np.pi/2,lbd[0],lbd[1],np.pi/2)
        else:
            return (np.pi/2,lbd[2],lbd[3],3*np.pi/2)

def lambda_max(e,om):
    omega = mod_2pi_minus_pi_over_2(om)
    lbd = np.sort(mod_2pi_minus_pi_over_2(om+inv_fx(e,np.pi/2-om)))
    if omega<=np.pi/2:
        return tuple(lbd[-2:])   # the last 2 values
    else:
        return tuple(lbd[:2])    # the first 2 values

def one_minus_h_coth(h):
    if h > 0.003:
        return 1-h*np.tan(np.pi/2 - h)  #   x = 1 - h cot(h)
    else:
        h2 = h*h
        return (1+h2/15)*h2/3           # approximation around h == 0

def big_equa(c,h,u):   # equation to be solve in u, or in h.  0 <= c & u <=1 et 0 <= h <= π
    c2 = c*c
    u2 = u*u
    h2 = h*h
    x = one_minus_h_coth(h)
    y = x*(1-x)*(1-x)
    r = (2+3*u)*c2 + 5*(1-c2)*u2
    s = 3*u2*(2+3*u) - 2*c2*(2-13*u+2*u2+9*u*u2) + c2*c2*(16+9*u)*(1-u)*(1-u)
    c2u = (c2*(1-u)+u)*x
    return 4*y*y*u - 6*h2*y*((r-2*u)*x-3*u) - 3*h2*h2*(c2u*c2u*c2u*(u-6) + s*x*x - 3*r*x + 5*u)

def solHofU(c,u):      # 0 <= c & u <=1
    if u>.9999:
        return np.pi/2 + 2*(1-u)*(3-5*c*c)/3/np.pi   # use approx near u==1 to avoid problems
    #fa = big_equa(c,u*np.pi/2,u)
    #fb = big_equa(c,2.,u)
    #if (fa*fb)>0:  # cannot solve with brentq
    #    print ("c = ",c)
    #    print ("u = ",u)
    #   print ("big_equa(c,u*np.pi/2,u) = ",fa)
    #    print ("big_equa(c,2.,u) = ",fb)
    #   if fa<1e-20:
    #       return u*np.pi/2
    #   if fb<1e-20:
    #       return 2.
    return optimize.brentq(lambda x: big_equa(c,x,u), u*np.pi/2, 2.)   # the root is between x0 and x1

def h0(c):
    c2 = c*c
    def h0equa(h):
        coth = np.tan(np.pi/2 - h)
        z = 1 - h*coth
        return 3 + 2*(1-4*c2-coth*coth)*z + 3*c2*c2*z*z
    return optimize.brentq( h0equa, 2., np.pi )

def solUofH(c,h):      # 0 <= c <=1  , h0(c) <= h <= π
    #print("big_equa(c,h,0) = ",big_equa(c,h,0))
    #print("big_equa(c,h,1) = ",big_equa(c,h,1))
    return optimize.brentq(lambda x: big_equa(c,h,x), -.01, 0.2)   # the root is between x0 and x1

def e2LH(c,u,h):       #  the critical points are given by e^2 == e2LH(c,u,h)
    c2 = c*c
    x = one_minus_h_coth(h)
    c2u = c2*(1-u)
    cot2 = np.square(np.tan(np.pi/2 - h))  # pb pour h == 0 !!
    d = np.square(c2u + u)*x
    a = np.square(1-c2u*x-u*x)*u/d/x
    b = (c2u + u*cot2)/d
    return (9*a*a+4*np.square(b+u-1)-4*a*(3*b-4+4*u))/np.square(2*b+6*u-3*a-6)

def e2H(c,h):
    return e2LH(c,solUofH(c,h),h)

def e2L(c,u):
    c2 = c*c
    if u==1:
        return np.square((1-c2)/(3-c2))
    if u==0:
        return 1
    return e2LH(c,u,solHofU(c,u))

def eL_true_min(c):
    c2 = c*c
    f0 = np.square((1-c2)/(3-c2))
    x0 = np.sqrt(np.sqrt(3/5)-c)
    bra = (min(0.65,x0),min(.7,1.2*x0))
    opt = optimize.minimize_scalar(lambda x: (e2L(c,np.square(np.cos(x)))-f0)*10e5, bracket=bra)
    return opt.x

def eLmin(c):   # approximation (+/-2*10^-10) of the minimum of sqrt(e2L(c,u)), 0<=c<=sqrt(3/5)
    e0 = 0.2972715143047017
    sqr5_3 = np.sqrt(5/3)
    cb = 1 - c*sqr5_3
    n = 0.16861013326926652 + c*(-1.3096008005125173 + c*(4.817837283049987 \
      + c*(-9.860117891921941 + c*(11.375859106972062 + c*(-4.990764732854919 \
      + c*(-4.126936931195104 + c*(8.961745019854606 + c*(-8.308489466870162 \
      + c*(4.165889277592115 + c*(-0.9121661026396392))))))))))
    d = 1 + c*(-7.98165538827119 + c*(30.009787537630253 + c*(-63.81175063434554 \
      + c*(80.23931530613446 + c*(-54.093167692450876+ c*(14.592656991969651))))))
    return (e0*cb + c*sqr5_3/6) + cb*c*n/d

def eLminArg(c):   # approximation of the arg-minimum of e2L(c,cos(lbd)^2), 0<=c<=sqrt(3/5)
    n = 0.789186069228168 + c*(-5.6037907960691005 + c*(20.194250325013805 + c*(-42.51995032299711 \
      + c*(55.21404101446179 + c*(-36.226742757251486 + c*(0.04963687739253456 + c*(10.108994799476692 \
      + c*(-0.913800554069458))))))))
    d = 1 +c*(-7.746082571165008 + c*(29.81435468902234 + c*(-67.78771752967735 \
      + c*(96.94609981976748 + c*(-80.58273442265427 + c*(29.30303273600362))))))
    return np.sqrt(np.sqrt(3/5)-c)*n/d

def lbd_critL(e,c):
    c2 = c*c
    def omegaL(lbd):
        w = np.tan(np.pi/2-lbd)
        w2 = w*w
        h = solHofU(c,np.square(np.cos(lbd)))
        x = one_minus_h_coth(h)
        z = (c2+w2)/(1+w2)
        d = 6 - 2*(c2+w2*np.square(np.tan(np.pi/2 - h)))/(z*z*x) + 3*w2*np.square((1-z*x)/(z*x))
        return lbd + np.arccos((1-4/d)/e)*np.sign(d)*np.sign(x*z-1)
    if (e > (1-c2)/(3-c2)):                         # L1 - 1 critical value
        lbd = optimize.brentq(lambda x: e*e-e2L(c,np.square(np.cos(x))), 0, np.pi/2)
        return ((lbd,omegaL(lbd)),)
    elif ((c < np.sqrt(3/5)) and (e>eLmin(c))):     # L2 - 2 critical values
        lbd1 = optimize.brentq(lambda x: e*e-e2L(c,np.square(np.cos(x))), 0, eLminArg(c))
        lbd2 = optimize.brentq(lambda x: e*e-e2L(c,np.square(np.cos(x))), eLminArg(c), np.pi/2 )
        return ( (lbd1,omegaL(lbd1)), (lbd2,omegaL(lbd2)), )
    else:           # L0 - no critical value
        return ()

def cZero(e):    # for 1/6 < e <= 0.65963 :
    # if c < cZero(e), omegaL>0 :interval with 2x2 solutions lbd_minmaxL ; else interval with no sol for lbd_minmaxL
    cmin = max(0.64,np.sqrt(max(0,(1-3*e)/(1-e))))
    guess = (1.4409964474638386 +e*(-6.845708912614622+e*(20.1724756790539+e*(-26.518385158308988 \
      +e*(14.85684832957946)))))/(1+e*(-0.21904747609608766+e*(0.11200219017022854)))
    return optimize.brentq(lambda x: lbd_critL(e,x)[0][1], max(cmin,guess-.001), min(1.,guess+.002))

def lbd_minmaxL(e,eps,per,critPoints):    #
    def root(x1,x2):
        return optimize.brentq(lambda x: DminmaxPhi_Dlon(x,eps,e,per), x1, x2)
    n = len(critPoints)
    if n==0:                                      # L0 - no critical value => 1 root
        return (root(1e-5,np.pi/2 - 1e-5),)
    if n==1:                                      # L1 - 1 critical value => 0 or 2 roots
        lbdC = critPoints[0][0]
        perC = critPoints[0][1]
        if per>perC:
            return ()
        else:
            xa = lbdC*(per+np.pi/2)/(perC+np.pi/2)
            xb = max(per,-np.pi/2)
            return (root(xa + 1e-5,np.pi/2 - 1e-5), root(xb + 1e-5, xa - 1e-5),)
    else:  # n==2                                 # L2 - 2 critical values => 1 or 3 roots
        lbdC1 = critPoints[0][0]
        perC1 = critPoints[0][1]
        lbdC2 = critPoints[1][0]
        perC2 = critPoints[1][1]
        if per<perC1:
            xa = lbdC1*(per+np.pi/2)/(perC1+np.pi/2)
            return (root(xa + 1e-5,np.pi/2 - 1e-5),)
        elif per>perC2:
            xa = min(lbdC1*per/perC1, np.pi/2)
            return (root(1e-5, xa - 1e-5),)
        else:
            xa = min(lbdC1*per/perC1, np.pi/2)
            xb = lbdC2*(per+np.pi/2)/(perC2+np.pi/2)
            return (root(xb + 1e-5, np.pi/2 - 1e-5),root(xa + 1e-5, xb + 1e-5),root(1e-5, xa - 1e-5),)

def subplot_lbd_minmaxL(subplt,e,eps):
    critPoints = lbd_critL(e,np.cos(eps))
    pp = np.pi/2
    pi_over2_eps = pp - 1e-5     # ... avoid the exact boundary value...
    n = len(critPoints)
    subplt.axis([-pp, pp, -pp, pp])
    xyc = xc_yc(e)
    if (xyc[0]!=0):
        om0 = np.linspace(-pp,xyc[1]-1e-5,100)
        lbd0 = list(map(lambda z: lambda_min(e,z)[1],om0))
        lbd1 = list(map(lambda z: lambda_min(e,z)[2],om0))
        subplt.fill_between(om0,lbd0,lbd1,color='pink')
        om0 = np.linspace(xyc[2]+1e-5,pp,100)
        lbd0 = list(map(lambda z: lambda_min(e,z)[1],om0))
        lbd1 = list(map(lambda z: lambda_min(e,z)[2],om0))
        subplt.fill_between(om0,lbd0,lbd1,color='pink')

    def do_plot(f,a,b,n=100,label='',color=""):
        x = np.linspace(a,b,n)
        y = np.empty(len(x))
        for i in range(len(x)):
            y[i] = f(x[i])
        subplt.plot(x,y,label=label,color=color)

    if n==0:
        domain_name = "L0"
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[0],   -pi_over2_eps, pi_over2_eps,color='b') # 1 root
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[0], -pi_over2_eps, pi_over2_eps,color='b')
    elif n==1:
        lbdC = critPoints[0][0]
        perC = critPoints[0][1] - 1e-5
        if perC<0:
            domain_name = "L1b"
        else:
            domain_name = "L1a"
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[0],  -pi_over2_eps, perC,color='b') # 2 roots
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[1],  -pi_over2_eps, perC,color='b')
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[0], -perC, pi_over2_eps,color='b')
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[1], -perC, pi_over2_eps,color='b')
        subplt.plot([perC], [lbdC], 'o',color='k')
        subplt.plot([-perC], [-lbdC], 'o',color='k')
    else:
        domain_name = "L2"
        lbdC1 = critPoints[0][0]
        lbdC2 = critPoints[1][0]
        perC1 = critPoints[0][1]
        perC2 = critPoints[1][1]
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[0], -pi_over2_eps, perC1 - 1e-5,color='b') # 1 root
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[0], perC1 + 1e-5, perC2 - 1e-5,color='b')  # 3 roots
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[1], perC1 + 1e-5, perC2 - 1e-5,color='b')
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[2], perC1 + 1e-5, perC2 - 1e-5,color='b')
        do_plot(lambda z: lbd_minmaxL(e,eps,z,critPoints)[0], perC2 + 1e-5, pi_over2_eps,color='b')  # 1 root
        
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[0], -perC1 + 1e-5, pi_over2_eps,color='b')
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[0], -perC2 + 1e-5, -perC1 - 1e-5,color='b')
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[1], -perC2 + 1e-5, -perC1 - 1e-5,color='b')
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[2], -perC2 + 1e-5, -perC1 - 1e-5,color='b')
        do_plot(lambda z: -lbd_minmaxL(e,eps,-z,critPoints)[0], -pi_over2_eps, -perC2 - 1e-5,color='b')
        subplt.plot([perC1], [lbdC1], 'o',color='k')
        subplt.plot([perC2], [lbdC2], 'o',color='k')
        subplt.plot([-perC1], [-lbdC1], 'o',color='k')
        subplt.plot([-perC2], [-lbdC2], 'o',color='k')
    subplt.set_title(f"e = {e}  c = cos($\\epsilon$) = {np.cos(eps):.3f} :{domain_name}")
    subplt.set_xlabel(r'$\varpi$')
    subplt.set_ylabel(r'$\lambda$')

def plot_lbd_minmaxL(e,eps):
    subplot_lbd_minmaxL(plt.gca(),e,eps)
    plt.show()

def eH0(c):     # with c between c0s and 1
    x = one_minus_h_coth(h0(c))
    c2 = c*c
    return abs( (1-c2*x)/(1-3*c2*x) )

c0s = 0.3891493340546299    # solution of eH0(c)==1, ie. minimal value of c for the eH0(c) curve (otherwise e>1)
#     ie. solution of sqrt(1-c*c/4)==cotangent((2*c*c-1)/(c*c*sqrt(4-c*c)))
c0r = 0.59743621859         # solution of eH0(c)==0
#     ie. solution of 0==sqrt(1-c*c)+cotangent(sqrt(1-c*c)/(c*c))

def eHmin1(x):    #approximation better than 10^-6  # with c between 0 and 1
    n = x*(0.00003502357487424751 + x*(-0.001998108717320389 + x*(0.5595586251037636 \
      + x*(-8.867139747404073 + x*(64.91655661085856 + x*(-284.9994687909545 \
      + x*(828.7870921994659 + x*(-1666.8808698287296 + x*(2347.4083124452163 \
      + x*(-2285.168880445924 + x*(1471.554675962431 + x*(-566.1025914636236 \
      + x*(98.84842259724307)))))))))))))
    d = 1 + x*(-15.987097634046595 + x*(114.09035631818206 + x*(-472.1107950149408 \
      + x*(1211.434256763127 + x*(-1795.3809291951673 + x*(623.1020675195199 \
      + x*(3718.1436290211523 + x*(-9922.043171720201 + x*(13798.555052409418 \
      + x*(-12323.65990312766 + x*(7155.077062705843 + x*(-2489.993586433917 \
      + x*(397.9398245639788)))))))))))))
    return n/d

def arg1_approx(x):    # argmin or argmax of e2H(x,lbd)... smoother (better?) than the minimization algorithm
    if x < 0.5:
        return (3.141592653589793+x*(- 24.373254784671218+x*(80.81702613906958+x*(-159.15719967933768 \
          + x*(211.75145226192728+x*(-170.15071654562783+x*(36.46382731528965+x*(38.714949995234775 \
          + x*(-13.812872795884752))))))))) / (1 + x*(-7.758239143409276+x*(25.974433934965603 \
          + x*(-52.592073349529784+x*(73.79150502347504+x*(-66.50888269464632+x*(27.26283161277805)))))))
    else:
        return (2.195810546585893+x*(-21.16114968341698+x*(79.03128612670795+x*(-128.10737383279658 \
          + x*(9.944967430332687+x*(320.21797704223087+x*(-605.8894291037616+x*(596.6680278623417 \
          + x*(-363.41366503129973+x*(132.0509930847394+x*(-21.274500336914116))))))))))) / \
          (1+x*(-10.24688253869529+x*(43.24523501633+x*(-96.13937176438999+x*(118.71300111182673 \
          + x*(-77.27638366786168+x*(20.79235342508534)))))))

def arg2_approx(x):
    return (2.6565353539155434+x*(-8.150153700090657+x*(-23.075732420524407+x*(178.03120939252912 \
      + x*(-451.0315970645998+x*(647.1709850751215+x*(-594.6884128797976+x*(358.54889266513277 \
      + x*(-131.38609163587483+x*(21.97814981462397)))))))))) \
        /(1+x*(-3.928824383747655+x*(-0.09370119176837198+x*(28.18601059846761+x*(-66.93003737136101 \
      + x*(70.43535007265795+x*(-35.63875465010337+x*(6.987682309682637))))))))

def argMax_approx(x):
    return min(arg1_approx(x),arg2_approx(x))
def argMin1_approx(x):
    if x<c0max:
        return arg1_approx(x)
    else:
        return max(arg1_approx(x),arg2_approx(x))

carg12_approx = 0.5275312608408457   # solution of arg1_approx(x)==arg2_approx(x)

def eHmin1_approx(x):
    if x<0.05:
        #return 0     # very small and difficult to compute...
        return eHmin1(x)   # seems better...
    #elif x<c0max:
    #    return np.sqrt(e2H(x,arg1_approx(x)))
    else:
        return np.sqrt(e2H(x,argMin1_approx(x)))

def eHmax_approx(x):
    return np.sqrt(e2H(x,argMax_approx(x)))

c0max = 0.3387915804    # solution of eHmax(c)==1
cm = 0.527521           # solution of eHmax(c)==eHmin1(c) (in fact, "exchange" of the two curves arg1/arg2)

def eHmax(x):    #approximation better than 10^-6  # with c between c0max and 1
    if x < cm:
        return (-1.29166255753111+x*(15.41939398843474+x*(-76.99702534204498+x*(205.8877197288026 \
          +x*(-310.9718576286165+x*(251.5886376631189+x*(-85.19381093095183))))))) / \
          (1+x*(-9.49375442268299+x*(12.868159786206178+x*(189.4010879998579+x*(-1133.861762443162 \
          +x*(2959.9732114077137+x*(-4200.47431200051+x*(3172.2391449636502+x*(-1005.7947728248703)))))))))
    else:
        return (1.728837196218946+x*(-15.111614443959972+x*(53.214349189842125+x*(-96.33321621761493 \
          +x*(94.39666140706993+x*(-47.4013052456214+x*(9.370967869283248))))))) / \
          (1+x*(-12.79655961428312+x*(57.38163983539084+x*(-122.14468585402392+x*(133.31262238469455 \
          +x*(-71.62678752726478+x*(14.453874283847266)))))))

c0min2 = 0.3298973524027073    # solution of eHmin2(c)==1

def argMin2_approx(x):
    return (2.613313446817506+x*(-39.40226635183805+x*(259.1006718548223+x*(-966.9154075111475 \
      +x*(2221.1169345169674+x*(-3146.0842710357674+x*(2484.8748211148486+x*(-563.9211757298043 \
      +x*(-715.7375360627449+x*(607.9707583339692+x*(-143.07776760027488)))))))))))  \
        /(1 +x*(-15.096431493595128+x*(99.73477392261881+x*(-376.52638726561565+x*(888.2184994498242 \
      +x*(-1340.2481144619549+x*(1262.815388509247+x*(-679.0406779170286+x*(159.47091365763484)))))))))

def eHmin2_approx(x):
    return np.sqrt(e2H(x,argMin2_approx(x)))

def lbd_critH(e,c):    # return domain name and tuple of H-critical values: (name,((lambdaC,omegaC),...))
    c2 = c*c
    def lbd_omegaH(h1,h2):
        h = optimize.brentq(lambda x: e*e-min(1.1,e2H(c,x)), h1, h2)
        u = solUofH(c,h)
        w = np.sqrt(u/(1-u))
        w2 = w*w
        x = one_minus_h_coth(h)
        z = (c2+w2)/(1+w2)
        d = 6 - 2*(c2+w2*np.square(np.tan(np.pi/2 - h)))/(z*z*x) + 3*w2*np.square((1-z*x)/(z*x))
        lbd = np.pi/2 - np.arctan(w)
        return (lbd, lbd + np.arccos((1-4/d)/e)*np.sign(d)*np.sign(x*z-1))

# first, most likely case
    if (c>c0r and e<eH0(c)):                 # H0b - no critical value - Earth's case
        #print('H0b')
        return ('H0b',())
# e>1/3
    if (c>c0s and e>1/3 and e>eH0(c)):       # H0a - no critical value
        #print('H0a')
        return ('H0a',())
    if (c>c0max and e>1/3 and e>eHmax_approx(c)):       # H1c - 1 critical value
        #print('H1c')
        return ('H1c',( lbd_omegaH(h0(c),argMin2_approx(c)), ))
    if (c>c0min2 and e>1/3 and e>eHmin2_approx(c)):     # H3b - 3 critical values
        #print('H3b')
        return ('H3b',( lbd_omegaH(h0(c),argMin2_approx(c)), \
                       lbd_omegaH(argMin2_approx(c),argMax_approx(c)), \
                       lbd_omegaH(argMax_approx(c),argMin1_approx(c)) ))
    if (e>1/3):                                         # H1d - 1 critical value
        #print('H1d')
        return ('H1d',( lbd_omegaH(h0(c),argMin1_approx(c)), ))
# e<1/3 && e>eH0
    if (c>c0s and e>eH0(c) and e<eHmin1_approx(c)):     # H1a - 1 critical value
        #print('H1a')
        return ('H1a',( lbd_omegaH(h0(c),argMax_approx(c)), ))
    if (c>c0s and e>eH0(c) and e>eHmax_approx(c)):      # H1b - 1 critical value
        #print('H1b')
        return ('H1b',( lbd_omegaH(argMin1_approx(c), np.pi), ))
    if (c>c0s and e>eH0(c)):                            # H3a - 3 critical values
        #print('H3a')
        return ('H3a',( lbd_omegaH(h0(c),argMax_approx(c)), \
            lbd_omegaH(argMax_approx(c),argMin1_approx(c)), \
            lbd_omegaH(argMin1_approx(c),np.pi) ))
# e<1/3 && e<eH0 && e<eHmin1
    if (e<eHmin1_approx(c) and c>c0min2 and e>eHmin2_approx(c)):   # H2c - 2 critical values
        #print('H2c')
        return ('H2c',( lbd_omegaH(h0(c),argMin2_approx(c)), \
            lbd_omegaH(argMin2_approx(c),argMax_approx(c)) ))
    if (e<eHmin1_approx(c)):                            # H0c - no critical value
        #print('H0c')
        return ('H0c',())
# e<1/3 && e<eH0 && e>eHmin1
    if (c>c0max and e>eHmax_approx(c)):                 # H2a - 2 critical values
        #print('H2a')
        return ('H2a',( lbd_omegaH(h0(c),argMin2_approx(c)), \
            lbd_omegaH(argMin1_approx(c),np.pi) ))
    if (c>carg12_approx and e>eHmin2_approx(c)):        # H4b - 4 critical values
        #print('H4b')
        return ('H4b',( lbd_omegaH(h0(c),argMin2_approx(c)), \
            lbd_omegaH(argMin2_approx(c),argMax_approx(c)), \
            lbd_omegaH(argMax_approx(c),argMin1_approx(c)), \
            lbd_omegaH(argMin1_approx(c),np.pi) ))
    if (c>c0min2 and e>eHmin2_approx(c)):               # H4a - 4 critical values
        #print('H4a')
        return ('H4a',( lbd_omegaH(h0(c),argMin2_approx(c)), \
            lbd_omegaH(argMin2_approx(c),argMax_approx(c)), \
            lbd_omegaH(argMax_approx(c),argMin1_approx(c)), \
            lbd_omegaH(argMin1_approx(c),np.pi) ))
    else:                                               # H2b - 2 critical values
        #print('H2b')
        return ('H2b',( lbd_omegaH(h0(c),argMin1_approx(c)), \
            lbd_omegaH(argMin1_approx(c),np.pi) ))


def lbd_minmaxH(e,eps,per,critPoints):   #   π/2 < per < 3π/2
    c = np.cos(eps)
    domainH = critPoints[0]
    n = len(critPoints[1])
    #print("lbd_minmaxH - per = ",per)
    #lbdB0 = lambda_max(e,p0)[0]
    lbdB1 = 1.01*lambda_max(e,per)[1]   # safer to go a bit beyond the limit
    
    def derPhi(x):      # replacement for 0==DminmaxPhi_Dlon(x,eps,e,per)... numerically more favorable...
        # derPhi(0.3) > 0 (always true ..)
        # derPhi(π/2) < 0 (always true ..)
        # ->  there is 1 or 3 roots between 0.3 and π/2. Finding the first one is easy.
        if x>=np.pi/2:
            return -1
        cosxp = np.cos(x-per)
        tanx = np.tan(x)
        f = np.sqrt(max(0, 1 + 2*e*np.sin(x-per)*tanx/(1-e*cosxp)))
        cosa = (cosxp-e)/(1-e*cosxp)
        c2 = c*c
        cw = (c2*tanx*tanx+1)/(tanx*tanx+1)
        g = -tanx*np.sqrt( max(0,(1 + 2*e*(2*e-cosa-3*e*cosa*cosa)/(1-e*e))*cw - c2*f*f ) )
        if (g==0):
            return f*f*cw - 1
        return f*(f-g*(np.pi+np.arctan(f/g)))*cw - 1

    def root(g1=.3,g2=lbdB1):     # returns one root of derPhi(x)==0, with x between g1 and g2
        #print("per,f1,f2 = ",per,", ",derPhi(g1),", ",derPhi(g1))
        return np.array([optimize.root_scalar(lambda x: derPhi(x), method='brentq', bracket=(g1,g2)).root])
    def triple_root(ga=.3,gb=lbdB1):     # returns 3 roots of derPhi(x)==0, with x between ga and gb
        #print("ga,gb,gb° = ",ga,", ",gb,", ",gb*180/np.pi)
        #print("per°,fa,fb = ",per*180/np.pi,", ",derPhi(ga),", ",derPhi(gb))
        #my_plot(derPhi,ga,gb)
        x1 = optimize.root_scalar(lambda x: derPhi(x), method='brentq', bracket=(ga,gb)).root
        to_solve2 = (lambda x: derPhi(x)/(x-x1))
        gc = 2*np.arctan( (1+e*(3-3*np.pi+2*per))/(1+3*e) )  # almost equal (to order 1) to lambda_max(e,per)[1], but smaller (?)
        fa = to_solve2(ga)   # < 0
        fb = to_solve2(gb)   # < 0
        fc = to_solve2(gc)   # ?
        if (fc < 0):
        #if (False):
            ntry = 20
            #to_solve3 = (lambda x: to_solve2(gc-10**x))
            #print("lbdB = ",lambda_max(e,per)[1],"   x2 = ",x2)
            #print("fa = ",fa,"   fb = ",fb,"   fc = ",fc)
            #print("fa = ",to_solve3(-np.log10(gc-ga)),"   fc+ = ",to_solve3(10))
            for i in range(ntry):  # 0 to 19
                gi = gc-10**( -(ntry-i)/2 )   # 10**(-9.5) to 10**(-.5)
                fi = to_solve2(gi)
                #print("f(",i,") = ",fi)
                if fi>0:
                    gc = gi
                    fc = fi
                    break
        if (fc < 0):
            fabc = max(fa,max(fc,fb))     # < 0
            #print("fa = ",fa)
            #print("fb = ",fb)
            
            #my_plot(to_solve2,1.54,gb)
            test_solve2 = (lambda z: to_solve2(z*ga+(1-z)*gb))
            for i in range(10000):   #certainly not optimal !!  -- should biais the distribution towards gb for per near the boundaries (π/2,3π/2)
                zz = random.random()  # 0 < zz < 1
                if test_solve2(zz) > fabc:    # found a bracketing for a maximum of test_solve2... which should be >0
                    max2 = optimize.minimize_scalar(lambda z: -test_solve2(z), bracket=(0,zz,1))
                    break
            #print("i, zz = ",i,", ",zz)
            assert i<9999, "error in lbd_minmaxH.triple_root : i="+str(i)
            assert max2.fun<0, "error in lbd_minmaxH.triple_root : i="+str(i)
            gc = max2.x*ga+(1-max2.x)*gb
        x2 = optimize.root_scalar(lambda x: to_solve2(x), method='brentq', bracket=(ga,gc)).root
        x3 = optimize.root_scalar(lambda x: to_solve2(x), method='brentq', bracket=(gc,gb)).root
        #print("3sol = ",np.array([x1,x2,x3]))
        return np.array([x1,x2,x3])
    
    if len(critPoints[1])==0:
        omCrit = np.array([])
    else:
        omCrit = np.sort(mod_2pi_minus_pi_over_2(np.array(critPoints[1])[:,1]))
    
    if domainH == 'H0b' or domainH == 'H0a':
        return root()
    elif domainH == 'H0c':
        return triple_root()
    elif domainH == 'H1a' or domainH == 'H1b':
        if per>omCrit:
            return root()
        else:
            return triple_root()
    elif domainH == 'H1c' or domainH == 'H1d':
        if per<omCrit:
            return root()
        else:
            return triple_root()
    elif domainH == 'H2a' or domainH == 'H2b' or domainH == 'H2c':
        if per>omCrit[0] and per<omCrit[1]:
            return root()
        else:
            return triple_root()
    elif domainH == 'H3b':
        if per<omCrit[0] or (per>omCrit[1] and per<omCrit[2]):
            return root()
        else:
            return triple_root()
    elif domainH == 'H3a':
        if per<omCrit[0] or (per>omCrit[1] and per<omCrit[2]):
            return triple_root()
        else:
            return root()
    elif domainH == 'H4a' or domainH == 'H4b':
        if (per>omCrit[0] and per<omCrit[1]) or (per>omCrit[2] and per<omCrit[3]):
            return root()
        else:
            return triple_root()
    return

def lambda_minmaxH(ecc,eps,per,critPoints):
    #print("lambda_minmaxH - per = ",per)
    if per<np.pi/2:
        lbd_1 = lbd_minmaxH(ecc,eps,np.pi-per,critPoints)
        lbd_2 = lbd_minmaxH(ecc,eps,np.pi+per,critPoints)
        return np.concatenate([np.pi-lbd_1,-np.pi+lbd_2])
    else:
        lbd_1 = lbd_minmaxH(ecc,eps,per,critPoints)
        lbd_2 = lbd_minmaxH(ecc,eps,2*np.pi-per,critPoints)
        return np.concatenate( [lbd_1, 0*np.pi-lbd_2] )

#def plot_lbd_minmaxH(e,eps,small = 1e-4):  # small ... to avoid exact boundary values... needs fine tuning...
#    critPoints = lbd_critH(e,np.cos(eps))
#    domainH = critPoints[0]
#    if len(critPoints[1])==0:
#        omCrit = np.array([np.pi/2,3*np.pi/2])
#    else:
#        omCrit0 = mod_2pi_minus_pi_over_2(np.array(critPoints[1])[:,1])
#        omCrit = np.sort(np.concatenate((omCrit0,2*np.pi-omCrit0,[np.pi/2,3*np.pi/2])))
#
#    def my_plot_array(f,a,b,n=100,label='',color=""):
#        x = np.linspace(a,b,n)
#        n = len(f(a))
#        y = np.empty([len(x),n])
#        for i in range(len(x)):
#            y[i,:] = f(x[i])
#        for k in range(n):
#            if color=='':
#                plt.plot(x,y[:,k],label=label)
#            else:
#                plt.plot(x,y[:,k],label=label,color=color)
#
#    n = len(omCrit)
#    for i in range(n-1):
#        if omCrit[i]+small < omCrit[i+1]-small:
#            my_plot_array(lambda z: lambda_minmaxH(e,eps,z,critPoints),omCrit[i]+small, omCrit[i+1]-small,color='b')
#        else:
#            print("warning: plot domain too small")
#
#    om1 = np.linspace(np.pi/2+1e-5,3*np.pi/2-1e-5,100)
#    lbd0 = list(map(lambda z: lambda_max(e,z)[0],om1))
#    lbd1 = list(map(lambda z: lambda_max(e,z)[1],om1))
#    plt.fill_between(om1,lbd0,-np.pi/2,color='pink')
#    plt.fill_between(om1,lbd1,np.pi/2,color='pink')
#    plt.axis([np.pi/2, 3*np.pi/2, -np.pi/2, np.pi/2])
#    plt.title(f"e = {e}  c = cos($\\epsilon$) = {np.cos(eps):.3f} :{domainH}")
#    plt.xlabel(r'$\varpi$')
#    plt.ylabel(r'$\lambda$')
#    plt.show()
#    return

def subplot_lbd_minmaxH(subplt,e,eps,small = 1e-4):  # small ... to avoid exact boundary values... needs fine tuning...
    critPoints = lbd_critH(e,np.cos(eps))
    critPts = np.array(critPoints[1])
    domainH = critPoints[0]
    if len(critPoints[1])==0:
        omCrit = np.array([np.pi/2,3*np.pi/2])
    else:
        omCrit0 = mod_2pi_minus_pi_over_2(critPts[:,1])
        omCrit = np.sort(np.concatenate((omCrit0,2*np.pi-omCrit0,[np.pi/2,3*np.pi/2])))

    def my_plot_array(f,a,b,n=100,label='',color=""):
        x = np.linspace(a,b,n)
        n = len(f(a))
        y = np.empty([len(x),n])
        for i in range(len(x)):
            y[i,:] = f(x[i])
        for k in range(n):
            if color=='':
                subplt.plot(x,y[:,k],label=label)
            else:
                subplt.plot(x,y[:,k],label=label,color=color)

    n = len(omCrit)
    for i in range(n-1):
        if omCrit[i]+small < omCrit[i+1]-small:
            my_plot_array(lambda z: lambda_minmaxH(e,eps,z,critPoints),omCrit[i]+small, omCrit[i+1]-small,color='b')
        else:
            print("warning: plot domain too small")

    for i in range(len(critPoints[1])):
        subplt.plot([omCrit0[i]], [mod_2pi_minus_pi_over_2(critPts[i][0])], 'o',color='k')
        subplt.plot([2*np.pi-omCrit0[i]], [mod_2pi_minus_pi_over_2(-critPts[i][0])], 'o',color='k')

    om1 = np.linspace(np.pi/2+1e-5,3*np.pi/2-1e-5,100)
    lbd0 = list(map(lambda z: lambda_max(e,z)[0],om1))
    lbd1 = list(map(lambda z: lambda_max(e,z)[1],om1))
    subplt.fill_between(om1,lbd0,-np.pi/2,color='pink')
    subplt.fill_between(om1,lbd1,np.pi/2,color='pink')
    subplt.axis([np.pi/2, 3*np.pi/2, -np.pi/2, np.pi/2])
    subplt.set_title(f"e = {e}  c = cos($\\epsilon$) = {np.cos(eps):.3f} :{domainH}")
    subplt.set_xlabel(r'$\varpi$')
    subplt.set_ylabel(r'$\lambda$')

def plot_lbd_minmaxH(e,eps,small = 1e-4):       # plot a single figure
    subplot_lbd_minmaxH(plt.gca(),e,eps,small)
    plt.show()

def subplot_boundaries(subplt,ecc):
    xyc = xc_yc(ecc)
    pp = np.pi/2-1e-5
    ccolor = 'pink'
    x,y = [-pp,3*pp,3*pp,-pp,-pp],[3*pp,3*pp,-pp,-pp,3*pp]
    subplt.plot(x, y, color=ccolor,linewidth=1.0)
    x,y = [pp,pp],[-pp,3*pp]
    subplt.plot(x,y,y,x, color=ccolor,linewidth=1.)
    om0 = np.linspace(-pp,pp,100)
    lbd0 = list(map(lambda z: lambda_max(ecc,z)[0],om0))
    lbd1 = list(map(lambda z: lambda_max(ecc,z)[1],om0))
    subplt.fill_between(om0,lbd0,pp,color=ccolor)
    subplt.fill_between(om0,lbd1,3*pp,color=ccolor)
    om1 = np.linspace(np.pi/2+1e-5,3*pp,100)
    lbd0 = list(map(lambda z: lambda_max(ecc,z)[0],om1))
    lbd1 = list(map(lambda z: lambda_max(ecc,z)[1],om1))
    subplt.fill_between(om1,lbd0,-np.pi/2+1e-5,color=ccolor)
    subplt.fill_between(om1,lbd1,np.pi/2+1e-5,color=ccolor)
    if (xyc[0]!=0):
        om0 = np.linspace(-pp,xyc[1]-1e-5,100)
        lbd0 = list(map(lambda z: lambda_min(ecc,z)[1],om0))
        lbd1 = list(map(lambda z: lambda_min(ecc,z)[2],om0))
        subplt.fill_between(om0,lbd0,lbd1,color=ccolor)
        om0 = np.linspace(xyc[2]+1e-5,pp,100)
        lbd0 = list(map(lambda z: lambda_min(ecc,z)[1],om0))
        lbd1 = list(map(lambda z: lambda_min(ecc,z)[2],om0))
        subplt.fill_between(om0,lbd0,lbd1,color=ccolor)
        om0 = np.linspace(np.pi/2+1e-5,xyc[3]-1e-5,100)
        lbd0 = list(map(lambda z: lambda_min(ecc,z)[1],om0))
        lbd1 = list(map(lambda z: lambda_min(ecc,z)[2],om0))
        subplt.fill_between(om0,lbd0,lbd1,color=ccolor)
        om0 = np.linspace(xyc[4]+1e-5,3*pp,100)
        lbd0 = list(map(lambda z: lambda_min(ecc,z)[1],om0))
        lbd1 = list(map(lambda z: lambda_min(ecc,z)[2],om0))
        subplt.fill_between(om0,lbd0,lbd1,color=ccolor)
    subplt.axis([-np.pi/2, 3*np.pi/2, -np.pi/2, 3*np.pi/2])
    subplt.text(.0, .0, 'L-branch',horizontalalignment='center')
    subplt.text(.0, np.pi, 'H-branch',horizontalalignment='center')
    subplt.text(np.pi, .0, 'H-branch',horizontalalignment='center')
    subplt.text(np.pi, np.pi, 'L-branch',horizontalalignment='center')
    subplt.set_title(r'eccentricity = '+str(ecc))
    subplt.set_xlabel(r'$\varpi$')
    subplt.set_ylabel(r'$\lambda$')

def plot_Hdomains():
    my_plot(lambda z: eHmin1_approx(z),0,1, label='eHmin1')
    my_plot(lambda z: eHmax_approx(z),c0max,1, label='eHmax')
    my_plot(lambda z: eHmin2_approx(z),c0min2,c0r, label='eHmin2')
    my_plot(lambda z: eH0(z),c0s,1, label='eH0')
    my_plot(lambda z: 1/3,0,1, label='1/3')
    plt.text(.6, .7, 'H0a')
    plt.text(.36, .7, 'H1c')
    plt.text(.33, .6, 'H3b')
    plt.text(.2, .7, 'H1d')
    plt.text(.2, .15, 'H2b')
    plt.text(.53, .2, 'H1b')
    plt.text(.65, .13, 'H1a')
    plt.text(.65, .22, 'H3a', fontsize=8)
    plt.text(.8, .1, 'H0b')
    plt.text(.41, .01, 'H0c')
    plt.text(.5, .03, 'H2c')
    plt.text(.49, .08, 'H4b', fontsize=8)
    plt.text(.41, .12, 'H4a')
    plt.text(.42, .22, 'H2a')
    plt.axis([0, 1, 0, 1])
    plt.title('Critical points on the H-branch')
    plt.xlabel(r'c = cos($\epsilon$) = cos(obliquity)')
    plt.ylabel('e = eccentricity')
    plt.legend()

def plot_Hdomains_zoomed():
    x0=0.52
    x1=0.54
    #my_plot(lambda z: eHmin1_approx(z),x0,x1)
    my_plot(lambda z: eHmax_approx(z)-eHmin1_approx(z),x0,x1, label='eHmax-eHmin1',color='#ff7f0e')
    #my_plot(lambda z: eHmin2_approx(z),c0min2,c0r)
    my_plot(lambda z: eH0(z)-eHmin1_approx(z),x0,x1, label='eH0-eHmin1',color='#d62728')
    #my_plot(lambda z: 1/3,0,1)
    plt.text(.525, .00001, 'H2a')
    plt.text(.535, .00001, 'H1b')
    plt.text(.521, .0000022, 'H4a')
    plt.text(.536, .0000022, 'H3a')
    plt.text(.5311, .0000001, 'H4b', fontsize=8)
    plt.axis([x0, x1, 0, .00002])
    plt.title('Critical points on the H-branch')
    plt.xlabel(r'c = cos($\epsilon$) = cos(obliquity)')
    plt.ylabel('e - eHmin1')
    plt.legend()

def plot_Ldomains():
    my_plot(lambda z: eLmin(z),0,np.sqrt(3/5), label='eLmin') #,color='r')
    my_plot(lambda z: (1-np.square(z))/(3-np.square(z)),0,1, label='eL1') #,color='r')
    xe = np.linspace(0.1667,0.6596,50)
    gg = np.empty(len(xe))
    for i in range(len(xe)):
        gg[i] = cZero(xe[i])
    plt.plot(gg,xe,':', label='cZero')
    plt.plot([np.sqrt(3/5)], [1/6], '.',color='k')
    plt.text(.5, .6, 'L1a')
    plt.text(.9, .3, 'L1b')
    plt.text(.3, .1, 'L0')
    plt.text(.1, .3, 'L2')
    plt.axis([0, 1, 0, 1])
    plt.title('Critical points on the L-branch')
    plt.xlabel(r'c = cos($\epsilon$) = cos(obliquity)')
    plt.ylabel('e = eccentricity')
    plt.legend()

def polar(eps,lbd):
    sin_delta = np.sin(eps)*np.sin(lbd)
    return np.arctan( np.sqrt(1-sin_delta*sin_delta)/sin_delta )

def L_limits(ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    #print("prep = ",prep)
    x_limits = list(lambda_min(ecc,prep))     # limits on the polar circles: y = ± polar(x) (len = 2 or 4)
    xx_limits = sorted(x_limits+x_limits)     # we further add corresponding points at the poles (len = 4 or 8)
    yy_limits = list(-polar(eps,xx_limits))   # ... for the polar circles
    if prep>np.pi/2:
        yy_limits[0]  = -np.pi/2                  # ... for the poles
        yy_limits[-1] = np.pi/2
    else:
        yy_limits[0]  = np.pi/2                  # ... for the poles
        yy_limits[-1] = -np.pi/2
    if len(x_limits)>2:
        yy_limits[2]=-yy_limits[2]  #
        yy_limits[5]=-yy_limits[5]
        #print("yy_limits = ",yy_limits)
        if prep>0 and prep<np.pi:
            yy_limits[3] = np.pi/2
            yy_limits[4] = np.pi/2
        else:
            yy_limits[3] = -np.pi/2
            yy_limits[4] = -np.pi/2
    #print("xx_limits = ",xx_limits)
    #print("yy_limits = ",yy_limits)
    xxyy_limits = ()
    for i in range(len(xx_limits)):
        xxyy_limits += ((xx_limits[i],yy_limits[i]),)
    return xxyy_limits

def L_branch(ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    xy_limits = L_limits(ecc,eps,prep)
    critP = lbd_critL(ecc,np.cos(eps))
    min_max = ()
    if prep < np.pi/2:
        for mm in lbd_minmaxL(ecc,eps,prep,critP):
            min_max += (mm,)
        for mm in lbd_minmaxL(ecc,eps,-prep,critP):
            min_max += (-mm,)
    else:
        for mm in lbd_minmaxL(ecc,eps,prep-np.pi,critP):  # in [-π/2,π/2]
            min_max += (mm+np.pi,)                        # in [π/2,3π/2]
        for mm in lbd_minmaxL(ecc,eps,np.pi-prep,critP):  # in [-π/2,π/2]
            min_max += (np.pi-mm,)                        # in [π/2,3π/2]
    minmax_Phi = ()
    for i in range(len(min_max)):
        minmax_Phi += ((min_max[i],minmaxPhi(min_max[i],eps,ecc,prep)),)
    #print("minmax_Phi = ",minmax_Phi)
    return sorted(xy_limits+minmax_Phi, key=itemgetter(0))

def minmaxPhiL(lon,eps,e,per):   # for the L-branch
    if lon==np.pi/2:
        return -polar(eps,np.pi/2)
    elif lon==3*np.pi/2:
        return polar(eps,np.pi/2)
    else:
        return minmaxPhi(lon,eps,e,per)

def H_branch(ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    x_limits = list(lambda_max(ecc,prep))     # limits on the polar circles: y = ± polar(x)
    y_limits = list(polar(eps,x_limits))
    x_limits += x_limits
    if prep>np.pi/2:
        y_limits += list((-np.pi/2,np.pi/2))
    else:
        y_limits += list((np.pi/2,-np.pi/2))
    xy_limits = ()
    for i in range(len(x_limits)):
        xy_limits += ((x_limits[i],y_limits[i]),)
    critH = lbd_critH(ecc,np.cos(eps))
    min_max = np.sort(mod_2pi_minus_pi_over_2(np.array(lambda_minmaxH(ecc,eps,prep,critH))))
    #print("min_max = ",min_max)
    minmax_Phi = ()
    for i in range(len(min_max)):
        minmax_Phi += ((min_max[i],minmaxPhi(min_max[i],eps,ecc,prep)),)
    #print("minmax_Phi = ",minmax_Phi)
    return sorted(xy_limits+minmax_Phi)

def L_minmax(lat,ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    brL = L_branch(ecc,eps,prep)
    small = 1e-8
    #print("brL = ",brL)
    #print("pre = ",pre)
    #print("eps = ",eps)
    #print("ecc = ",ecc)
    minmax_list = ()
    for i in range(len(brL)-1):
        if (brL[i][1]-lat)*(brL[i+1][1]-lat) < 0:
            #print("L = ",brL[i][0],", ",brL[i+1][0])
            #print("lat = ",lat,", ",brL[i][1],", ",brL[i+1][1])
            #print("lat = ",lat,", ",minmaxPhi(brL[i][0],eps,ecc,prep),", ",minmaxPhi(brL[i+1][0],eps,ecc,prep))
            #print("lat = ",lat,", ",minmaxPhiL(brL[i][0],eps,ecc,prep),", ",minmaxPhiL(brL[i+1][0],eps,ecc,prep))
            #if brL[i][0]==brL[i+1][0]:
            if abs(brL[i][0]-brL[i+1][0])<small:
                x = brL[i][0]
            elif abs(lat-minmaxPhiL(brL[i][0],eps,ecc,prep))<small:
                x = brL[i][0]
            elif abs(lat-minmaxPhiL(brL[i+1][0],eps,ecc,prep))<small:
                x = brL[i+1][0]
            else:
                #print("brL[i][0] = ",brL[i][0],", ",minmaxPhi(brL[i][0],eps,ecc,prep))
                #print("brL[i+1][0] = ",brL[i+1][0],", ",minmaxPhi(brL[i+1][0],eps,ecc,prep))
                #print("brL[i][0] = ",brL[i][0],", ",minmaxPhiL(brL[i][0],eps,ecc,prep))
                #print("brL[i+1][0] = ",brL[i+1][0],", ",minmaxPhiL(brL[i+1][0],eps,ecc,prep))
                x = optimize.root_scalar(lambda z: lat-minmaxPhiL(z,eps,ecc,pre), method='brentq', bracket=(brL[i][0],brL[i+1][0])).root
            fx = inso.inso_dayly_radians(x,lat,eps,ecc,prep)
            minmax_list += ((x,np.maximum(0,fx)),)
    #print("x, i = ",x,", ",solar*inso_dayly_radians(x,lat,obliquity,ecc,pre))
    return minmax_list

def H_minmax(lat,ecc,eps,pre):
    brH = H_branch(ecc,eps,pre)
    small = 1e-8
    #print("brH = ",brH)
    minmax_list = ()
    for i in range(len(brH)-1):
        if (brH[i][1]-lat)*(brH[i+1][1]-lat) < 0:
            #print("H = ",brH[i][0],", ",brH[i+1][0])
            if brH[i][0]==brH[i+1][0]:
                x = brH[i][0]
            else:
                x = optimize.root_scalar(lambda z: lat-minmaxPhi(z,eps,ecc,pre), method='brentq', bracket=(brH[i][0],brH[i+1][0])).root
            minmax_list += ((x,inso.inso_dayly_radians(x,lat,eps,ecc,pre)),)
    #print("x, i = ",x,", ",solar*inso_dayly_radians(x,lat,obliquity,ecc,pre))
    return minmax_list

def minmax_dayly_inso(lat,ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    return np.array(L_minmax(lat,ecc,eps,prep)+H_minmax(lat,ecc,eps,prep))

def largest_max(lat,ecc,eps,pre):
    #prep = mod_2pi_minus_pi_over_2(pre)
    #minmax_inso = np.array(L_minmax(lat,ecc,eps,prep)+H_minmax(lat,ecc,eps,prep))
    minmax_inso = minmax_dayly_inso(lat,ecc,eps,pre)
    max_i = np.argmax(minmax_inso[:,1])
    return minmax_inso[max_i]

def smallest_min(lat,ecc,eps,pre):
    #prep = mod_2pi_minus_pi_over_2(pre)
    #minmax_inso = np.array(L_minmax(lat,ecc,eps,prep)+H_minmax(lat,ecc,eps,prep))
    minmax_inso = minmax_dayly_inso(lat,ecc,eps,pre)
    min_i = np.argmin(minmax_inso[:,1])
    return minmax_inso[min_i]


###################
#
#   for a given insolation value "thresh", returns the lists of true longitudes for which the daily insolation equals "thresh"
#   - in up_list when insolation increases
#   - in down_list when insolation decreases
#
def up_and_down_list(min_max_table,thresh,lat,ecc,eps,prep):
    up_list = ()
    down_list = ()
    for i in range(len(min_max_table)-1):
        if (min_max_table[i][1]-thresh)*(min_max_table[i+1][1]-thresh) < 0:
            a = min_max_table[i][0]
            b = min_max_table[i+1][0]
            if b<a:
                b += 2*np.pi
            x = optimize.root_scalar(lambda z: thresh-inso.inso_dayly_radians(z,lat,eps,ecc,prep), method='brentq', bracket=(a,b)).root
            if (min_max_table[i+1][1]>min_max_table[i][1]):
                up_list += (x,)
            else:
                down_list += (x,)
    return (up_list,down_list)


###################
#
#   return the insolation value "thresh" above which the insolation is half of the year
#
def inso_caloric_threshold(lat,ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    LMinMax = L_minmax(lat,ecc,eps,prep)
    HMinMax = H_minmax(lat,ecc,eps,prep)
    minmax_inso = np.array(LMinMax+HMinMax+(LMinMax[0],))
    max_i = np.max(minmax_inso[:,1])
    min_i = np.min(minmax_inso[:,1])
    def f(x):
        if (x<=min_i):
            return 1
        if (x>=max_i):
            return 0
        (up_list,down_list) = up_and_down_list(minmax_inso,x,lat,ecc,eps,pre)
        if (up_list==() and minmax_inso[0][1]>x):  # always above threshold
            return 1
        elif (up_list==() and minmax_inso[1][1]<x):  # always below threshold
            return 0
        else:
            time_above = 0
            for i in range(len(up_list)):
                down_i = down_list[i]
                up_i = up_list[i]
                if down_i<up_list[i]:
                    down_i += 2*np.pi
                time_above += inso.length_of_season(up_list[i],down_i,ecc,pre)
            return time_above
    thresh = optimize.root_scalar(lambda z: 0.5-f(z), method='brentq', bracket=(min_i,max_i)).root
    return thresh


###################
#
#   integrated insolation above some threshold. Dimensionless.
#       it should be multiplied by solar constant (W/m2) and year duration (s) to get the result in J/m2
#
def integrated_inso_above(thresh,lat,ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    LMinMax = L_minmax(lat,ecc,eps,prep)
    HMinMax = H_minmax(lat,ecc,eps,prep)
    minmax_inso = np.array(LMinMax+HMinMax+(LMinMax[0],))   # ( (lon_min1, inso_min1), (lon_max1,inso_max1)..., (lon_min1, inso_min1))
    (up_list,down_list) = up_and_down_list(minmax_inso,thresh,lat,ecc,eps,prep)
    if (up_list==() and minmax_inso[0][1]>thresh):  # if insolation always above threshold (first min>thresh and no crossing), integrate over [0,2π]
        up_list += (0,)
        down_list += (2*np.pi,)
    integral = 0.
    for i in range(len(up_list)):
        if down_list[i]>up_list[i]:
            integral += inso.inso_irrad(down_list[i],lat,eps)-inso.inso_irrad(up_list[i],lat,eps)
        else:
            integral += inso.inso_irrad(2*np.pi+down_list[i],lat,eps)-inso.inso_irrad(up_list[i],lat,eps)
    pisq = 2*np.pi*np.pi*np.sqrt(1-ecc*ecc)
    return integral/pisq

def integrated_inso_below(thresh,lat,ecc,eps,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    LMinMax = L_minmax(lat,ecc,eps,prep)
    HMinMax = H_minmax(lat,ecc,eps,prep)
    minmax_inso = np.array(LMinMax+HMinMax+(LMinMax[0],))
    (up_list,down_list) = up_and_down_list(minmax_inso,thresh,lat,ecc,eps,prep)
    if (up_list==() and minmax_inso[1][1]<thresh):  # if insolation always above threshold (first max<thresh and no crossing), integrate over [0,2π]
        up_list += (0,)
        down_list += (2*np.pi,)
    integral = 0.
    up_list += (up_list[-1],)
    for i in range(len(down_list)):
        if up_list[i+1]>down_list[i]:
            integral += inso.inso_irrad(up_list[i+1],lat,eps)-inso.inso_irrad(down_list[i],lat,eps)
        else:
            integral += inso.inso_irrad(2*np.pi+up_list[i+1],lat,eps)-inso.inso_irrad(down_list[i],lat,eps)
    pisq = 2*np.pi*np.pi*np.sqrt(1-ecc*ecc)
    return integral/pisq

###################
#
#   the caloric summer insolation
#       dimensionless: it should be multiplied by solar constant (W/m2) to get the result in W/m2
#   NB: it returns 2*integrated_inso_above() since it is multiplied by T (year) and divided by T/2 (interval duration)
#           to return the averaged value (W/m2)
#       it should be multiplied by solar*T/2 to get the result in J/m2
#
def inso_caloric_summer(lat,ecc,eps,pre):
    return 2*integrated_inso_above(inso_caloric_threshold(lat,ecc,eps,pre),lat,ecc,eps,pre)

def inso_caloric_winter(lat,ecc,eps,pre):
    return 2*integrated_inso_below(inso_caloric_threshold(lat,ecc,eps,pre),lat,ecc,eps,pre)



def plot_inso_with_min_max(solar,ecc,obliquity,pre):
    prep = mod_2pi_minus_pi_over_2(pre)
    small = 1e-6
    delta = 1*deg_to_rad
    x_lon = np.arange(-np.pi/2,3*np.pi/2,delta)
    x_lat = np.arange(-np.pi/2,np.pi/2,delta)
    x,y = np.meshgrid(x_lon,x_lat)
    z = solar*inso.inso_dayly_radians(x,y,obliquity,ecc,prep)
    #plt.figure(figsize=(14, 9))
    plt.axis([-np.pi/2,3*np.pi/2, -np.pi/2,np.pi/2])
    CS=plt.contour(x,y,z,levels=11,colors='0.2',linewidths=1.0)
    plt.clabel(CS, fontsize=10, fmt="%.0f")
    #plt.colorbar()
    
    #polar day:
    my_plot(lambda z: polar(obliquity,z),small,np.pi-small,color='k')
    my_plot(lambda z: polar(obliquity,z),-np.pi/2,-small,color='k')
    my_plot(lambda z: polar(obliquity,z),np.pi+small,3*np.pi/2,color='k')
    #polar night:
    x_night = np.linspace(small,np.pi-small,100)
    y_night = -polar(obliquity,x_night)
    plt.fill_between(x_night,y_night,-np.pi/2,color='0.7')
    x_night = np.linspace(-np.pi/2,-small,100)
    y_night = -polar(obliquity,x_night)
    plt.fill_between(x_night,y_night,np.pi/2,color='0.7')
    x_night = np.linspace(np.pi+small,3*np.pi/2,100)
    y_night = -polar(obliquity,x_night)
    plt.fill_between(x_night,y_night,np.pi/2,color='0.7')
    
    def plot_day_line(x0,sgn):
        x_day = np.linspace(x0,x0,2)
        y_day = np.linspace(polar(obliquity,x0),sgn*np.pi/2,2)
        plt.plot(x_day,y_day,'r')
    
    #plot the L-branch from polar night to polar night (possibly 2 branches)
    x_min_limits = lambda_min(ecc,prep)
    #print("x_min_limits = ",x_min_limits)
    x_min = np.linspace(x_min_limits[0]+small,x_min_limits[1]-small,50)
    gg = np.empty(len(x_min))
    for i in range(len(x_min)):
        gg[i] = minmaxPhi(x_min[i],obliquity,ecc,prep)
    plt.plot(x_min,gg,'r')
    if len(x_min_limits)>2:
        x_min = np.linspace(x_min_limits[2]+small,x_min_limits[3]-small,100)
        gg = np.empty(len(x_min))
        for i in range(len(x_min)):
            gg[i] = minmaxPhi(x_min[i],obliquity,ecc,prep)
        plt.plot(x_min,gg,'r')
        if prep>0 and prep<np.pi:
            plot_day_line(x_min_limits[1],1)
            plot_day_line(x_min_limits[2],1)
        else:
            plot_day_line(x_min_limits[1],-1)
            plot_day_line(x_min_limits[2],-1)
    
#plot the H-branch from polar day to polar day
    x_max_limits = lambda_max(ecc,prep)
    #print("x_max_limits = ",x_max_limits)
    x_max = np.linspace(x_max_limits[0]+small,x_max_limits[1]-small,100)
    gg = np.empty(len(x_max))
    for i in range(len(x_max)):
        gg[i] = minmaxPhi(x_max[i],obliquity,ecc,prep)
    plt.plot(x_max,gg,'r')
    if prep>np.pi/2:
        plot_day_line(x_max_limits[0],-1)
        plot_day_line(x_max_limits[1],1)
    else:
        plot_day_line(x_max_limits[0],1)
        plot_day_line(x_max_limits[1],-1)
    
#blue dots for each turning point of each branch
    brL = L_branch(ecc,obliquity,pre)
    for i in range(len(brL)):
        plt.plot([brL[i][0]], [brL[i][1]], 'o',color='b')
    #print("brL = ",brL)
    
    brH = H_branch(ecc,obliquity,pre)
    for i in range(len(brH)):
        plt.plot([brH[i][0]], [brH[i][1]], 'o',color='b')
#print("brH = ",brH)
#plt.title("daily insolation with min and max (in W/m2)")
    plt.ylabel("latitude (radians)")
    plt.xlabel("true longitude (radians)")
#plt.show()

#   matplotlib default colors:
#['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']
#
############################################################
#     end of library minmax_inso
############################################################

#test when running: $ python minmax_inso.py

if __name__ == '__main__':
    
    #astro.astrofiles_path = "./astrofiles/"
    
    ##########  computing the critical values : one example
    ##########    for each of 14 the possible cases (H-branch)
    ##########    and each of 3 the possible cases (L-branch)
    if False:
        print("lbd_critH(.1,.9) = ",lbd_critH(.1,.9))
        print("lbd_critH(.4,.6) = ",lbd_critH(.4,.6))
        print("lbd_critH(.4,.4) = ",lbd_critH(.4,.4))
        print("lbd_critH(.4,.38) = ",lbd_critH(.4,.38))
        print("lbd_critH(.4,.2) = ",lbd_critH(.4,.2))
        print("lbd_critH(.1,.6) = ",lbd_critH(.1,.6))
        print("lbd_critH(.3,.6) = ",lbd_critH(.3,.6))
        print("lbd_critH(.1762,.6) = ",lbd_critH(.1762,.6))
        print("lbd_critH(.2,.45) = ",lbd_critH(.2,.45))
        print("lbd_critH(.2,.42) = ",lbd_critH(.2,.42))
        print("lbd_critH(.2,.4) = ",lbd_critH(.2,.4))
        print("lbd_critH(.05,.45) = ",lbd_critH(.05,.45))
        print("lbd_critH(.05,.5) = ",lbd_critH(.05,.5))
        print("lbd_critH(.109819,.532) = ",lbd_critH(.109819,.532))
        print("lbd_critL(.05,.9) = ",lbd_critL(.05,.9))
        print("lbd_critL(.4,.9) = ",lbd_critL(.4,.9))
        print("lbd_critL(.3,.2) = ",lbd_critL(.3,.2))
#
# plot the L and H domains in (c,e) space
#
    if False:
        plot_Ldomains()
        plt.show()
        plot_Hdomains()
        plt.show()
        plot_Hdomains_zoomed()
        plt.show()
#
# plot lambdaM as a function of precession (pi_tildeM) on the L-branch for some specific cases
#
    if False:
        plot_lbd_minmaxL(.066,22*deg_to_rad)  #L1a
        plot_lbd_minmaxL(.1,23*deg_to_rad)    #L1a
        plot_lbd_minmaxL(.95,np.arccos(.5))   #L1b
        plot_lbd_minmaxL(.3,np.arccos(.38))   #L1b
        plot_lbd_minmaxL(.3,np.arccos(.37))   #L2

#
# plot lambdaM as a function of precession (pi_tildeM) on the H-branch for all different cases
#
    if False:
        plot_lbd_minmaxH(.4,np.arccos(.7))  # H0a
        plot_lbd_minmaxH(.06,np.arccos(.8),small = 1e-3)  # H0b
        plot_lbd_minmaxH(.05,np.arccos(.45),small = 2e-4)  # H0c
        plot_lbd_minmaxH(.1,np.arccos(.6))  # H1a
        plot_lbd_minmaxH(.2,np.arccos(.6))  # H1b
        plot_lbd_minmaxH(.4,np.arccos(.4))  # H1c
        plot_lbd_minmaxH(.4,np.arccos(.3))  # H1d
        plot_lbd_minmaxH(.13,np.arccos(.45))  # H2a
        plot_lbd_minmaxH(.1,np.arccos(.4))  # H2b
        plot_lbd_minmaxH(.05,np.arccos(.5),small = 1e-4)  # H2c
        plot_lbd_minmaxH(.3114,np.arccos(.9))  # H3a
        plot_lbd_minmaxH(.4,np.arccos(.38))  # H3b
        plot_lbd_minmaxH(.1,np.arccos(.46))  # H4a
        plot_lbd_minmaxH(0.1098185,np.arccos(.532),small = 1e-8)  # H4b


#
# plot the summary figure for Earth's present-day (J2000) astronomical parameters
#
    if False:
        #obliquity = 23.439291111111835*deg_to_rad
        #ecc = 0.01670236225492288
        #pre = 102.91794451250462*deg_to_rad
        a = astro.AstroLaskar2004()
        solarC = 1365
        t = 0
        plot_inso_with_min_max(solarC,a.eccentricity(t),a.obliquity(t),a.precession_angle(t))
        plt.show()

#
# plot the summary figure for Mars's present-day (J2000) astronomical parameters
#
    if True:
        a = astro.AstroLaskarMars2004()
        mars_a = 1.523679335            # semi-major axis in A.U.
        solarC = 1365/(mars_a*mars_a)   # corresponding solar constant
        t = -10                         # 10 kyrBP
        plot_inso_with_min_max(solarC,a.eccentricity(t),a.obliquity(t),a.precession_angle(t))
        plt.title("Mars insolation at 10 kyrBP (in W/m2)")
        plt.show()


