/*     code to calculate neutrino overdensities around DM halos  */

/*     input parameters (given on command line) are halo mass in units of 10^12 Msun; observed redshift zo; and concentration parameter c  the neutrino overdensity is provided at 20 different radii to the command line   */

/*     the output is the radial fractional-neutrino density profile   */
/*     the mu-p integrands (i.e., the neutrino phase-space distributions) are also stored in the file integrand.out  */
       

/***************************************************************************/
/*   These include assorted utilities  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Recipes/nrutil.c"
#include "Recipes/nrutil.h"

/*  These are old numerical routines from Numerical Recipes   */
#include "Recipes/splint.c"    /* sets up interpolation table  */
#include "Recipes/spline.c"   /* finds interpolated values  */
#include "Recipes/odeint.c"   /* ode solver  */
#include "Recipes/rk4.c"      /*  ode stepper called by rkqs */
#include "Recipes/rkqc.c"    /* variable step size stepper */
#include "Recipes/gauleg.c"  /* calculate gauss-legendre weights */



#define sqr(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))
#define PI (3.141592653589793)

/***************************************************************************/
/*   These are variables used by multiple subroutines  */
static double zo,zi;           /*  observed and initial redshifts   */
static double M12,c;  /*  mass in units of 10^12 Msun and concentration parameter  */
static double Ri;   /*  comoving perturbation radius in Mpc   */
static double r200;     /*    physical r200 in Mpc   */
static double rs;       /*   NFW scale radius in Mpc   */
static double *xarray,*yarray,*x2array; /* interpolation arrays to determine integrations weights */
static int Nxysize;        /* number of points in interpolation table */
static double ymax;




/*************************************************************************/
#define RN_POINTS 100

// Function to generate logarithmically spaced points
void generate_logspace(double start, double end, double *radii, int num_points) {
    double log_start = log10(start);
    double log_end = log10(end);
    double step = (log_end - log_start) / (num_points - 1);

    for (int i = 0; i < num_points; i++) {
        radii[i] = pow(10, log_start + i * step);
    }
}

// Function to setup radii array
void setup_radii(double rs, double r200, double Ri, double *radii, int Nradii) {
    // Assuming vector(1, Nradii) allocates memory for radii, ensure Nradii >= RN_POINTS
    if (Nradii < RN_POINTS) {
        // Handle error or resize vector if needed
        // Example: Nradii = RN_POINTS;
    }

    // Generate 100 logarithmically spaced points between radii[1] and radii[11]
    generate_logspace(rs / 10, 3 * Ri, radii, RN_POINTS);
}




/***************************************************************************/
/*   This is where the action is  */

int main(int argc,char *argv[])
{
  double E(double),dPhidr(double,double),I(double),Phi(double,double);
  double *y,*dydt;    /*    phase-space variables  */
  //double *radii;       /*  vector of radii to evaluate density at  */
  double r,v,ell;      /*  comoving radius, comoving velocity  */
  double mnu;          /*  neutrino mass in eV  */
  double a;       /* smallest mu for r>Ri  */
  
  double z,zstep;    /*  z and deltaz for testing trajectories  */
  double vxinitial,vyinitial;     /*  initial comoving velocity  */
  /* these headers look complicated but you don't have to pay attention  */
  void spline(double x[],double y[], int n, double yp1, double ypn,double y2[]);
  void odeint(double ystart[], int nvar, double x1, double x2, double eps, double h1, double hmin, int *nok, int *nbad, void (*derivs)(double, double *, double *), void (*rkqc)(double *,double *,int,double *,double,double,double *,double *,double *,void (*)()));
  void rkqc(double y[], double dydx[], int n, double *x, double htry, double eps, double yscal[] ,double *hdid, double *hnext,void (*derivs)(double,double *,double *));
  void gauleg(double x1,double x2,double x[],double w[], int n);
  void allderivs(double z, double *y, double *dydt);

  int nok,nbad;
  //int Nradii=11;    /* number of radii to evaluate density at  */
  int Npts;     /* number of integration time steps   */
  double *glegx,*glegw;
  double x,mu;        /*    scaled velocity and cosine of velocity direction  */
  double density;    /*   neutrino density  */
  double xtov;      /*  scales velocity to exponent in FD distribution  */
  double vescape;   /*  escape velocity  */
  double vmax;
  int Nintegrand=100;    /*   number of points in v integration  */
  double xxx(double);     /*  determines values of x in v integrand */
  double stretch=5.0;    /*  quantity in determining x integration values  */
  double xmax;         /* upper limit of momentum (divided by Tnu) integrand  */
  
  int nlegroots=5;     /*  number of points for Gauss-Legendre integration over angles  */
  int i,j,ir;      /*   counters  for loops  */

  FILE *outfile;    /* file to store integrands  */

  y=vector(1,4);      /*   number of perturbation variables  */
  dydt=vector(1,4);      /*   number of perturbation variables  */
  //radii=vector(1,Nradii);   /*  vector of radii to evaluate density at  */

  glegx=vector(1,nlegroots);     /*    GL roots  */
  glegw=vector(1,nlegroots);     /*    GL weights  */

  
  /*  obtain weights and roots for Gauss-Legendre integrations  */
  gauleg(-1.0,1.0,glegx,glegw,nlegroots);
  
  /************************************************************************/
  /*    check roots and weights
  for(i=1; i<= nlegroots; i++) {
	printf("%d  %1.3e  %1.3e\n",i,glegx[i],glegw[i]);
  }
  exit(1);
  */

  M12=atof(argv[1]);     /*  halo mass in units of 10^{12} Msun  */
  zo=atof(argv[2]);      /*  observation redshift  */
  c=atof(argv[3]);       /*   concentration parameter  */
  mnu=atof(argv[4]);     /*   neutrino mass in eV  */


  zi = pow(200.0,1.0/3.0)*(1.0+zo) - 1.0;     /*   initial redshift  */
  Ri= 1.77 * pow(M12,1.0/3.0);   /* comoving perturbation radius in Mpc  */
  r200 = Ri/(1.0+zi);            /*  r200 aka r_vir   in Mpc  */
  rs=r200/c;                     /*   scale radius in Mpc  */


// Print the values
   printf("zi = %f\n", zi);
   printf("Ri = %f\n", Ri);
   printf("r200 = %f\n", r200);
   printf("rs = %f\n", rs);




  // Add new code to setup radii
  int Nradii = 100;  // Example Nradii value, ensure it's properly set
  double *radii = vector(1, Nradii);  // Allocate memory for radii

  // Set up the radii
  setup_radii(rs, r200, Ri, radii, Nradii);

  // Array to hold the radii values
  //double radii[RN_POINTS];

  // Set up the radii
  //setup_radii(rs, r200, Ri, radii);

  // radii[1]=rs/10.0;  radii[2]=rs/3.0;  radii[3]=2.0*rs/3.0; radii[4]=rs;  radii[5]=rs+(r200-rs)/3.0;
  // radii[6]=rs+2.0*(r200-rs)/3.0;  radii[7]=r200;  radii[8]= 2.0*r200; radii[9]=3.0*r200; radii[10]=Ri;  radii[11]=3.0*Ri;
  

  /************************************************************************/
  /*  here's a sanity check on parameters
      printf("zi=%1.3e  R=%1.3e  r200=%1.3e  I(c)=%1.3e\n",zi,Ri,r200,I(c));
  */
  

  /*    test acceleration profile
  outfile=fopen("acceleration.out","w");
  for(r=0.0; r<=Ri/100.0; r+=Ri/10000.0) {
    fprintf(outfile,"%1.3e  %1.3e  %1.3e  %1.3e\n",r,dPhidr(r,zo),dPhidr(r,(zi+zo)/2.0),dPhidr(r,zi));
  }
  fclose(outfile);
  exit(1);
  */
  /************************************************************************/
  


  /************************************************************************/
  /*    test potential profile   
  outfile=fopen("potential.out","w");
  for(r=0.0; r<=Ri*1.1; r+=Ri/100.0) {
    fprintf(outfile,"%1.3e  %1.3e  %1.3e  %1.3e\n",r,Phi(r,zo),Phi(r,(zi+zo)/2.0),Phi(r,zi));
  }
  fclose(outfile);
  exit(1);
  */

  /*  check the calculation of vinitial near the profile center
  for(vyinitial=100.0;vyinitial<=3000.0; vyinitial+=100.0) {
    y[1]= 0.000001;
    y[2] = 0.0;
    y[3] = 0.0;
    y[4] = vyinitial;
    nok=1;
    odeint(y,4,zo,zi,1.0e-6,(zo-zi)/100.0,0.0,&nok,&nbad,allderivs,rkqs);
    printf("%1.3e  %1.3e\n",vyinitial,sqrt( sqr(y[2]) +sqr(y[4])));
  }
  exit(1);
  */
  /************************************************************************/

  /************************************************************************/
  /*   test the trajectories
  outfile = fopen("mapping.out","w");
  xtov = 504.3*(1.0+zo)/(mnu/0.1);   
  x=1.0;
  r=0.212;
  mu = 0.5;
  y[1]= r;
  y[2]=mu*x*xtov;
  y[3]=0.0;
  y[4]=sqrt(1-mu*mu)*x*xtov;
  nok=1;    
  z=zo;  zstep=(zi-zo)/1000.0;      
  for(z=zo;z<=zi;z+=zstep) {
    odeint(y,4,z,z+zstep,1.0e-5,zstep/10.0,0.0,&nok,&nbad,allderivs,rkqc);
    r=sqrt( sqr(y[1])+sqr(y[3]));
    v = sqrt( sqr(y[2]) +sqr(y[4]));
    ell=  y[1]*y[4] - y[2] * y[3];
    fprintf(outfile,"%1.3e  %1.3e  %1.3e  %1.3e  %1.3e   %1.3e  %1.3e  %1.3e  %1.3e\n",z,y[1],y[2],y[3],y[4], r, sqrt(sqr(y[1])+sqr(y[3])),v, ell );
  }
  fclose(outfile);
  exit(1);
  */
  /************************************************************************/

  /************************************************************************/
  /*   test initial->final velocity mapping at center
  outfile = fopen("mapping.out","w");
  printf("vescape=%e  vthermal=%e\n",sqrt(-Phi(0.0001,0)),504.0*(1.0+zo)/(mnu/0.1));
    for(vyinitial=100.0;vyinitial<=3000.0; vyinitial+=10.0) {
    y[1]= 0.000001;
    y[2] = 0.0;
    y[3] = 0.0;
    y[4] = vyinitial;
    nok=1;
    odeint(y,4,zo,zi,1.0e-6,(zo-zi)/100.0,0.0,&nok,&nbad,allderivs,rkqc);
    fprintf(outfile,"%1.3e  %1.3e\n",vyinitial,sqrt( sqr(y[2]) +sqr(y[4])));
  }
  */

  printf("vthermal=%e\n",504.0*(1.0+zo)/(mnu/0.1));
  /************************************************************************/


  outfile=fopen("integrand.out","w");
  /*  calculate neutrino density at radii[i]  */
  xtov = 504.0*(1.0+zo)/(mnu/0.1);   
  for(ir=1;ir<=Nradii;ir++) {     /*  loop over all radii  */
    r = radii[ir];
    density=0.0;   /* initial the neutrino density to 0.0  */
    /*  figure out the upper limit for the velocity (momentum) integral */
    if( 2.0*sqrt(-Phi(r,zo)) > 12.0 * xtov) {
      vmax = 2.0*sqrt(-Phi(r,zo));
    }
    else {
      vmax = 12.0*xtov;
    }
    xmax=vmax/xtov;
    /* get roots and weights for x integral  */
    /* first set up the interpolation table  */
    Nxysize=Nintegrand*10;
    xarray=vector(1,Nxysize);
    x2array=vector(1,Nxysize);
    yarray=vector(1,Nxysize);
    xarray=vector(1,Nxysize);
    x2array=vector(1,Nxysize);
    yarray=vector(1,Nxysize);
    /*  set up the interpolation table to find points for velocity (x) integral  */
    for(i=1;i<=Nxysize;i++) {
      x=((double)i)/((double)Nxysize)*xmax;
      xarray[i]=x;
      yarray[i]=xmax /cube(stretch) * (2.0*sqr(xmax) - exp(-stretch*x/xmax)*(sqr(stretch)*x*x + 2.0* stretch*x*xmax +2.0*sqr(xmax)));
    }
    spline(yarray,xarray,Nxysize,1.0e40,1.0e40,x2array);    /*  calculates derivatives for cubic spline interpolation  */
    ymax=yarray[Nxysize];
    for(i=1;i<=Nintegrand;i++) {     /*   sum over momenta  */
      x=xxx( ((double)i-0.5)/((double)Nintegrand)*ymax);
      if(r<Ri) {     /* for r<Ri we integrate over all angles  */
	for(j=1;j<=nlegroots;j++) {     /* sum over angles  */
	  mu=glegx[j];
	  y[1]=r;
	  y[2]=mu*x*xtov;
	  y[3]=0.0;
	  y[4]=sqrt(1-mu*mu)*x*xtov;
	  odeint(y,4,zo,zi,1.0e-5,(zo-zi)/100.0,0.0,&nok,&nbad,allderivs,rkqc);
	  v = sqrt( sqr(y[2]) +sqr(y[4]));
	  density += (ymax/(double)Nintegrand)*exp(stretch*x/xmax)/ (1.0 + exp( v/xtov))*glegw[j];
	  fprintf(outfile,"%e  %e  %e  %e  %e  %e\n",r,mu,sqrt(-Phi(r,zo)),x,v,(ymax/(double)Nintegrand)*exp(stretch*x/xmax)/ (1.0 + exp( v/xtov))*glegw[j]);
	}
      }
      else {   /*  for r>Ri we integrate over a smaller range of angles  */
	a=sqrt(1.0-sqr(Ri/r));
	for(j=1;j<=nlegroots;j++) {   /* sum over angles  */
	  mu=(1.0-a)/2.0*glegx[j] + (1.0+a)/2.0;
	  y[1]=r;
	  y[2]=mu*x*xtov;
	  y[3]=0.0;
	  y[4]=sqrt(1-mu*mu)*x*xtov;
	  odeint(y,4,zo,zi,1.0e-5,(zo-zi)/100.0,0.0,&nok,&nbad,allderivs,rkqc);
	  v = sqrt( sqr(y[2]) +sqr(y[4]));
	  density += (1.0-a)/2.0*(ymax/(double)Nintegrand)*exp(stretch*x/xmax)/ (1.0 + exp( v/xtov))*glegw[j];
	  fprintf(outfile,"%e  %e  %e  %e  %e  %e\n",r,mu,sqrt(-Phi(r,zo)),x,v,(ymax/(double)Nintegrand)*exp(stretch*x/xmax)/ (1.0 + exp( v/xtov))*glegw[j]);
	}
      }
    }
    if(r<Ri) {
      density = density/1.80309/2.0;  /* this gives the density in units of the homogeneous value  */
    }
    else {
      density = (density+(1.0+a)*1.80309)/1.80309/2.0;  /* this gives the normalized density including the contribution from trajectories that don't hit the perturbation  */
    }
  printf("%e  %e\n",r,density);

  /*  free vector to be used again for the next radius   */
  free_vector(xarray,1,Nxysize);
  free_vector(x2array,1,Nxysize);
  free_vector(yarray,1,Nxysize);
  }

  free_vector(y,1,4);      /*   number of perturbation variables  */
  free_vector(dydt,1,4);      /*   number of perturbation variables  */
  free_vector(radii,1,Nradii);

  free_vector(glegx,1,nlegroots);
  free_vector(glegw,1,nlegroots);

}



/*   The expansion rate divided by H0 */
double E(double z)
{
  double Omegam=0.315;     /*  NR matter density today in units of critical  */

  return sqrt( (Omegam)*pow(1.0+z,3.0)+(1.0-Omegam));
}


/*    differential equations for particle trajectory   */
void allderivs(double z, double *y, double *dydt)
{
  /*   y[1] = x   y[2] = v_x    y[3] =  y   y[4] = v_y    */

  double H0=68.0;           /* Hubble parameter in units of km/sec/Mpc  */
  double E(double),dPhidr(double,double);
  double H=H0*E(z);
  double r=sqrt(sqr(y[1])+sqr(y[3]));

  dydt[1] = -y[2]*(1.0+z)/H;
  dydt[2] = 1.0/(1.0+z)/H * dPhidr(r,z)* y[1]/r;
  dydt[3] = -y[4]*(1.0+z)/H;
  dydt[4] = 1.0/(1.0+z)/H * dPhidr(r,z)* y[3]/r;

}


/*   time dependence of halo growth  */
double xi(double z)
{
  if(z<zi) {
    return (zi-z)/(zi-zo);
  }
  else {
    return 0.0;
  }
  
}


/*     radial mass profile for NFW halo  */
double I(double c)
{
  return log(1.0+c) -c/(1.0+c);
}



/*    acceleration in the gravitational potential  */
/*    r here is the comoving radius  */
double dPhidr(double r,double z)
{
  double  GM = 4.375e3 * M12;    /* G times 10^{12} Msun in (km/sec)^2 Mpc  */

  double answer;
  double xi(double),I(double);

  r=r+rs/10000.0;  /*  soften the singularity to avoid numerical issues  */

  answer = - GM * (1.0+z) * xi(z) / sqr(r);    

  if(r<r200*(1.0+z)) {
    answer *= (I(r/rs/(1.0+z))/I(c) - cube(r/Ri) );
  }
  else {
    if(r<Ri) {
      answer *= (1.0-cube(r/Ri));
    }
    else {
      answer = 0.0;
    }
  }

  return -answer;
}


/*   the gravitational potential  */
/*    r here is the comoving radius  */
double Phi(double r,double z)
{
  double  GM = 4.375e3 * M12;    /* G times 10^{12} Msun in (km/sec)^2 Mpc  */
  double answer;
  double xi(double),I(double);

  r=r+rs/100.0;  /*  soften the singularity to avoid numerics  */

  if(r<r200*(1.0+z)) {
    answer = -GM*(1.0+z)*xi(z)*(1.0/ r /I(c)*log(1.0+r/rs/(1.0+z))
				-1.0/(r200*(1.0+z))/I(c)*log(1.0+c)
				+1.0/(r200*(1.0+z)))
      +  GM * xi(z)*(1.0+z)*(3.0*sqr(Ri)-sqr(r))/2.0/cube(Ri);
  }
  else {
    if(r<Ri) {
      answer = -GM * xi(z)* (1.0+z)/r
	+  GM* xi(z)*(1.0+z)*(3.0*sqr(Ri)-sqr(r))/2.0/cube(Ri);

    }
    else {
      answer=0.0;
    }
  }
  return answer;
}

double xxx(double yyy)    /* give the inverse of (x^2-2x-2)e^(-x) +2  */
{
  double x;
  void splint(double xa[], double ya[], double y2a[], int n,double x, double *y);

  if(yyy<ymax) {
    splint(yarray,xarray,x2array,Nxysize,yyy,&x);
  }
  else {
    x=xarray[Nxysize];
  }

  return x;
}
