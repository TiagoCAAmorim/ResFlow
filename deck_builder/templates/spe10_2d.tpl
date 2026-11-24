** Synthetic 2d waterflood model: SPE10 permeability patch, {n_prod}P{n_inj}I, {schedule} BHP schedule
** case {case_id}
TITLE1 '2D WATERFLOOD {n_prod}P{n_inj}I'

XDR ON
INUNIT  MODSI
OUTUNIT  MODSI
WPRN WELL 0
WPRN GRID 0
WPRN ITER NONE
OUTPRN WELL LAYER
OUTPRN RES NONE
OUTPRN GRID NONE
WSRF GRID TIME
WSRF WELL TIME
OUTSRF GRID PERMI PERMJ PERMK PRES RS SG SO
OUTSRF WELL LAYER ALL
OUTSRF RES ALL
OUTSRF SPECIAL SIMPERF
WRST FREQ-TIME 0

GRID
CORNER {ni} {nj} {nk}
INCLUDE '{corners_file}'

NULL CON 1
POR CON 0.10
PERMI ALL
INCLUDE '{perm_file}'
PERMJ ALL
INCLUDE '{perm_file}'
PERMK CON 0.1

PINCHOUTARRAY CON 1
CPOR 56.000E-06
PRPOR 450
PVCUTOFF 200

INCLUDE '{pvt_file}'
PTYPE CON            1

ROCKFLUID
INCLUDE '{kr_file}'

INITIAL
VERTICAL BLOCK_CENTER WATER_OIL

REFDEPTH 5452.50
REFPRES 450
DWOC 9999

DATUMDEPTH 5400 INITIAL
PB CON        150.0

NUMERICAL
NORM PRESS 5.0
NORM SATUR 0.01

DTMAX  {dtmax}
DTMIN  0.001

CONVERGE PRESS  1.00
CONVERGE SATUR 0.001

RUN
DATE {start_date}
GROUP 'PROD' ATTACHTO 'FIELD'
GROUP 'INJW' ATTACHTO 'FIELD'

** One block per producer (P1, P2, ...)
#for producers
WELL  '{well}' ATTACHTO 'PROD'
PRODUCER '{well}'
OPERATE  MIN  BHP  {bhp_limit}  CONT REPEAT
GEOMETRY  K  0.0762  0.37  1.0  0.0
    PERF      GEOA  '{well}'
    {i} {j} 1:{nk}   1.0  OPEN FLOW-TO  'SURFACE'

#end
** One block per injector (I1, I2, ...)
#for injectors
WELL  '{well}' ATTACHTO 'INJW'
INJECTOR MOBWEIGHT '{well}'
INCOMP WATER
OPERATE  MAX  BHP  {bhp_limit}  CONT REPEAT
GEOMETRY  K  0.0762  0.37  1.0  0.0
    PERF      GEOA  '{well}'
    {i} {j} 1:{nk}   1.0  OPEN FLOW-TO  'SURFACE'

#end
BHPDEPTH '*' 5452.50

SHUTIN '*'

** Report dates; the wells listed under a date get a new BHP target on it
#for dates
DATE {date}
#for controls
TARGET BHP '{well}'
  {bhp}
#end
#end
DATE {end_date}
STOP
