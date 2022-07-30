1I_2017U1_lightcurve.csv
Photometry used for Belton et al paper.


!!CONTACT meech@ifa.hawaii.edu AND ohainaut@eso.org BEFORE PUBLISHING ANYTHING BASED ON THESE DATA.



File format:
- - - 
texUT,texMJD,texcorrMMJD,texrAU,texdAU,texaPh,Mag,texcorrm,sigma,Filt,flag
2017-10-25T01:04:16,58051.045,51.045,1.362,0.398,19.31,22.012,22.012,0.0814,g,V1g
2017-10-25T01:05:42,58051.046,51.046,1.362,0.398,19.31,22.115,22.115,0.0875,g,V1g
...

Columns:
- - - 
textUT: epoch of the measurement
MJD: 	modified julian date of the measurement
MMJD: 	MJD - 58000 - travel time, normalized to distance of the first point
rAU, dAU: helio- and geo-centric distances, AU.
aPh: 	solar phase angle
Mag: 	measured magnitude
corrm: 	magnitude, corrected for geo- and helio-centric distance (normalized to the distance of the first point),
	corrected for solar phase (using 0.04mag/deg, normalized to the phase of the first point);
	corrected for colour to g-filter using colours in Meech+17  
sigma: 	error on measured magnitude
Filt: 	filter
Flag: 	a code identifying the run and filter
    Our data:
        'V1g', 'V1r','V1i','V1zAv',  'V2gAv', 'V2rAv','V2iAv','V2zAv',           # VLT
        'G2g', 'G2rAv','G2i','G2zAv',  'G3gAv', 'G3rAv','G3iAv','G3zAv', 'G3y',  # Gemini,
        'c3w',                  # CFHT
        'U3z', 'U3y' ,          # UKIRT
        'Kbr', 'Kbi',           # KECK
        'H21V','H21Vgood',      # HST 2017-Nov-21
        'c22w3Av','c23w3Av'     # CFHT 2017-Nov-22 + 23
    Other's published data:
        'Wy1R',                      # Jewitt Wyin
        'GN3r','GN4r',               # T. Gregory Guzik/Nasza GN 
        'GN5g', 'GN5r', 'GN5J',      # Bannister GeminiN
        'WHT5g', 'WHT5r', 'WHT5i',   # Bannister WHT
        'APO',                       # Bolin APO,
        'D6r',                       # Knight DCT 
        'No1B', 'No1V', 'No1R',      # Jewitt NOT1
        'No2R',                      # Jewitt NOT2 
        'M21w',                      # Magellan Nov.21 

Note that in some cases, the individual data points are listed,
and are interleaved with averaged points (which are typically marked with Av),
for instance, V2gAv are the averaged points from the V2g series.
One should keep either the Av points and comment out the individual points, or
keep the individual points and comment out their Av.

The list above has all the runs that were used in our Belton paper.

