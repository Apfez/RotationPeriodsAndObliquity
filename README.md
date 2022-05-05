# RotationPeriodsAndObliquity

Finds photometric rotation periods from TESS light curves. Uses SPOC light curves per default, but if no such light curve is available, uses TESS-SPOC and QLP Full-Frame-Images instead.
If a projected rotation speed, radius and potentially a projected obliquity is known for the system, uses these to calculate a posterior probability of the stellar obliquity.

Light curve:
![WASP-7v1](https://user-images.githubusercontent.com/63327679/166900617-9c8abd7b-cc95-4798-8dd1-4c7278ce9da6.png)

Top left: Light curve and binned light curve, colored according to found rotation period. Top right: Small overview of the appended sectors and masked transits.
Middle: Auto-Correlation function of the light curve. Highest peak within maximum rotation period (given vsini and radius of star) is chosen, see McQuillan et al 2014 (https://arxiv.org/abs/1402.5694). The ACF is smoothed with a gaussian kernel to remove very high frequency noise. An estimate of this kernel width is performed, but sometimes a manual adjustment of this value is necessary. Orbital period of potential orbiting planets as well as an "expected" rotation period (only available in the effective temperature range 5900K-6300K, using the relation from Louden et al 2021 (https://arxiv.org/abs/2012.00776)) is shown here too. 
Bottom left: Determination of the final period as well as uncertainty from the first chosen period as well as subsequent periods integers hereof. Bottom right: The entire light curve folded with the determined rotation period. 
![WASP-7v2](https://user-images.githubusercontent.com/63327679/166900639-7db43383-695f-4bfa-a32e-9ba5922e5296.png)

Top left: Rotation period and uncertainty as per the first plot. Top middle: Stellar radius (R) and uncertainty. Top right: Rotation speed projected along the line-of-sight (vsini) as well the inferred rotation speed from R and P. 
Middle: Inferred probability density of the stellar inclination i_s from vsini and v as per Masuda and Winn 2020 in red. Orbital inclination in blue and Projected obliquity in green from the literature. And the inferred probability density of the stellar obliquity colored in accordance with the colormap also used for the bottom figures.
Bottom: 3D view of the stellar obliquity probability. Values of the stellar obliquity are shown as contours emanating from the mean value of the axis of orbital angular momentum, n_o. All relevant angles are shown in this plot too.
All parameters are taken from the NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/) and/or TEPCat (https://www.astro.keele.ac.uk/jkt/tepcat/obliquity.html)
