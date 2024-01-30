import numpy as np
import matplotlib.pyplot as plt
import your 
from astropy.time import Time
import astropy.units as u

def get_data(filfile, flo=None, fhi=None, tscale=1.0):
    # flo = 8715, fhi = 8790
    in_yr = your.Your(filfile)
    dt = in_yr.tsamp
    freqs = np.arange(in_yr.nchans) * in_yr.foff + in_yr.fch1
    tt = np.arange(in_yr.your_header.nspectra) * dt
    dat = in_yr.get_data(nstart=0, nsamp=in_yr.your_header.nspectra)

    # if freq range specified, use that
    if flo is not None and fhi is not None:
        xx = np.where( (freqs >= flo) & (freqs <= fhi) )[0]
        freqs = freqs[xx]   
        dat = dat[:, xx]
    else: pass

    # Possibly just a one-off fix 
    tt *= tscale

    return tt, freqs, dat


def read_obs_file(obsfile):
    """
    Read the obs file to get scan times
    """
    times_doy = []
    targets = []
    with open(obsfile) as fin:
        for line in fin:
            cols = line.split()
            if len(cols) != 6:
                continue
            else: pass

            times_doy.append(cols[0])
            targets.append(cols[5])

    times_doy = np.array(times_doy)
    targets   = np.array(targets)

    return times_doy, targets


def doy_str_to_time(tt_doy, year=2024):
    """
    Convert array of DOY:HH:MM:SS times to MJDs
    """
    # Get doy
    doys = np.array([ int(tt.split(':')[0]) for tt in tt_doy ])
    
    # Get time strings
    hms_time = np.array([ tt.split(':',1)[1] for tt in tt_doy ])

    # Get time object for jan 01 of year, at hms_times
    ystart = "%d-01-01" %year
    tt_jan1 = np.array([ "%sT%s" %(ystart, hms) for hms in hms_time ])

    tt1 = Time(tt_jan1, format='isot', scale='utc')
    tt = tt1 + (doys - 1) * u.day

    return tt


def calc_offsets(filfile, obsfile):
    """
    Read in obsfile and filfile header

    Calculate scan start times in seconds from start
    of filfile 
    """
    # Get mjd start of filfile
    in_yr = your.Your(filfile)
    mjd_start = in_yr.your_header.tstart
    tt_start = Time(mjd_start, format='mjd')
    year = tt_start.ymdhms[0]

    # get scan times / target names from obsfile 
    tt_doy, names = read_obs_file(obsfile)

    # Convert to astropy time objects
    tt_scans = doy_str_to_time(tt_doy, year=year)
    tt_start = Time(mjd_start, format='mjd')

    # Find offsets
    dts = (tt_scans - tt_start).sec

    return dts, names


def get_start_stops(tscans, scanlen, skipstart=30):
    """
    From tscans get (start, stop) for each scan.

    If scanlen > 0, then get that much data 

    Note: first ~30 sec or so seems to be not on source yet
    """ 
    tlast = tscans[-1] + scanlen + skipstart
    tstops = np.hstack( (tscans[1:], np.array([tlast])) )
    tstops -= 10
    tstarts = np.zeros( len(tstops) )
 
    for ii in range(len(tstarts)):
        tt = max( tstops[ii] - scanlen, tscans[ii] + skipstart )
        tstarts[ii] = tt

    return tstarts, tstops


def get_scans_and_times(filfile, obsfile, scanlen, skipstart=30):
    """
    Get names of scans and start stop times
    """
    tscans, names = calc_offsets(filfile, obsfile)
    tstarts, tstops = get_start_stops(tscans, scanlen, 
                                      skipstart=skipstart)
    return names, tstarts, tstops


def get_tindex(tt, tstart, tstop):
    """
    get indices for (tstart, tstop) range
    """
    xx = np.where( (tt >= tstart) & (tt <= tstop) )[0]
    return xx


def get_scan_data(tt, dd, tstart, tstop):
    """
    Get times and data
    """
    xx = get_tindex(tt, tstart, tstop)
    tt_out = tt[xx]
    dd_out = dd[xx]
    return tt_out, dd_out


def get_scan_spec(tt, dd, tstart, tstop):
    """
    get spectrum for scan defined by start stop
    """
    tts, dds = get_scan_data(tt, dd, tstart, tstop)
    spec = np.mean(dds, axis=0)
    return spec


def get_scan_spec_norm(tt, dat, scan_on, scan_off, 
                       tstarts, tstops):
    """
    Get normalized spectrum of source

    scan_on = index of on scan
    scan_off = index of off scan
    """
    start_on = tstarts[scan_on]
    stop_on  = tstops[scan_on]
    
    start_off = tstarts[scan_off]
    stop_off  = tstops[scan_off]

    spec_on  = get_scan_spec(tt, dat, start_on, stop_on)
    spec_off = get_scan_spec(tt, dat, start_off, stop_off)

    norm_spec = (spec_on - spec_off) / spec_off

    return norm_spec


def make_label_plot(tt, dat, names, tstarts, tstops):
    """
    Make a plot of avg power vs time and label 
    the scans 
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    # plot data
    dd = np.mean(dat, axis=1)
    ax.plot(tt, dd)
    
    tdurs = tstops - tstarts 
    tmids = 0.5 * (tstarts + tstops)

    # get avg power during scan
    pmids = np.zeros( len(tstarts) )
    for ii in range(len(tstarts)):
        tt_i, dd_i = get_scan_data(tt, dat, tstarts[ii], tstops[ii])
        pp_i = np.mean(dd_i)
        pmids[ii] = pp_i

    # set xlim
    ax.set_xlim(tstarts[0] - tdurs[0], tstops[-1] + tdurs[-1])

    # label scans
    ymin = 0
    ymax = pmids[-1] * 2
    ax.set_ylim(ymin, ymax)
    dy = 0.1 * (ymax - ymin)
    
    for ii, name in enumerate(names):
        offset = (1 + 0.75 * (ii//2%2)) * dy * (-1)**ii
        ax.text(tmids[ii], pmids[ii] + offset, name, ha='center')

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Power (arb)", fontsize=14)
        

    plt.show()
    return
 

def make_avg_plot(tt, dat, names, tstarts, tstops):
    """
    Make a plot of scan averaged power
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    # plot data
    dd = np.mean(dat, axis=1)
    
    tdurs = tstops - tstarts 
    tmids = 0.5 * (tstarts + tstops)

    # get avg power during scan
    pavgs = np.zeros( len(tstarts) )
    pstds = np.zeros( len(tstarts) )
    for ii in range(len(tstarts)):
        tt_i, dd_i = get_scan_data(tt, dat, tstarts[ii], tstops[ii])
        dd_i = np.mean(dd_i, axis=1)
        pp_i = np.mean(dd_i)
        pp_s = np.std(dd_i)
        pavgs[ii] = pp_i
        pstds[ii] = pp_s #/ np.sqrt( len(dd_i) )

    # set xlim
    ax.set_xlim(tstarts[0] - tdurs[0], tstops[-1] + tdurs[-1])

    # Plot values
    pnames  = ['3C286', '3C286OFF', '3C147', '3C147OFF']
    cols    = [ 'k', 'r', 'k' ,'r']
    markers = [ 'o', 'o', 's', 's' ] 

    for ii, ppn in enumerate(pnames):
        xx = np.where( names == ppn )[0]
        if 'OFF' in ppn:
            ax.plot(tmids[xx], pavgs[xx], marker=markers[ii], 
                    ms=10, ls='', color=cols[ii], label=ppn)
            ax.errorbar(tmids[xx], pavgs[xx], yerr=pstds[xx], 
                        marker='', color=cols[ii], ls='')
        else:
            ax.plot(tmids[xx], pavgs[xx], marker=markers[ii], 
                    ms=10, ls='', color=cols[ii], label=ppn)
            ax.errorbar(tmids[xx], pavgs[xx], yerr=pstds[xx], 
                        marker='', color=cols[ii], ls='')

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Power (arb)", fontsize=14)
    
    plt.legend(loc='center left')

    plt.show()
    return


def get_corr_flux(tt, dat, names, tstarts, tstops, Tsys=20):
    """ 
    Get corrected flux densities
    """
    tmids = 0.5 * (tstarts + tstops)

    has_OFF = np.array([ 'OFF' in nn for nn in names ])
    on_xx = np.where( has_OFF == False)[0]
    off_xx = np.where( has_OFF == True)[0]

    flux_avgs  = np.zeros(len(on_xx))
    flux_stds  = np.zeros(len(on_xx))
    flux_names = []
    flux_times = np.zeros(len(on_xx))

    for ii in range(len(on_xx)):
        scan_on  = on_xx[ii]
        scan_off = off_xx[ii]

        name_on  = names[on_xx[ii]]
        name_off = names[off_xx[ii]]

        if scan_off - scan_on != 1:
            print("On / Off Scans: %d, %d" %(scan_on, scan_off))
            print("Need off to follow on")

        norm_ii = get_scan_spec_norm(tt, dat, scan_on, scan_off, 
                                     tstarts, tstops)

        norm_ii *= Tsys

        flux_avgs[ii]  = np.mean(norm_ii)
        flux_stds[ii]  = np.std(norm_ii)
        flux_times[ii] = tmids[scan_on]
        flux_names.append(name_on)

    flux_names = np.array(flux_names)

    return flux_times, flux_names, flux_avgs, flux_stds


def make_avg_corr_plot(tt, dat, names, tstarts, tstops, Tsys=20):
    """
    Make a plot of scan averaged power
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    tdurs = tstops - tstarts 
    tmids = 0.5 * (tstarts + tstops)

    flux_times, flux_names, flux_avgs, flux_stds = \
          get_corr_flux(tt, dat, names, tstarts, tstops, Tsys=Tsys)

    unames = np.unique(flux_names)

    # Plot values
    markers = ['o', 's', 'v']
    for ii, sname in enumerate(unames):
        print(flux_names)
        print(sname)
        xx = np.where( flux_names == sname )[0]
        ax.plot(flux_times[xx], flux_avgs[xx], marker=markers[ii], 
                 c='k', ls='', ms=10, label=sname)
        ax.errorbar(flux_times[xx], flux_avgs[xx], yerr=flux_stds[xx], 
                    color='k', ls='')

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Flux Density (Jy)", fontsize=14)
    
    plt.legend(loc='upper right')

    plt.show()
    return


def make_avg_corr_pol_plot(tt, dat1, dat2, names, 
                           tstarts, tstops, Tsys=20):
    """
    Make a plot of scan averaged power for both pols
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    tdurs = tstops - tstarts 
    tmids = 0.5 * (tstarts + tstops)

    ftimes1, fnames1, favgs1, fstds1 = \
          get_corr_flux(tt, dat1, names, tstarts, tstops, Tsys=Tsys)
    
    ftimes2, fnames2, favgs2, fstds2 = \
          get_corr_flux(tt, dat2, names, tstarts, tstops, Tsys=Tsys)

    unames = np.unique(fnames1)

    # Plot values
    markers = ['o', 's', 'v']
    for ii, sname in enumerate(unames):
        xx1 = np.where( fnames1 == sname )[0]
        xx2 = np.where( fnames2 == sname )[0]
        
        ax.plot(ftimes1[xx1], favgs1[xx1], marker=markers[ii], 
                 c='k', ls='', label=sname + " Pol 1")
        ax.errorbar(ftimes1[xx1], favgs1[xx1], yerr=fstds1[xx1], 
                    color='k', ls='')
        
        ax.plot(ftimes2[xx2], favgs2[xx2], marker=markers[ii], 
                 c='r', ls='', label=sname + " Pol 2")
        ax.errorbar(ftimes2[xx2], favgs2[xx2], yerr=fstds2[xx2], 
                    color='r', ls='')

        print("%s Pol1:  " %sname, favgs1[xx1])
        print("%s Pol2:  " %(" " * len(sname)),  favgs2[xx2])


    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Flux Density (Jy)", fontsize=14)
    
    plt.legend(loc='upper right', fontsize=12)

    plt.show()
    return
