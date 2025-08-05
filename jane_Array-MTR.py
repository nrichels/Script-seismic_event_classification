import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from obspy import *
from obspy.core.inventory.inventory import Inventory
from obspy.core import AttribDict

import matplotlib.dates as mdates
import matplotlib.cm as cm


from obspy.clients.fdsn import Client
from obspy.signal.invsim import cosine_taper
import obspy_arraytools as AA
import os
import sys



client = Client("http://tarzan")
arraystats = ["XG.UP1..GLZ","XG.UP2..GLZ","XG.UP3..GLZ","XG.UP4..GLZ","XG.UP5..GLZ","XG.UP6..GLZ"]


ts= t = UTCDateTime("2024-03-22T05:00:00")
e = UTCDateTime("2024-03-22T08:00:00")
output_path = "./Grenzgletscher_fk"
figure_path = "./Grenzgletscher_fk/figure"
fl=1
fh=20.00
win_len=2.0 
win_frac=0.1
sll_x=-0.5
slm_x=0.5
sll_y=-0.5
slm_y=0.5
sl_s=0.025
thres_rel = 0.5

while (ts+3600) < e:
    start = (ts)
    end = (ts+3600)
    ts += 3600
    try:
        sz = Stream()
        inv= Inventory()
        i = 0
        for station in arraystats:
            net,stat,loc,chan=station.split('.')
            tr = client.get_waveforms(network=net,station=stat,location=loc,channel=chan, starttime=start, endtime=end)
            ii = client.get_stations(network=net,station=stat,location='',channel=chan, starttime=start, endtime=end,level="response")
            print(tr)
            sz += tr
            inv += ii
        sz.merge()
        sz.detrend("linear")
        sz.attach_response(inv)
        vc = sz.select(component="Z")
        array = AA.SeismicArray("",inv)
        array.inventory_cull(vc)
        print(array.center_of_gravity)
        outray = 0. 
        outray = array.fk_analysis(vc, frqlow=fl, frqhigh=fh, prefilter=True,\
                         static3d=False, array_response=False,vel_corr=4.8, wlen=win_len,\
                         wfrac=win_frac,sec_km=True,
                         slx=(sll_x,slm_x),sly=(sll_y,slm_y),
                         sls=sl_s)

        trace1 = Trace(data=outray.max_rel_power)
        trace1.stats.channel = 'REL'
        out = outray.max_rel_power

        trace2 = Trace(data=outray.max_abs_power)
        trace2.stats.channel = 'ABS'
        out = np.vstack([out,outray.max_abs_power])

        trace3 = Trace(data=outray.max_pow_baz)
        trace3.stats.channel = 'BACK'
        out = np.vstack([out,outray.max_pow_baz])

        trace4 = Trace(data=outray.max_pow_slow)
        trace4.stats.channel = 'SLOW'
        out = np.vstack([out,outray.max_pow_slow])

        #saving f-k analysis results into mseed file
        fk = Stream()
        tr = Trace()

        delta = outray.timestep

        tr.stats.network = outray.inventory.networks[0].code
        tr.stats.station = outray.inventory.networks[0][0].code
        tr.stats.channel = "ZGC"
        tr.stats.location = ""
        tr.data = outray.max_rel_power
        tr.stats.starttime = outray.starttime
        tr.stats.delta = delta

        fk += tr

        tr = Trace()
        tr.stats.network = outray.inventory.networks[0].code
        tr.stats.station = outray.inventory.networks[0][0].code
        tr.stats.channel = "ZGI"
        tr.stats.location = ""
        tr.stats.starttime = outray.starttime
        tr.data = outray.max_abs_power
        tr.stats.delta = delta

        fk += tr

        tr = Trace()
        tr.stats.network = outray.inventory.networks[0].code
        tr.stats.station = outray.inventory.networks[0][0].code
        tr.stats.channel = "ZGS"
        tr.stats.location = ""
        tr.stats.starttime = outray.starttime
        tr.data = outray.max_pow_baz
        tr.stats.delta = delta

        fk += tr

        tr = Trace()
        tr.stats.network = outray.inventory.networks[0].code
        tr.stats.station = outray.inventory.networks[0][0].code
        tr.stats.channel = "ZGA"
        tr.stats.location = ""
        tr.stats.starttime = outray.starttime
        tr.data = outray.max_pow_slow
        tr.stats.delta = delta

        fk += tr

        myday = "%03d"%fk[0].stats.starttime.julday

        pathyear = str(fk[0].stats.starttime.year)
        # open catalog file in read and write mode in case we are continuing d/l,
        # so we can append to the file
        mydatapath = os.path.join(output_path, pathyear)

        # create datapath 
        if not os.path.exists(mydatapath):
            os.mkdir(mydatapath)

        mydatapath = os.path.join(mydatapath, fk[0].stats.network)
        if not os.path.exists(mydatapath):
            os.mkdir(mydatapath)

        mydatapath = os.path.join(mydatapath, fk[0].stats.station)

        # create datapath 
        if not os.path.exists(mydatapath):
                os.mkdir(mydatapath)


        for tr in fk:
            print("saving to " + mydatapath)
            print(tr)
            mydatapathchannel = os.path.join(mydatapath,tr.stats.channel + ".D")

            if not os.path.exists(mydatapathchannel):
                os.mkdir(mydatapathchannel)

            netFile = tr.stats.network + "." + tr.stats.station +  "." + tr.stats.location + "." + tr.stats.channel+ ".D." + pathyear + "." + myday
            netFileout = os.path.join(mydatapathchannel, netFile)

            # try to open File
            print(netFileout)
            try:
                netFileout = open(netFileout, 'ab')
            except:
                netFileout = open(netFileout, 'w')
            tr.write(netFileout , format='MSEED',encoding="FLOAT64")
            netFileout.close()

        #print(outray)

        # Plot FK
        labels = ['ref','rel.power', 'abs.power', 'baz', 'slow']
        xlocator = mdates.AutoDateLocator()
        fig = plt.figure()
        alphas = out[0,:]
        condition1 = (out[0,:] < thres_rel)
        condition2 = (out[3,:] > 0.4) 
        tt = np.ma.masked_array(fk[0].times("matplotlib"),mask=condition1)
        tt = np.ma.masked_array(tt,mask=condition2)
        axis = []

        for i, lab in enumerate(labels):
            try:
                if i == 0:
                    ax = fig.add_subplot(5, 1, i + 1,sharex=None)
                    ax.plot(vc[0].times("matplotlib"),vc[0].data)
                else:
                    ax = fig.add_subplot(5, 1, i + 1,sharex=axis[0])
                    mask_v = np.ma.masked_array(out[i-1,:],mask=condition1)
                    mask_v = np.ma.masked_array(mask_v,mask=condition2)
                    ax.scatter(tt,mask_v, c=out[0,:], alpha=alphas,
                       edgecolors='none', cmap=cm.viridis_r)
                    ax.set_ylabel(lab)
                    ax.set_ylim(mask_v.min()-0.1, mask_v.max()+0.1)
                    ax.xaxis.set_major_locator(xlocator)
                    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
                axis.append(ax)
            except Exception as er:
                sys.stderr.write("Error:" + str(er))
                traceback.print_exc()
        fig.suptitle( 'jane-fk %s' % ( start ))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
        plt.savefig("%s/FK-%s.png"%(figure_path,start.strftime('%Y-%m-%dT%H')))
        #plt.show()
        plt.close("all")
    except:
        continue