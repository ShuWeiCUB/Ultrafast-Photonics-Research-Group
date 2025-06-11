# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:29:44 2024

@author: ECEE1B79
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Set parameters 
plt.rcParams.update({
'font.family': 'Arial',  # Font family
'font.size': 11,          # Font size
'axes.labelweight': 'normal',  # Label weight
'axes.labelcolor': 'blue',   # Label color
'axes.labelsize': 11,         # Label size
'axes.titlesize': 11,         # Title size
'axes.titleweight': 'bold',   # Title weight
})
def mysuptitle(title):
    plt.suptitle(title)
    plt.tight_layout()
def myfig(options = 'Default',ratio = None): # Allowed figure size for plotting into manuscripts
    x = np.ones(3)
    y = x
    if options == 'Default': # Width of 1 coloumn in IEEE
        if ratio == None:
            ratio = 0.6
        fig,ax = plt.subplots(1,1,figsize = (4,4*ratio));
        axes = [ax]
        axes[0].set_xlabel('xlabel')
        axes[0].set_ylabel('ylabel')
        axes[0].set_title('title')
        
    
    if options == '3Vert': # Width of 1 coloumn in IEEE
        if ratio == None:
            ratio = 0.6
        fig,ax = plt.subplots(3,1,figsize = (3.5,(3.5*ratio)*3));
        axes = ax.flatten()
        for axs in axes:
            axs.set_xlabel('xlabel')
            axs.set_ylabel('ylabel')
            axs.set_title('title')
    if options == '3Vertx2': # Width of 2 coloumn in IEEE
        if ratio == None:
            ratio = 0.6
        fig,ax = plt.subplots(3,2,figsize = (3.5*2,(3.5*ratio)*3));
        axes = ax.flatten()
        for axs in axes:
            axs.set_xlabel('xlabel')
            axs.set_ylabel('ylabel')
            axs.set_title('title')
            
    if options == '2Vert': # Width of 1 coloumn in IEEE
        if ratio == None:
            ratio = 0.6    
        fig,ax = plt.subplots(2,1,figsize = (3.5,(3.5*ratio)*2));
        axes = ax.flatten()
        for axs in axes:
            axs.set_xlabel('xlabel')
            axs.set_ylabel('ylabel')
            axs.set_title('title')
            
    if options == '4Square': # Width of 1 coloumn in IEEE
        if ratio == None:
            ratio = 0.7
        fig,ax = plt.subplots(2,2,figsize = (5,(2.5*ratio)*2));
        axes = ax.flatten()
        for axs in axes:
            axs.set_xlabel('xlabel')
            axs.set_ylabel('ylabel')
            axs.set_title('title')
            
    if options == '4Square_DW': # Width of 1 coloumn in IEEE
        if ratio == None:
            ratio = 0.7
        fig,ax = plt.subplots(2,2,figsize = (10,(5*ratio)*2));
        axes = ax.flatten()
        for axs in axes:
            axs.set_xlabel('xlabel')
            axs.set_ylabel('ylabel')
            axs.set_title('title')
            
    if options == 'Square': # Width of 1 coloumn in IEEE
        if ratio == None:
            ratio = 0.5
        fig,ax = plt.subplots(1,1,figsize = (5,(5*ratio)));
        axes = ax
        
        axes.set_xlabel('xlabel')
        axes.set_ylabel('ylabel')
        axes.set_title('title')
        
    return fig, axes

def tickmaker(xy,xory):
    dxy = np.max(xy) - np.min(xy)
    if xory == 'x':
        ticks = np.linspace(np.min(xy),np.max(xy),3)/2
        
    elif xory == 'y':
        ticks = np.linspace(np.min(xy),np.max(xy),3)
    
    if dxy == 0:
        mag = 1
        ticks = np.array([-0.1,0,0.1])
    else:
        mag = int(np.log10(dxy))
        ticks = np.round(ticks,np.sign(mag)*mag+1)
    return ticks

def myplot(xx, yy = np.array([None]), y2 = np.array([None]), xlabel = 'xlabel',ylabel = 'ylabel',\
           title = 'title',linestyle = ['-'], color ='b', marker = None,markersize = 2,xticks = None,yticks=None,\
               border_color = None, ax = None,xl = None,yl = None,\
                   yscale = 'linear',xscale = 'linear',grid = True,alpha = 1,\
                       linewidth = 3,legend = False,label = [''],\
                           yaxcolor = 'k',yyax = False):
    if type(xx) == type(np.array([])):
        xx = [xx]
        yy = [yy]
        linestyle = (linestyle)
    if len(linestyle) != len(xx) and len(linestyle) == 1:
        linestyle = ['-','--','-.',':','-','--','-.']
        
    if len(color) != len(xx) and len(color) ==1:
        color = ['b','r','g','c','m','y','k']
    if len(label) != len(yy) and len(label)==1:
        label = ['','','','','','','','','','','','','','','','','','','','']
        
        
    if yyax:
        ax = ax.twinx()
        ax.spines['right'].set_color(yaxcolor)
        ax.tick_params(axis = 'y',colors = yaxcolor)
        ax.yaxis.label.set_color(yaxcolor)
        ax.spines['left'].set_visible(False)
        grid = False
    if marker != None:
        if len(marker) != len(xx):
            marker = ['o','s','*','x','D','plus','^']
    else:
        marker = [None,None,None,None,None,None,None,None,None,None,None,None,None,None]

    for x,y,i in zip(xx,yy,range(0,len(xx))):
        
        if xl == None:
            xLim = (x.min(),x.max())
        else:
            xLim = np.sort(xl)
        INXmin = np.argmin(np.abs(x-xLim[0]))    
        INXmax = np.argmin(np.abs(x-xLim[1]))+1
        
        xplot = x[INXmin:INXmax]
        yplot = y[INXmin:INXmax]
        
        if yl == None:
            yLim = (np.sign(y.min())*np.abs(y.min())*1.2,y.max()*1.2)
        else:
            yLim = yl
        
        if i == 0:
            if type(xticks ) != type(np.array([])):
                if xticks == None:
                    xticks = tickmaker(xLim,'x')
            if type(yticks ) != type(np.array([])):
                if yticks == None:
                    yticks = tickmaker(yLim,'y')
            # Plotting
        
        # Check if passed function handle
        if ax == None:
            fig,axs = myfig()
            ax = axs[0]
            
        ax.plot(xplot,yplot,linestyle = linestyle[i], \
                marker = marker[i], markersize = markersize, \
                color = color[i], linewidth = linewidth,alpha = alpha,label = label[i])
    if legend:
        leg = ax.legend(loc = 'best',framealpha=1,facecolor = 'white',edgecolor = 'black')
       
    # setting Axtetcis
    # x axis\
        
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xLim)
    ax.set_xticks(xticks)
    if not yyax:
        ax.spines['left'].set_color(yaxcolor)
        ax.tick_params(axis = 'y',colors = yaxcolor)
        ax.yaxis.label.set_color(yaxcolor)
        
        #
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_ylim(yLim)
    ax.set_yticks(yticks)
    # 
    ax.set_title(title)
    # print(xticks)
    ax.grid(visible = grid,which = 'both',axis = 'both')
    return ax


def mypcolor(x, y,z, y2 = np.array([None]), xlabel = 'xlabel',ylabel = 'ylabel',\
           title = 'title',linestyle = '-', color ='b', marker = None,xticks = None,yticks=None,\
               border_color = None, ax = None,xl = None,yl = None,\
                   yscale = 'linear',xscale = 'linear',grid = 'on'):
    
        
        # Check if passed function handle
        if ax == None:
            fig,axs = myfig()
            ax = axs[0]
        if xl == None:
            xLim = (x.min(),x.max())
        else:
            xLim = xl
        INXminx = np.argmin(np.abs(x-xLim[0]))    
        INXmaxx = np.argmin(np.abs(x-xLim[1]))+1
        xplot = x[INXminx:INXmaxx]
        # print(INXmin)
        # print(INXmax)
        if yl == None:
            yLim = (np.sign(y.min())*np.abs(y.min())*1.1,y.max()*1.1)
        else:
            yLim = yl
        INXminy = np.argmin(np.abs(x-yLim[0]))    
        INXmaxy = np.argmin(np.abs(x-yLim[1]))+1
        yplot = y[INXminy:INXmaxy]
        
        z = z[INXminx:INXmaxx,INXminy:INXminy]
            
        ax.imagesc(x,y,z,)
        ax.set_xscale(xscale)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        ax.set_xlim(xLim)
        #
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabel)
        ax.set_yticks(yticks)
        ax.set_ylim(yLim)
        # 
        ax.set_title(title)
        # print(xticks)
        ax.grid(visible = grid,which = 'both',axis = 'both')
    
def myimshow(x,y,z,xlabel = 'xlabel',ylabel = 'ylabel',\
           title = 'title', ax = None,xticks = None,yticks=None,xl = None,yl = None,aspect = None,cbar = True):
    
        
   # Check if passed function handle
   if ax == None:
       fig,ax = myfig(options = 'Square')
       
   if xl == None:
       xLim = (x.min(),x.max())
       # print(xLim)
   else:
       xLim = xl
   INXminx = np.argmin(np.abs(x-xLim[0]))    
   INXmaxx = np.argmin(np.abs(x-xLim[1]))+1
   xplot = x[INXminx:INXmaxx]
   # print(INXmin)
   # print(INXmax)
   if yl == None:
       yLim = (np.sign(y.min())*np.abs(y.min())*1.1,y.max()*1.1)
   else:
       yLim = yl
   INXminy = np.argmin(np.abs(y-yLim[0]))    
   INXmaxy = np.argmin(np.abs(y-yLim[1]))+1
   yplot = y[INXminy:INXmaxy]
   
   zplot = z[INXminx:INXmaxx,INXminy:INXmaxy]
       
   if aspect == None: # default is square
       asp = (yplot.max()-yplot.min())/(xplot.max()-xplot.min())
   else:
       asp = aspect*(yplot.max()-yplot.min())/(xplot.max()-xplot.min())
   ax.imshow(zplot, extent = (yplot.min(),yplot.max(),xplot.min(),xplot.max()), aspect = asp)
   ax.set_title(title)
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   plt.tight_layout()
   
   #plt.show()
               

def mywaterfallplot(xx, yy,zz, y2 = np.array([None]), xlabel = 'xlabel',ylabel = 'ylabel',\
           title = 'title',linestyle = '-', color ='b', marker = None,xticks = None,yticks=None,\
               border_color = None, ax = None,xl = None,yl = None,\
                   yscale = 'linear',xscale = 'linear',grid = 'on'):
    
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111, projection = '3d', facecolor = 'white')
    nz = len(z)
    n  = int(nz/6)
    yl = [-4,4]
    for i in range(0, nz, n):
        tinx = ((t>=yl[0])*(t<=yl[-1]))
        yy = t[tinx]
        xx = z[int(i)]*np.ones(len(yy))
        zz = h2_pred[i,tinx]
        zz2 = h2_sim[i,tinx]
        if i == 0:
            ax.plot(xx,yy,zz2,'-b',label = 'SSFM')
            ax.plot(xx,yy,zz,'--r',label = 'MD-PINN')
        else:
            ax.plot(xx,yy,zz2,'-b')
            ax.plot(xx,yy,zz,'--r')
def savemyfig(name = None,cd = 'cwd'):
    if name == None:
        name = 'Figure_'+str(plt.gcf().number) + '.png'
    import os
    if cd == 'cwd':
        fname = os.path.join(os.getcwd(),name)
    else:
        fname = os.path.join(cd,name)
    plt.tight_layout()
    plt.savefig(fname,dpi = 600)
    return



def imagesc(x=None, y=None, data=None, cmap='viridis', colorbar=True, ax=None, 
            xlabel=None, ylabel=None, title=None):
    """
    Mimics MATLAB's imagesc function with axis scaling, axes handle support, and optional colorbar.
    
    Parameters:
    - x: 1D array-like, optional
        The x-axis values (default is pixel indices).
    - y: 1D array-like, optional
        The y-axis values (default is pixel indices).
    - data: 2D array-like
        The input data to be displayed as an image.
    - cmap: str, optional
        The colormap to use for the image (default is 'viridis').
    - colorbar: bool, optional
        Whether to display a colorbar (default is True).
    - ax: matplotlib.axes.Axes, optional
        An existing axes object to plot into. If None, a new figure and axes are created.
    - xlabel: str, optional
        Label for the x-axis (default is None).
    - ylabel: str, optional
        Label for the y-axis (default is None).
    - title: str, optional
        Title of the plot (default is None).
    """
    if data is None:
        raise ValueError("Data must be provided.")
    
    # Determine x and y ranges
    if x is None:
        x = np.arange(data.shape[1])  # Default to pixel indices
    if y is None:
        y = np.arange(data.shape[0])  # Default to pixel indices
    
    # Create a meshgrid for the axes
    extent = [x[0], x[-1], y[0], y[-1]]
    
    # Scale data to [0, 1] range for the colormap
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    normalized_data = (data - data_min) / (data_max - data_min) if data_max > data_min else data
    
    # Use provided axes or create new
    created_new_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_new_fig = True
    img = ax.imshow(normalized_data, cmap=cmap, extent=extent, aspect='auto', origin='lower')
    
    # Add colorbar if requested
    if colorbar:
        cbar = plt.colorbar(img, ax=ax, label='Scaled Intensity')
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Auto-scale axes
    ax.axis('tight')
    
    # Apply tight layout if a new figure was created
    if created_new_fig:
        plt.tight_layout()
        plt.show()


