from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import qasync
import asyncio
import aiohttp
import sys

#Initialize the QT app
app = QtWidgets.QApplication(sys.argv)

class Visualizer():
    def __init__(self):
        #Declare the snapshot variable as None and initialize pyqtgraph's window
        self.latest_snapshot = None
        self.win = pg.PlotWidget()
        #Title
        self.win.setTitle("Real-Time Market Depth Chart", size="18pt")
        #Axis Labels
        label_style = {'font-size': '14pt', 'font-weight': 'bold'}
        self.win.setLabel('bottom', 'Price', units='$', **label_style)
        self.win.setLabel('left', 'Cumulative Depth', units='Shares', **label_style)
        #Axis Tick Numbers
        tick_font = QtGui.QFont()
        tick_font.setPointSize(11)
        self.win.getAxis('bottom').setTickFont(tick_font)
        self.win.getAxis('left').setTickFont(tick_font)
        #Aesthetics
        self.win.showGrid(x=True, y=True, alpha=0.3)
        pg.setConfigOptions(antialias=True)
        #Initialize persistent filled RG curves with wide borders
        self.bids_curve = self.win.plot(pen='g', fillLevel=0, brush=(0, 255, 0, 50))
        self.bids_curve = self.win.plot(pen=pg.mkPen('g', width=2), fillLevel=0, brush=(0, 255, 0, 50))
        self.asks_curve = self.win.plot(pen='r', fillLevel=0, brush=(255, 0, 0, 50))
        self.asks_curve = self.win.plot(pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255, 0, 0, 50))
        #Show the window
        self.win.show()

        #50ms timer on the pyqtgraph coroutine that updates the graph in real time
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.chart_plotter)
        self.timer.start(50)

    def chart_plotter(self):
        #Parse the json snapshot into two distinct 2D np arrays, checking if they're empty
        if self.latest_snapshot:
            #Bids
            if self.latest_snapshot['bids']:
                bid_arr = np.array(self.latest_snapshot['bids'])
                #Prices
                bid_arr_price = bid_arr[:, 0]
                #Volumes
                bid_arr_vol = np.cumsum(bid_arr[:, 1])
                #Plot the data
                self.bids_curve.setData(bid_arr_price, bid_arr_vol)
            else:
                self.bids_curve.setData([], [])

            #Asks
            if self.latest_snapshot['asks']:
                ask_arr = np.array(self.latest_snapshot['asks'])
                #Prices
                ask_arr_price = ask_arr[:, 0]
                #Volumes
                ask_arr_vol = np.cumsum(ask_arr[:, 1])
                #Plot the data
                self.asks_curve.setData(ask_arr_price, ask_arr_vol)
            else:
                self.asks_curve.setData([], [])
            
    async def socket_listener(self, session):
        #Connect to the depth-streaming WebSocket and parse the response into a dictionary
        async with session.ws_connect("ws://localhost:8000/stream") as ws:
            async for msg in ws:
                self.latest_snapshot = msg.json()
 
#Initialize the visualizer object
visualizer = Visualizer()

#Main qasync-shared event loop
async def main():
    while True:
        #Create an aiohttp session and run the socket listener coroutine
        async with aiohttp.ClientSession() as session:
            await visualizer.socket_listener(session)

        
#Initialize the shared event loop
loop = qasync.QEventLoop(app)
asyncio.set_event_loop(loop)
with loop: loop.run_until_complete(main())