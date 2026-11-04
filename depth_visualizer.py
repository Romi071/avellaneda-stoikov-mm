from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import qasync
import asyncio
import aiohttp
import socket
import json
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
        self.bids_curve = self.win.plot(pen=pg.mkPen('g', width=2), fillLevel=0, brush=(0, 255, 0, 50))
        self.asks_curve = self.win.plot(pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255, 0, 0, 50))

        #Create the vertical reservation price line
        self.res_line = pg.InfiniteLine(angle=90, 
                                        movable=False, 
                                        pen=pg.mkPen('#FFD700', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                        label='Res Px: ${value:0.2f}',
                                        labelOpts={'position': 0.85, 'color': '#FFD700', 'fill': (20, 20, 20, 200), 'movable': True})
        self.res_line.hide()
        self.win.addItem(self.res_line)
        
        #Show the window
        self.win.show()

        #UDP socket and telemetry setup
        self.telemetry = None
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(("127.0.0.1", 9999))
        self.udp_socket.setblocking(False)
 
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

        #Extract the data from the UDP socket until no data is present (try except on BlockingIOError)
        try:
            while True:
                #Recieve the raw bytes data from the UDP socket (set max buffer size to 1024 bytes to have enough leeway)
                data, _ = self.udp_socket.recvfrom(1024)
                self.telemetry = json.loads(data.decode('utf-8'))
        except BlockingIOError:
            pass

        if self.telemetry:
            #Make the wealth marker update in 20 fps
            if self.latest_snapshot and self.latest_snapshot['bids'] and self.latest_snapshot['asks']:
                live_mid = (bid_arr_price[0] + ask_arr_price[0]) / 2
                wealth = self.telemetry['balance'] + self.telemetry['inventory'] * live_mid
            else:
                wealth = self.telemetry['wealth']

            #Create the reservation price line
            self.res_line.setPos(self.telemetry['res_price'])
            self.res_line.show()

            #Dynamic colors and signs for PnL and Inventory
            pnl_color = "#00FF66" if wealth >= 0 else "#FF3355"
            pnl_sign = "+" if wealth >= 0 else "-"
            inv = self.telemetry['inventory']
            inv_color = "#00D4FF" if inv > 0 else ("#FFAA00" if inv < 0 else "#FFFFFF")
            inv_sign = "+" if inv > 0 else ""

            #Build the HTML HUD string and push it to the title bar
            hud_html = (f"<span style='font-family: Consolas, monospace; font-size: 14pt; color: #888888;'>"
                        f"PnL: <b style='color: {pnl_color};'>{pnl_sign}${abs(wealth):.2f}</b> &nbsp;|&nbsp; "
                        f"Cash: <b style='color: #FFFFFF;'>${self.telemetry['balance']:.2f}</b> &nbsp;|&nbsp; "
                        f"Inv: <b style='color: {inv_color};'>{inv_sign}{inv:.2f}</b> &nbsp;|&nbsp; "
                        f"Res Px: <b style='color: #FFD700;'>${self.telemetry['res_price']:.2f}</b>"
                        f"</span>")
            self.win.setTitle(hud_html)
            
    async def socket_listener(self, session):
        #Connect to the depth-streaming WebSocket and parse the response into a dictionary
        async with session.ws_connect("ws://localhost:8000/stream") as ws:
            async for msg in ws:
                self.latest_snapshot = msg.json()
 
#Initialize the Visualizer object
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