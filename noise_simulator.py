import asyncio
import aiohttp
import random
from collections import deque

class MarketSimulator():
    def __init__(self):
        #Initial params
        self.fair_price = 100
        self.sigma = 0.5
        self.poisson_intensity = 2
        self.dt = 0.25
        self.N_max = 50

    async def maker_loop(self, session):
        while True:
            #Maker Loop heartbeat frequency
            await asyncio.sleep(self.dt)

            #Step the fair price using Arithmetic Brownian Motion (ABM)
            Z = random.gauss(0, 1)
            self.fair_price += self.sigma * self.dt**(1/2) * Z

            delta_bid = random.uniform(0.02, 0.2)
            delta_ask = random.uniform(0.02, 0.2)

            bid_price = round(self.fair_price - delta_bid, 2)
            ask_price = round(self.fair_price + delta_ask, 2)

            bid_order = {'side': 'buy', 'price': bid_price, 'amount': random.uniform(0.1, 5)}
            ask_order = {'side': 'sell', 'price': ask_price, 'amount': random.uniform(0.1, 5)}

            async with session.post("/order", json=bid_order) as resp:
                resp.raise_for_status()
                bid_logs = await resp.json()
            async with session.post("/order", json=ask_order) as resp:
                resp.raise_for_status()
                ask_logs = await resp.json()

#Initialize the market simulator object
market_sim = MarketSimulator()

#Main Event Loop
async def main():
    async with aiohttp.ClientSession(base_url='http://localhost:8000') as session:
            #Noise Maker loop
            await market_sim.maker_loop(session)

asyncio.run(main())