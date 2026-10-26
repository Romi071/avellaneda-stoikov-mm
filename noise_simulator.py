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
        self.pareto_alpha = 1.5
        self.dt = 0.25
        self.N_max = 50
        self.active_maker_orders = deque([])

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

            bid_order = {'side': 'buy', 'price': bid_price, 'amount': random.uniform(0.1, 5), 'client_tag': 'sim_maker'}
            ask_order = {'side': 'sell', 'price': ask_price, 'amount': random.uniform(0.1, 5), 'client_tag': 'sim_maker'}

            async with session.post("/order", json=bid_order) as resp:
                resp.raise_for_status()
                bid_logs = await resp.json()
            async with session.post("/order", json=ask_order) as resp:
                resp.raise_for_status()
                ask_logs = await resp.json()
            
            #Add the orders to the tracker deque
            if bid_logs['amount'] > 0:
                self.active_maker_orders.append({'orderId': bid_logs['orderId'], 'secret_token': bid_logs['secret_token']})
            if ask_logs['amount'] > 0:
                self.active_maker_orders.append({'orderId': ask_logs['orderId'], 'secret_token': ask_logs['secret_token']})

            #Garbage cleaner
            while len(self.active_maker_orders) > self.N_max:
                cancel_order = self.active_maker_orders.popleft()
                async with session.post("/cancel", json=cancel_order) as resp:
                    try:
                        resp.raise_for_status()
                        cancel_success = await resp.json()
                    except aiohttp.ClientResponseError:
                        pass

    async def taker_loop(self, session):
        while True:
            #Taker Loop random exponential frequency
            taker_dt = random.expovariate(self.poisson_intensity)
            await asyncio.sleep(taker_dt)

            #Taker order params
            taker_side = random.choice(['buy', 'sell'])
            if taker_side == 'buy':
                taker_price = round(5 * self.fair_price, 2)
            else:
                taker_price = round(1/5 * self.fair_price, 2)
            #Taker amount is calculated using a capped Pareto distribution with parameter pareto_alpha to simulate realistic market conditions
            taker_amount = max(0.1, min(25, 0.1 * random.paretovariate(self.pareto_alpha)))
            
            #Post the taker IOC order
            taker_order = {'side': taker_side, 'price': taker_price, 'amount': taker_amount, 'order_type': 'IOC', 'client_tag': 'sim_taker'}
            async with session.post("/order", json=taker_order) as resp:
                resp.raise_for_status()
                await resp.json()

#Initialize the market simulator object
market_sim = MarketSimulator()

#Main Event Loop
async def main():
    async with aiohttp.ClientSession(base_url='http://localhost:8000') as session:
            #Concurrent Maker and Taker Event loop
            await asyncio.gather(market_sim.maker_loop(session), market_sim.taker_loop(session))

asyncio.run(main())