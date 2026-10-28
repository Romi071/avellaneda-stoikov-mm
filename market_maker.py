import asyncio
import aiohttp
import numpy as np

class Avellaneda_Stoikov():
    def __init__(self):
        #Initial Parameters
        self.mid_price = None
        self.inventory = 0  #Initial q_t (inventory in shares units)
        self.sigma = 0.5
        self.gamma = 1      #Risk Aversion Constant
        self.kappa = 1      #OB liquidity density
        self.tau = 1        #Constant instead of the original time horizon tau = T - t


    async def quoting_loop(self):
        #Sleep for 0.5 seconds between cycles
        await asyncio.sleep(0.5)

        #If the mid price doesn't exist at the moment, sleep until it does
        while not self.mid_price:
            await asyncio.sleep(0.25)

        #Compensation for inventory variance risk over tau units of time
        c = self.gamma * self.sigma**2 * self.tau

        #AS Equation 1 (Reservation Price):
        res_price = self.mid_price - c * self.inventory

        #AS Equation 2 (Optimal Spread):
        opt_spread = c + (2/self.gamma) * np.log(1 + self.gamma / self.kappa)

        #AS Equation 3 (Optimal executable quotes):
        bid_price = round(res_price - opt_spread / 2, 2)
        ask_price = round(res_price + opt_spread / 2, 2)

        #Create the orders
        bid_order = {'side': 'buy', 'price': bid_price, 'amount': 0, 'client_tag': 'as_mm'}
        ask_order = {'side': 'sell', 'price': ask_price, 'amount': 0, 'client_tag': 'as_mm'}

    

    async def socket_listener(self, session):
        async with session.ws_connect("ws://localhost:8000/stream") as ws:
            async for msg in ws:
                #Parse real time market depth into a dict
                market_depth = msg.json()
                #Update the mid price the bot will use as (Best_bid_price + Best_ask_price) / 2 (if both are non-empty)
                if market_depth['bids'] and market_depth['asks']:
                    self.mid_price = (market_depth['bids'][0][0] + market_depth['asks'][0][0]) / 2
                else:
                    self.mid_price = None
                

#Initialize the Avellaneda Stoikov market maker object
market_maker = Avellaneda_Stoikov()

#Main async event loop
async def main():
    while True:
        #Create an aiohttp session and run the socket listener coroutine
        async with aiohttp.ClientSession() as session:
            await market_maker.socket_listener(session)


#Run the main event loop
asyncio.run(main())