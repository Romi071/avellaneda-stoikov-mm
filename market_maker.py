import asyncio
import aiohttp
import numpy as np

class Avellaneda_Stoikov():
    def __init__(self):
        #Initial Parameters
        self.mid_price = None
        self.inventory = 0          #Initial q_t (inventory in shares units)
        self.max_inventory = 100    #Max acceptable inventory before stopping to post bids
        self.sigma = 0.5
        self.gamma = 1      #Risk Aversion Constant
        self.kappa = 1      #OB liquidity density
        self.tau = 1        #Constant instead of the original time horizon tau = T - t
        self.active_orders = {}
        self.event = asyncio.Event()


    async def quoting_loop(self, session):
        while True:
            #Sleep for 0.5 seconds between cycles
            await asyncio.sleep(0.5)

            #Cancel the two resting orders
            #Bid order:
            if 'bid' in self.active_orders:
                bid_cancel = self.active_orders['bid']
                self.event.clear()
                async with session.post("http://localhost:8000/cancel", json=bid_cancel) as resp:
                    if resp.status in [200, 404]:
                        cancel_logs = await resp.json()
                        cancelled_amount = cancel_logs['cancelled_amount']
                    else:
                        resp.raise_for_status()
                    #Await for the socket listener to update the mid price
                    if resp.status == 200:
                        await self.event.wait()
                #Add to inventory what was bought (order amount - cancelled amount)
                self.inventory += bid_cancel['amount'] - cancelled_amount
                #Delete from active orders
                del self.active_orders['bid']

            #Ask order:
            if 'ask' in self.active_orders:
                ask_cancel = self.active_orders['ask']
                self.event.clear()
                async with session.post("http://localhost:8000/cancel", json=ask_cancel) as resp:
                    if resp.status in [200, 404]:
                        cancel_logs = await resp.json()
                        cancelled_amount = cancel_logs['cancelled_amount']
                    else:
                        resp.raise_for_status()
                    #Await for the socket listener to update the mid price
                    if resp.status == 200:
                        await self.event.wait()
                #Subtract from inventory what was sold (order amount - cancelled amount)
                self.inventory -= ask_cancel['amount'] - cancelled_amount
                #Delete from active orders
                del self.active_orders['ask']


            #If the mid price doesn't exist at the moment, sleep until it does
            while not self.mid_price:
                self.event.clear()
                await self.event.wait()
            
            #Fix the amounts
            bid_amount = 1
            ask_amount = 1

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
            bid_order = {'side': 'buy', 'price': bid_price, 'amount': bid_amount, 'client_tag': 'as_mm'}
            ask_order = {'side': 'sell', 'price': ask_price, 'amount': ask_amount, 'client_tag': 'as_mm'}

            #Post the orders
            #Bid order:
            if self.inventory <= self.max_inventory:
                async with session.post("http://localhost:8000/order", json=bid_order) as resp:
                    resp.raise_for_status()
                    bid_logs = await resp.json()
                #Add to inventory what was instantly bought (order amount - logs[amount])
                self.inventory += bid_amount - bid_logs['amount']                
                #Add the order to the tracker deque
                if bid_logs['amount'] > 0:
                    self.active_orders['bid'] = {'orderId': bid_logs['orderId'], 'secret_token': bid_logs['secret_token'], 'amount': bid_logs['amount'], 'price': bid_price}
            
            #Ask order:
            if self.inventory >= -self.max_inventory:
                async with session.post("http://localhost:8000/order", json=ask_order) as resp:
                    resp.raise_for_status()
                    ask_logs = await resp.json()
                #Subtract from inventory what was instantly sold (order amount - logs[amount])
                self.inventory -= ask_amount - ask_logs['amount']
                #Add the order to the tracker deque
                if ask_logs['amount'] > 0:
                    self.active_orders['ask'] = {'orderId': ask_logs['orderId'], 'secret_token': ask_logs['secret_token'], 'amount': ask_logs['amount'], 'price': ask_price}


    async def socket_listener(self, session):
        while True:
            async with session.ws_connect("ws://localhost:8000/stream") as ws:
                async for msg in ws:
                    #Parse real time market depth into a dict
                    market_depth = msg.json()
                    #Update the mid price the bot will use as (Best_bid_price + Best_ask_price) / 2 (if both are non-empty)
                    if market_depth['bids'] and market_depth['asks']:
                        self.mid_price = (market_depth['bids'][0][0] + market_depth['asks'][0][0]) / 2
                    else:
                        self.mid_price = None
                    self.event.set()    
                

#Initialize the Avellaneda Stoikov market maker object
market_maker = Avellaneda_Stoikov()

#Main async event loop
async def main():
    #Create an aiohttp session and gather both coroutines
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(market_maker.socket_listener(session), market_maker.quoting_loop(session))


#Run the main event loop
asyncio.run(main())