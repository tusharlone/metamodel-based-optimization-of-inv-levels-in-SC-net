# @author Tushar Lone (Ph.D. Scholar, IIT Goa)

import numpy as np
import simpy
import sys,os
import random

# input parameters
ARR_RATE_LAMBDA = 10 # arrival rate of the customer
INV_STORE = [300, 700] # Inventory capacity (s,S)
DELIVERY_TIME = 2 # delivery time from distributor to retailer
DELIVERY_COST = 1000 # delivery cost per shipment
INV_HOLD_COST = 5 # inventory holding cost
ITEM_COST = 100 # cost of an item
SHELF_LIFE = 7 # Item expires in 7 days
PROFIT_PER_ITEM = 100 # profit earned per item
OLD_PROFIT = 100 # this is created to save original profit in case of dynamic pricing
PURCHASE_LIM = (1,10)
NUM_OF_DAYS = 1000
SHELF_LIFE_THRESHOLD = 5
NUM_OFFER_DAYS = 2

# variables for convinience 
item_shelf_monitor = [] # i'th ele in this list represents number of items having 'i' days of shelf life
# 0th element represents number of items expired
        
# variable to record stats
profit = 0
total_item_sold = []
inventory_levels = []
supply_chain_inv_hold_cost = []
supply_chain_delivery_cost = []
lambda_rec = []
demand_per_day = 0
demand_per_day_rec = []
customer_who_returned = 0
total_customers = 0
items_discarded = []
avg_shelf_life_arr = []

# This is customer class
# obj of this class represents a customer
# the customer can have one retailer
class Customer:
    def __init__(self, purchase_lim, retailer, id_num):
        self.order_count = random.randint(purchase_lim[0],purchase_lim[1])
        self.retailer = retailer
        self.id_num = id_num
        
    def place_order(self):
        # customer places order
        # gets the items, pays for it and leaves
        # if number of items are not available then waits
        global PROFIT_PER_ITEM, profit, total_item_sold, lambda_rec, ARR_RATE_LAMBDA, demand_per_day, customer_who_returned
        # record customer demand 
        demand_per_day += self.order_count
        print(f"PROFIT_PER_ITEM = {PROFIT_PER_ITEM}, cust_num {self.id_num} with demand = {self.order_count}")
        # get the number of items from the inventory
        if(self.retailer.item_store.level>=self.order_count):
            self.retailer.item_store.get(self.order_count)
            # pay the cost for the items
            profit += self.order_count*PROFIT_PER_ITEM
            total_item_sold[-1] += self.order_count
            print(f"got it! profit = {profit}, total_item_sold = {total_item_sold[-1]}")
        else:
            # return without placing order
            customer_who_returned += 1
            print(f"did not get :( returned empty handed!) customer_who_returned = {customer_who_returned}")
            return;
        

# this is a retailer class
# obj of this class represents a retailer
# the retailer can have one supplier
class Retailer:
    def __init__(self, env, distributor):
        global item_shelf_monitor
        self.env = env
        # this retailer has its own inventory 
        self.item_store = simpy.Container(self.env,capacity=INV_STORE[1],init=INV_STORE[1])
        self.distributor = distributor
        self.delivery_ordered = False
        item_shelf_monitor[SHELF_LIFE] = INV_STORE[1]
    
    def place_order(self, env):
        global DELIVERY_COST, SHELF_LIFE, supply_chain_inv_hold_cost, supply_chain_delivery_cost, INV_HOLD_COST, INV_STORE, item_shelf_monitor
        while(True):
            # incured the holding cost of items in inventory today
            supply_chain_inv_hold_cost.append(self.item_store.level*INV_HOLD_COST)
            if(self.item_store.level<=INV_STORE[0] and not self.delivery_ordered):
                num_items_to_order = INV_STORE[1] - self.item_store.level
                print("low level ", self.item_store.level, " at ", self.env.now, num_items_to_order, " items ordered")
                # plcae order
                self.env.process(self.get_delivery(env,num_items_to_order))
            yield env.timeout(1)
    
    def get_delivery(self,env,num_items_to_order):
        global DELIVERY_TIME, item_shelf_monitor, SHELF_LIFE, supply_chain_delivery_cost, DELIVERY_COST
        self.delivery_ordered = True
        # wait for delivery time
        yield self.env.timeout(DELIVERY_TIME)
        # get items in inventory
        self.item_store.put(num_items_to_order)
        item_shelf_monitor[SHELF_LIFE] += num_items_to_order
        supply_chain_delivery_cost.append(num_items_to_order)
        print("delivery from distributor! inventory level = ",self.item_store.level," at ",self.env.now," item_shelf_monitor=",item_shelf_monitor)
        self.delivery_ordered = False

# this is a distributor class
# obj of this class represents a distributor
# it can place order to his supplier
# for time being it is passive, since supply chain does not have a supplier
class Distributor:
    def __init__(self, env):
        self.env = env

        
# this function simulates the arrival of the customers 
# arrival follows Poisson distribution with arrival rate lambda
def customer_arrivals(env,retailer,offer_sale = False):
    global SHELF_LIFE_THRESHOLD, avg_shelf_life_arr, PROFIT_PER_ITEM, ARR_RATE_LAMBDA, items_discarded
    global ITEM_COST, inventory_levels, demand_per_day, demand_per_day_rec, total_item_sold, total_customers
    global OLD_PROFIT, item_shelf_monitor, SHELF_LIFE
    avg_shelf_life_check = 0
    dyn_price_start = False
    print(f"OLD_PROFIT={OLD_PROFIT}, PROFIT_PER_ITEM={PROFIT_PER_ITEM}")
    OLD_PROFIT = PROFIT_PER_ITEM
    cust_id = 0
    while(True):
        print("*************** Day ",env.now," **********************")
        total_item_sold.append(0)
        # calculate arrival rate
        ARR_RATE_LAMBDA = 20 - (ITEM_COST+PROFIT_PER_ITEM)/20
        print(f"ARR_RATE_LAMBDA = {ARR_RATE_LAMBDA}")
        num_of_cust = np.random.poisson(lam=ARR_RATE_LAMBDA, size=1)[0]
        total_customers += num_of_cust
        print(f"Total {num_of_cust} arrived today!")
        for i in range(0,num_of_cust):
            customer_i = Customer(PURCHASE_LIM,retailer,cust_id)
            customer_i.place_order()
            cust_id += 1
        
        # record the current arrival rate
        lambda_rec.append(ARR_RATE_LAMBDA)
        print("at ",env.now," demand = ",demand_per_day,", inv lvl = ", retailer.item_store.level, " at ",env.now)
        # record the inventory levels at the end of the day
        inventory_levels.append(retailer.item_store.level)
        demand_per_day_rec.append(demand_per_day)
        demand_per_day = 0
        
        # update the shelf monitor after consuming items
        items_to_consume = int(total_item_sold[-1])
        i = 1
        while(items_to_consume != 0):
            if(i>SHELF_LIFE):
                print("no sufficient items available!")
                break
            if(item_shelf_monitor[i]<items_to_consume):
                items_to_consume -= item_shelf_monitor[i]
                item_shelf_monitor[i] = 0
            else:
                item_shelf_monitor[i] = item_shelf_monitor[i] - items_to_consume
                items_to_consume = 0
            i += 1

        # shift items in the shelf life monitor array
        for i in range(1,SHELF_LIFE+1):
            item_shelf_monitor[i-1] = item_shelf_monitor[i]
        item_shelf_monitor[SHELF_LIFE] = 0
        print("item_shelf_monitor = ",item_shelf_monitor)
        # discard items which expired today
        if(item_shelf_monitor[0]>0):
            # include loss of discarded items in supply chain cost
            retailer.item_store.get(item_shelf_monitor[0])
            items_discarded.append(item_shelf_monitor[0])
            item_shelf_monitor[0] = 0
            print("item_shelf_monitor (expiration) = ",item_shelf_monitor)
        
        # check if avg shelf life of items is < 5 
        # once we check the avg shelf life, no need to check for next two days
        # since next two days profit = 0

        avg_shelf_life = 0
        # if dynamic pricing is eabled
        if(offer_sale):
            if(avg_shelf_life_check==NUM_OFFER_DAYS):
                avg_shelf_life_check = 0
                dyn_price_start = False
                
            if(sum(item_shelf_monitor)==0):
                # no items in inventory
                avg_shelf_life = 0
            else:
                # calculate avg shelf life    
                for i in range(1,SHELF_LIFE+1):
                    avg_shelf_life += i*item_shelf_monitor[i]
                avg_shelf_life = avg_shelf_life/sum(item_shelf_monitor)
            print("avg_shelf_life = ",avg_shelf_life," avg_shelf_life_check = ",avg_shelf_life_check)
            
            # check if avg shelf life is less than SHELF_LIFE_THRESHOLD
            if(avg_shelf_life>0 and avg_shelf_life<SHELF_LIFE_THRESHOLD):
                if(avg_shelf_life_check==0):
                    PROFIT_PER_ITEM = 0
                    dyn_price_start = True
                    print("Dynamic pricing starts, PROFIT_PER_ITEM = ",PROFIT_PER_ITEM)
                    env.process(resume_price(env,NUM_OFFER_DAYS+1))
            
            if(dyn_price_start):
                avg_shelf_life_check += 1
                    
        avg_shelf_life_arr.append(avg_shelf_life)
        yield env.timeout(1)
        
def resume_price(env,Days):
    global PROFIT_PER_ITEM, OLD_PROFIT
    yield env.timeout(Days)
    # resume the prices
    PROFIT_PER_ITEM = OLD_PROFIT
    print("resume PROFIT_PER_ITEM = ",PROFIT_PER_ITEM)

def print_current_design_settings():
    global ARR_RATE_LAMBDA, INV_STORE, DELIVERY_TIME, DELIVERY_COST, INV_HOLD_COST, ITEM_COST, SHELF_LIFE, PROFIT_PER_ITEM, PURCHASE_LIM, NUM_OF_DAYS, SHELF_LIFE_THRESHOLD, NUM_OFFER_DAYS    
    print(f"****\tCurrent Design Settings\t****\n")
    print(f"Customer arrival rate (lambda) = {ARR_RATE_LAMBDA}, (follows Poisson Ditribution)")
    print(f"Purchase limit of a customer = {PURCHASE_LIM} items per order")
    print(f"Inventory store capacity S = {INV_STORE[1]} items, threshold s = {INV_STORE[0]} items")
    print(f"Inventory holding cost H = {INV_HOLD_COST} per item per day")
    print(f"Shelf life of an item L = {SHELF_LIFE} days")
    print(f"Profit per item sold P = {PROFIT_PER_ITEM}")
    print(f"Item Cost c = {ITEM_COST}")
    print(f"Dynamic pricing: offer sale at discounted prices if avarage shelf life is less than L_s = {SHELF_LIFE_THRESHOLD} days")
    print(f"Sale at discounted prices are offered for next D_L = {NUM_OFFER_DAYS} days")
    print(f"Order delivery time (from Distributor to Retailor) D = {DELIVERY_TIME} days")
    print(f"Delivery cost C = {DELIVERY_COST}")
    print(f"A single Simulation runs for {NUM_OF_DAYS} days")
    print(f"\n****\t****\t****\t****")

    
def single_sim_run(N,s,S,D,C,H,c,p,L,Lt, print_log=False, offer_sale = False, print_stats = False):
    # stats variables
    global avg_shelf_life_arr, items_discarded, item_shelf_monitor, profit, total_item_sold, inventory_levels, supply_chain_inv_hold_cost, supply_chain_delivery_cost, lambda_rec, demand_per_day, demand_per_day_rec, customer_who_returned, total_customers
    
    # design settings
    global ARR_RATE_LAMBDA, INV_STORE, DELIVERY_TIME, DELIVERY_COST, INV_HOLD_COST, ITEM_COST, SHELF_LIFE, PROFIT_PER_ITEM, PURCHASE_LIM, NUM_OF_DAYS, SHELF_LIFE_THRESHOLD, NUM_OFFER_DAYS
    
    # set all design parameters
    NUM_OF_DAYS = N
    INV_STORE[0] = s
    INV_STORE[1] = S
    DELIVERY_TIME = D
    DELIVERY_COST = C
    INV_HOLD_COST = H
    ITEM_COST = c
    PROFIT_PER_ITEM = p
    OLD_PROFIT = p
    SHELF_LIFE = L
    SHELF_LIFE_THRESHOLD = Lt
    
    # reinit all variables to record stats
    #ARR_RATE_LAMBDA = 10
    profit = 0
    total_item_sold.clear()
    inventory_levels.clear()
    supply_chain_inv_hold_cost.clear()
    supply_chain_delivery_cost.clear()
    lambda_rec.clear()
    demand_per_day = 0
    demand_per_day_rec.clear()
    customer_who_returned = 0
    total_customers = 0
    item_shelf_monitor.clear()
    items_discarded.clear()
    avg_shelf_life_arr.clear()
    # initializing the item shelf life monitor
    for i in range(0,SHELF_LIFE+1):
        item_shelf_monitor.append(0)
    old_stdout = sys.stdout
    if(not print_log):
        sys.stdout = open(os.devnull, 'w')
    env = simpy.Environment()
    distributor = Distributor(env)
    retailer = Retailer(env,distributor)
    env.process(retailer.place_order(env))
    env.process(customer_arrivals(env,retailer,offer_sale))
    env.run(NUM_OF_DAYS)
    
    # enable print
    sys.stdout = old_stdout
    
    # calculate stats 
    # avg profit per day
    P_avg = profit/NUM_OF_DAYS
    
    # avg inv holding cost
    InvHold_cost = sum(supply_chain_inv_hold_cost)/NUM_OF_DAYS
    
    # avg delivery cost
    Del_cost = len(supply_chain_delivery_cost)*DELIVERY_COST/NUM_OF_DAYS
    
    # avg supply chain cost per day
    C_avg = InvHold_cost + Del_cost
    
    # Time-averaged number of items in the inventory
    I_avg = []
    for i in range(len(inventory_levels)):
        I_avg.append([i+1,sum(inventory_levels[0:i+1])/(i+1)])
        
    # Throughput (average number of items sold per-day)
    T_avg = np.mean(total_item_sold)
    
    # Monthly Inventory turnover ratio (R)
    turnover_ratio_R = []
    inx = 0
    for i in range(30,NUM_OF_DAYS,30):
        turnover_ratio_R.append(sum(total_item_sold[inx:i])/I_avg[i][1])
        inx = i
    
    # Fraction of customers that do not return
    customer_left = customer_who_returned*100/total_customers
    
    
    # avg number of items expired per day
    I_exp = sum(items_discarded)/NUM_OF_DAYS
    
    # Net avg profit per-day
    P_net = P_avg - C_avg - (I_exp*ITEM_COST)
    if print_stats:
        print(f"P_avg = {P_avg} Per-day avg profit generated by selling items")
        print(f"T_avg = {T_avg} Per-day avg throughput (items sold per-day)")
        print(f"I_avg = {I_avg[-1][1]} Time-averaged inventory level on any day")
        print(f"R_monthly = {turnover_ratio_R[0]:.2f} Monthly Inventory turnover ratio")
        print(f"I_exp = {I_exp} Avg number of items discarded daily due to expiry")
        print(f"Loss_discarded = {I_exp*ITEM_COST} Avg daily loss because of discarding expired items")
        print(f"Frac customers returned without getting items = {customer_left:.2f}%")
        print(f"C_avg = {C_avg} Per-day avg cost ")
        print(f"\tC_avg_holding = {InvHold_cost} Per-day avg cost of holding inventory")
        print(f"\tC_avg_delivery = {Del_cost} Per-day avg cost of refill delivery")
        print(f"---------------\nP_net = {P_net} Net avg profit per-day\n---------------")
    
    return P_avg, C_avg, InvHold_cost, Del_cost, I_avg[-1][1], T_avg, P_net