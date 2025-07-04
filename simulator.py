import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import sys

def run_simulation(arrival_rate=2, service_rate=5, simulation_time=5, max_queue_size=1000):
    """
    Run simulation for sections 1,2 and 3.
    """
    # Parameters
    # arrival_rate -> lambda
    # service_rate -> mu
    # simulation_time -> T
    # max_queue_size -> N

    # Events
    ARRIVAL = 1
    DEPARTURE = 2

    # State variables
    current_time = 0.0
    queue = []
    server_busy = False
    event_list = []

    # Statistics
    total_wait_time = 0.0
    num_customers_served = 0

    # Genereate random seed
    random.seed()

    # Schedule the first arrival
    heapq.heappush(event_list, (random.expovariate(arrival_rate), ARRIVAL))

    while current_time < simulation_time and event_list:
        event_time, event_type = heapq.heappop(event_list)
        current_time = event_time
        
        if event_type == ARRIVAL:
            if not server_busy:
                server_busy = True
                service_time = random.expovariate(service_rate)
                total_wait_time += service_time
                heapq.heappush(event_list, (current_time + service_time, DEPARTURE))
            else:
                # If queue is not full, add current_time to it. Otherwise, drop the packet.
                if(len(queue)<max_queue_size):
                    queue.append(current_time)
            next_arrival = current_time + random.expovariate(arrival_rate)
            if(next_arrival < simulation_time):
                heapq.heappush(event_list, (next_arrival, ARRIVAL))
        elif event_type == DEPARTURE:
            num_customers_served += 1
            if queue:
                arrival_time = queue.pop(0)
                service_time = random.expovariate(service_rate)
                # If this package is going to end after the simulation end, add only the time until the end of the simulation
                wait_time = min((current_time + service_time) - arrival_time, simulation_time - arrival_time)
                total_wait_time += wait_time
                heapq.heappush(event_list, (current_time + service_time, DEPARTURE))
            else:
                server_busy = False

    # Results
    # print(f"Number of customers served: {num_customers_served}")
    # print(f"Average wait time: {total_wait_time / num_customers_served if num_customers_served > 0 else 0}")
    return [num_customers_served, total_wait_time / num_customers_served if num_customers_served > 0 else 0]

def make_tables():
    """
    Make tables for section 3.4.
    """
    # Calculate Error(x):
    arrival_rate = 2 # lambda
    service_rate = 5 # mu
    T_range = range(10,101,10)

    theoretical_num_of_customers_served = []
    simulation_num_customers_served = []
    theoretical_mean_time = []
    simulation_mean_time = []

    for T in T_range:
        theoretical_num_of_customers_served.append(arrival_rate*T - (arrival_rate / (service_rate - arrival_rate)))
        theoretical_mean_time.append(1/3)

        # Getting avarage values for 20 runs of the simulation with current T.
        mean_num_customers_served = 0
        mean_mean_time = 0
        for i in range(20):
            num_customers_served, mean_time = run_simulation(simulation_time=T)
            mean_num_customers_served += num_customers_served
            mean_mean_time += mean_time
        # Calculate the mean values.
        mean_num_customers_served = mean_num_customers_served / 20
        mean_mean_time = mean_mean_time / 20
        # Append to the simulation return values record.
        simulation_num_customers_served.append(mean_num_customers_served)
        simulation_mean_time.append(mean_mean_time)

    # Plot for mean time in system
    # errorArr = theoretical_values - real_values
    errorArr = np.subtract(theoretical_mean_time, simulation_mean_time)
    # errorArr = |theoretical_values - real_values|
    errorArr = np.abs(errorArr)
    #errorArr = |theoretical_values - real_values| / theoretical_values
    errorArr = errorArr / np.array(theoretical_mean_time)
    # errorArr = (|theoretical_values - real_values| / theoretical_values)*100
    errorArr = errorArr*100

    plt.plot(T_range, errorArr)
    plt.title("mean time error as a function of T")
    plt.xlabel("simulation time T")
    plt.ylabel("relative error in %")

    plt.show()

    #Plot for number of customers served
    # errorArr = theoretical_values - real_values
    errorArr = np.subtract(theoretical_num_of_customers_served, simulation_num_customers_served)
    # errorArr = |theoretical_values - real_values|
    errorArr = np.abs(errorArr)
    #errorArr = |theoretical_values - real_values| / theoretical_values
    errorArr = errorArr / np.array(theoretical_num_of_customers_served)
    # errorArr = (|theoretical_values - real_values| / theoretical_values)*100
    errorArr = errorArr*100

    plt.plot(T_range, errorArr)
    plt.title("number of served costumers error as a function of T")
    plt.xlabel("simulation time T")
    plt.ylabel("relative error in %")

    plt.show()

def is_number(num):
    try:
        float(num)
    except ValueError:
        return False
    return True

def is_int(num):
    try:
        int(num)
    except ValueError:
        return False
    return True

def parse_input():
    """
    Parsing the input for section 4. 
    Return the relevant parameters as numbers or lists of floats.
    """
    error_return_value = [-1,-1,-1,-1,-1,-1]

    input_length = len(sys.argv) - 1
    if input_length < 2:
        print("2 bad inpuit")
        return error_return_value
    # Check input validation
    if False in [is_number(x) for x in sys.argv[1:]]:
        print("1 bad inpuit")
        return error_return_value
    # First 2 values must be int
    if (not is_int(sys.argv[1])) or (not is_int(sys.argv[2])):
        print("1.5 bad inpuit")
        return error_return_value
    if int(sys.argv[2]) < 1:
        print("1.5 bad inpuit")
        return error_return_value
    
    simulation_time = int(sys.argv[1])
    num_of_servers = int(sys.argv[2])
    arrival_rate = -1
    prob_array_P = []
    queue_size_array_Q = []
    serving_rate_array_mu = []

    for i in range(3,input_length + 1):
        # Add all of the probabilities to the list
        if 2 < i and i <= 2 + num_of_servers:
            prob_array_P.append(float(sys.argv[i]))
        elif i == (2 + num_of_servers) + 1:
            arrival_rate = float(sys.argv[i])
        elif (2 + num_of_servers) + 1 < i and i <= 3 + 2*num_of_servers:
            queue_size_array_Q.append(float(sys.argv[i]))
        else:
            serving_rate_array_mu.append(float(sys.argv[i]))

    # Check that each of the lists are correct:
    if (not len(prob_array_P)==num_of_servers) or (not len(queue_size_array_Q)==num_of_servers) or (not len(serving_rate_array_mu)==num_of_servers):
        print("3 bad inpuit") 
        return error_return_value
    # Check if the probabilities are valid:
    if (False in [(p >= 0 and p <= 1) for p in prob_array_P]) or (sum(prob_array_P) != 1):
        print("4 bad inpuit")
        return error_return_value
    # Check that arrival times are non-negative:
    if (False in [mu >= 0 for mu in serving_rate_array_mu]) or arrival_rate < 0:
        print("5 bad inpuit")
        return error_return_value

    return simulation_time, num_of_servers, prob_array_P, arrival_rate, queue_size_array_Q, serving_rate_array_mu


def simulation(simulation_time, num_of_servers, prob_array_P, arrival_rate, queue_size_array_Q, serving_rate_array_mu):
    """
    Running the main simulation, section 4.

    The main idea is:
    Distribute all of the packages to the different servers. Each server will have a queue with times of incoming packages.
    Then each server could serve this packages, in a similar way to section 1.
    """
    # The below values are values that we want to find.
    num_of_served_packages = 0
    num_of_thrown_packages = 0
    last_serve_time = 0
    total_wait_time = 0
    avarage_wait_time = 0
    total_serve_time = 0
    avarage_serve_time = 0

    # First, distribute incoming packages to the servers.
    
    # Events
    ARRIVAL = 1
    DEPARTURE = 2

    # Initiallize a list of queues. Queue number i contains the packages for the i's server.
    servers_heaps = []

    for i in np.arange(0,num_of_servers):
        servers_heaps.append([])

    # Start going thorugh the incoming packages, each time insert the new package to a chosen queue(based on the given probabilities list).
    current_time = random.expovariate(arrival_rate)
    while current_time < simulation_time:
        # Choose a server to put the package inside of it. Choose according to given probabilities.
        server =  np.random.choice(np.arange(0, num_of_servers), p=prob_array_P)
        heapq.heappush(servers_heaps[server], (current_time, ARRIVAL))
        current_time = current_time + random.expovariate(arrival_rate)

    # Now we have a list of queues, each queue contains the packages that the load balancer assign to it.
    # Specifically, the queues contains the beggining time of the packages.

    # We want each server to deal with it's incoming packages, while recording the relevant data.
    # We will work in a similar way to section 1 in the assignment.
    for i in np.arange(0,num_of_servers):
        # Deal with each server individually
        server_busy = False
        max_queue_size = queue_size_array_Q[i]
        queue = []
        service_rate = serving_rate_array_mu[i]
        current_heap = servers_heaps[i]
        current_time = 0

        # If there are no packages to be served, continue to next server.
        if len(current_heap) == 0:
            continue
            
        while current_time < simulation_time and current_heap:
            current_time, event_type = heapq.heappop(current_heap)

            if event_type == ARRIVAL:
                if not server_busy:
                    server_busy = True
                    service_time = random.expovariate(service_rate)
                    # The package is served immediately so only has serve time. Did not have wait time.
                    total_serve_time += service_time
                    total_wait_time += 0
                    heapq.heappush(current_heap, (current_time + service_time, DEPARTURE))
                else: # The server is busy -> send the package to queue. If the queue is full -> drop the package.
                    if(len(queue)<max_queue_size):
                        queue.append(current_time)
                    else:
                        num_of_thrown_packages += 1
            elif event_type == DEPARTURE:
                num_of_served_packages += 1
                if queue:
                    arrival_time = queue.pop(0)
                    service_time = random.expovariate(service_rate)
                    total_wait_time += current_time - arrival_time
                    total_serve_time += service_time
                    heapq.heappush(current_heap, (current_time + service_time, DEPARTURE))

                    # Check if this is the last served package
                    if current_time + service_time > last_serve_time:
                        last_serve_time = current_time + service_time
                else:
                    server_busy = False
                if current_time > last_serve_time:
                    last_serve_time = current_time
        
    avarage_wait_time = total_wait_time / num_of_served_packages
    avarage_serve_time = total_serve_time / num_of_served_packages


    return num_of_served_packages, num_of_thrown_packages, last_serve_time, avarage_wait_time, avarage_serve_time

# Check if the input is valid. If valid -> run the main simulation.
simulation_time, num_of_servers, prob_array_P, arrival_rate, queue_size_array_Q, serving_rate_array_mu = parse_input()
if(prob_array_P!=-1):
    results = simulation(simulation_time, num_of_servers, prob_array_P, arrival_rate, queue_size_array_Q, serving_rate_array_mu)
    print(f"{results[0]:.4f} {results[1]:.4f} {results[2]:.4f} {results[3]:.4f} {results[4]:.4f}")
