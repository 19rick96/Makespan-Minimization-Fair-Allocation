# Makespan-Minimization-Fair-Allocation

Decentralized-OTA.py --> Implements Decentralized Operation Trading Algorithm. This algorithm has been designed to generate DEQx allocations. 
                         Additionally, the algorithm also minimizes the makespan for a set of agents/robots with identical functionalities 
                         but possibly different speeds. The agents must be connected but need not be fully connected.
                         
                         Properties of the Algorithm:
                        
                            1. The algorithm guarantees an approximation factor of 2 for identical and fully connected agents.
                            2. The algorithm guarantees an approx. factor of 1.618 for 2 agents when they have different speeds.
                            3. The algorithm guarantees an approx. factor of (1 + \sqrt(4n-3))/2 for n agents with different speeds.
                            4. The algorithm is paramtereless.
                            5. The algorithm can start from any random initialization, and can thus be implemented in a reactive fashion.
                            6. Although the approx. factors are for fully conncected network of agents, the algorithm performs very well 
                               for partially connected network as well.
                            7. In the current implementation of Decentralized-OTA(n), the agents need to communicate and store only its 
                               own allocation. Thus, the message sizes are equal to the number of operations belonging to an agent.
                            8. The algorithm converges when the agents reach a local concensus. Hence, the algorithm is fast and practical.
                               With some added complexity on the convergence criteria, it should be possible to guarantee the approximation 
                               factors even when the agents are not fully connected. (Future Work)
                             
Market Based Algorithm --> Guaranteed to generate an allocation that satisfies partial-DEQ1 and Pareto Optimality even when the agents are completely
                           non-identical. 
                           
                           Guarantees an Approximation factor of 1.5 for 2 non-identical agents. (This is the best possible theoretically)
                           Additionally, the algorithm generates extremely good allocations in terms of makespan.
