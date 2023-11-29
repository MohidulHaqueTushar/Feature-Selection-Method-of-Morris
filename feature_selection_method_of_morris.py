#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
    Simulation model (non-agent-based) of the labor and 
    financial markets with business cycles inspired by the Goodwin-Minsky-Keen
    model (Keen 1995). Grasselli's more recent works and recent 
    presentations, e.g. https://ms.mcmaster.ca/~grasselli/kruger2011.pdf
    
    The script exposes to a number of model parameters.
    Method of Morris used here to determine which parameters matter the most. 
    
        
Author: Md Mohidul Haque
"""


import matplotlib.pyplot as plt
import numpy as np

""" Simulation class. Contains the model and handles its simulation."""
class Simulation_Model():
    def __init__(self,
                 productivity_growth = 0.17,
                 population_growth = 0.0,
                 depreciation_rate = 0.4,
                 capital_output_ratio = 0.8,
                 interest_rate = 0.04,
                 debt_function_parameter = 1.1,
                 philipps_curve_exponent = 1,
                 philipps_curve_factor = 0.2,
                 wage_share_initial = 0.65,
                 employment_rate_initial = 0.9,
                 banking_share_initial = 0.5,
                 t_max = 100,
                 dt = 0.01):
        """
        Constructor method.

        Parameters
        ----------
        productivity_growth : float, optional
            Productivity growth parameter (alpha in Keen 1995). The default is 0.17.
        population_growth : float, optional
            Population growth parameter (beta in Keen 1995). The default is 0.0.
        depreciation_rate : float, optional
            Depreciation rate parameter (gamma in Keen 1995). The default is 0.4.
        capital_output_ratio : numeric, optional
            Capital-output-ratio parameter (nu in Keen 1995). The default is 0.8.
        interest_rate : float, optional
            Interest rate parameter (r in Keen 1995). The default is 0.04.
        debt_function_parameter : numeric, optional
            Debt function parameter (k in Keen 1995). The default is 1.1.
        philipps_curve_exponent : numeric, optional
            Employment rate exponent in the Philipps curve function. The 
            default is 1.
        philipps_curve_factor : numeric, optional
            Employment rate factor in the Philipps curve function. The 
            default is 0.2.
        wage_share_initial : float, optional
            Initial value of the wage share state variable (omega in Keen 1995). 
            The default is 0.65.
        employment_rate_initial : float, optional
            Initial value of the employment rate state variable (lambda in Keen
            1995). The default is 0.9.
        banking_share_initial : float, optional
            Initial value of the banking share of the economy state variable
            (d in Keen 1995). The default is 0.5.
        t_max : numeric, optional
            Length of the simulation. The default is 100.
        dt : numeric, optional
            Step size (granularity of the simulation). The default is 0.01.

        Returns
        -------
        None.

        """
                
        """ Record parameters"""
        self.alpha = productivity_growth
        self.beta = population_growth
        self.gamma = depreciation_rate
        self.nu = capital_output_ratio
        self.r = interest_rate
        self.kappa = debt_function_parameter
        self.philipps_curve_exponent = philipps_curve_exponent
        self.philipps_curve_factor = philipps_curve_factor
        self.t_max = t_max
        self.dt = dt
        
        """ Initialize state variables"""
        """ Wage share of income"""        
        self.w = wage_share_initial
        """ Employment rate"""
        self.v = employment_rate_initial    
        """ Banking share of the economy"""
        self.d = banking_share_initial
        
        """ Prepare history records"""
        self.history_t = []
        self.history_v = []
        self.history_w = []
        self.history_d = []

    def philipps_curve(self):
        """
        Philipps curve method.

        Returns
        -------
        numeric
            Value of the Philipps curve term used in the development equation
            of the wage share state variable.

        """
        return self.v**self.philipps_curve_exponent \
                        * self.philipps_curve_factor

    def f_y(self):
        """
        Debt function method. The function is used in the development equations
        of both the employment rate state variable and the banking share of the 
        economy state variable.

        Returns
        -------
        res : numeric
            Function value.

        """
        res = self.kappa / self.nu**2 * (1- self.w**1.1 - self.r*self.d**1.1)
        return res


    def run(self):
        """
        Run method. Handles the time iteration of the simulation

        Returns
        -------
        None.

        """

        for t in range(int(self.t_max / self.dt)):
            """ Wage share change"""
            dw = self.philipps_curve() * self.w - self.alpha * self.w
        	
            """ Employment rate change"""
            dv = self.f_y() * self.v * (1 - self.v**100) + \
                            (- self.alpha - self.beta - self.gamma) * self.v
            
            """ Banking share change"""
            dd = self.d * (self.r - self.f_y() + self.gamma) + \
                                        self.nu * self.f_y() - (1 - self.w)
            
            """ Compute absolutes"""
            self.w += dw * self.dt
            self.v += dv * self.dt
            self.d += dd * self.dt
            
            """ Make sure state variables are not out of bounds"""
            self.ensure_state_validity()
            
            """ Record into history"""
            self.history_w.append(self.w)
            self.history_v.append(self.v)
            self.history_d.append(self.d)
            self.history_t.append(t / self.dt)

    def ensure_state_validity(self):
        """
        Method for ensuring state validity. Checks that all state variables are
        still in their valid areas (between 0 and 1). It corrects the state
        variables otherwise to allow the simulation to continue gracefully.

        Returns
        -------
        None.

        """
        try:
            assert 0 < self.w 
        except:
            print("w=", self.w, "<0")
            self.w = 0.001
        try:
            assert self.w < 1 
        except:
            print("w=", self.w, ">1")
            self.w = 0.999
        try:
            assert 0 < self.v 
        except:
            print("v=", self.v, "<0")
            self.v = 0.001
        try:
            assert self.v < 1
        except:
            print("v=", self.v, ">1")
            self.v = 0.999
        try:
            assert 0 < self.d 
        except:
            print("d=", self.d, "<0")
            self.d = 0.001
        try:
            assert self.d < 1
        except:
            print("d=", self.d, ">1")
            self.d = 0.999
    
    def return_results(self, show_plot = False):
        """
        Method for returning and visualizing results
        
        Parameters
        ----------
        show_plot: bool, optional 
            show the plot of the results, by default false. 

        Returns
        -------
        simulation_history : dict
            Recorded data on the simulation run.

        """
        
        """ Prepare return dict"""                
        simulation_history = {"history_t": self.history_t,
                              "history_w": self.history_w,
                              "history_v": self.history_v,
                              "history_d": self.history_d}
        
        """ Create figure showing the development of the simulation in six
            subplots."""
        if show_plot:
            fig, ax = plt.subplots(nrows=2, ncols=3, squeeze=False)
            ax[0][0].plot(self.history_t, self.history_v)
            ax[0][0].set_xlabel("Time")
            ax[0][0].set_ylabel("Employment rate")
            ax[0][1].plot(self.history_t, self.history_w)
            ax[0][1].set_xlabel("Time")
            ax[0][1].set_ylabel("Wage share")
            ax[0][2].plot(self.history_t, self.history_d)
            ax[0][2].set_xlabel("Time")
            ax[0][2].set_ylabel("Banking share of the ec.")
            ax[1][0].plot(self.history_d, self.history_v)
            ax[1][0].set_xlabel("Banking share of the economy")
            ax[1][0].set_ylabel("Employment rate")
            ax[1][1].plot(self.history_d, self.history_w)
            ax[1][1].set_xlabel("Banking share of the ec.")
            ax[1][1].set_ylabel("Wage share")
            ax[1][2].plot(self.history_w, self.history_v)
            ax[1][2].set_xlabel("Wage share")
            ax[1][2].set_ylabel("Employment rate")
            plt.tight_layout()
            plt.savefig("business_cycle_simulation.pdf")
            plt.show()

        return simulation_history
    
class MorrisMethod():
    def __init__(self, sample_size=10):
        """
        Constructor method. Defined parameter ranges and Morris distances and
        prepares output variables.

        Parameters
        ----------
        sample_size : int, optional
            Number of different Morris trajectories. The default is 10.

        Returns
        -------
        None.

        """
        
        """ Record parameters"""
        self.sample_size = sample_size
        
        """ Define parameter ranges"""
        self.parameter_arrangement = {"productivity_growth": [0.05, 0.25, 'float'],
                                      "interest_rate": [0.0, 0.1, 'float'],
                                      "employment_rate_initial": [0.6, 0.95, 'float']}
        self.parameter_names = list(self.parameter_arrangement.keys())
        
        """ Collect randomly chosen parameter samples"""
        self.parameter_samples = {}
        self.create_samples()
        
        
        """ Define Morris distances"""
        self.morris_distances = {"productivity_growth": 0.05,
                                 "interest_rate": 0.01,
                                 "employment_rate_initial": 0.05} 
        
        """ Prepare output data structure"""
        self.morris_values = {"productivity_growth": [],
                              "interest_rate": [],
                              "employment_rate_initial": []} 

    def create_samples(self):
        """
        Method for the creation of samples for all dimensions.
        In this case, we do not check for orthogonality.

        Returns
        -------
        None.

        """

        """ create candidate samples"""
        for param in self.parameter_names:
            arrangement = self.parameter_arrangement[param]
            if arrangement[2] == 'int':
                sample = np.random.randint(arrangement[0],
                                           arrangement[1],
                                           size=self.sample_size)
            elif arrangement[2] == 'float':
                sample = np.random.uniform(arrangement[0],
                                           arrangement[1],
                                           size=self.sample_size)
            self.parameter_samples[param] = sample

        
    def run(self):
        """
        Method for performing the Method of Morris.

        Returns
        -------
        None.

        """
        
        """ Loop over Morris trajectories"""
        for i in range(self.sample_size):
            
            """ Simulation for the starting point of the Morris trajectory""" 
            S = Simulation_Model(productivity_growth=self.parameter_samples["productivity_growth"][i],
                                 interest_rate=self.parameter_samples["interest_rate"][i],
                                 employment_rate_initial=self.parameter_samples["employment_rate_initial"][i],
                                 )
            S.run()
            results = S.return_results()
            goodness = goodness_euclidian(results)
            
            """ Loop over the parameters, changing each one in sequence"""
            for parameter in self.parameter_names:
                """ Change parameter"""
                self.parameter_samples[parameter][i] += self.morris_distances[parameter]
                """ Simulation for the next step in the Morris trajectory"""
                S = Simulation_Model(productivity_growth=self.parameter_samples["productivity_growth"][i],
                                 interest_rate=self.parameter_samples["interest_rate"][i],
                                 employment_rate_initial=self.parameter_samples["employment_rate_initial"][i],
                                 )
                S.run()
                results = S.return_results()
                new_goodness = goodness_euclidian(results)
                """ Calculate the impact""" 
                impact = np.abs(goodness - new_goodness)
                
                """ Record impact"""
                self.morris_values[parameter].append(impact)
                
                """ Reset comparison variable for the next Morris step"""
                goodness = new_goodness
                
        print() #print an empty line
        """ Print output (mean impact) for each variable"""
        for parameter in self.parameter_names:
            morris_result = np.mean(self.morris_values[parameter])
            print("Impact of the parameter '{0:s}' is on average: {1:.3f}".format(parameter, morris_result))

def goodness_euclidian(simulation_history):
    """
    Function for computing the goodness from Simulation_Model results.
    The goodness here the euclidian distance of average values of all 
    three dependent variables from target values in the second half of 
    the simulation. The first half may contain a transient (before the 
    trajectory settles into a stable situation), so the result may be 
    better without that part.

    Parameters
    ----------
    simulation_history : dict
        Recorded data on the simulation run.
    
    Returns
    -------
    goodness : float
        Euclidian distance of the average simulated values from target 
        values
    
    """
    
    """ Target average employment rate (for EU 2023: unemployment rate 
        of 7%, thus 1 - 0.07 = 0.93)"""
    tv = 0.93
    """ Target average wage share (for EU 2021: 197m employees times 
        average wage 33500 Euros divided by total GDP of 14.5 trillion 
        Euros (data from OECD):  
        197 * 33500 / (14.5 * 1000000) = 0.455)"""
    tw = 0.455
    """ Target average banking share (for EU 2021 GDP of 14.5 trillion, 
        of which are financial and insurance activities (ISIC rev.4 
        section K) 600bn Euros and real estate activities (ISIC rev.4 
        section L) 1.4 trillion Euros (data from OECD), thus:
        (600 + 1400) / 14500 = 0.138)"""
    td = 0.138
    
    """Half the simulation runtime"""
    n = len(simulation_history["history_t"]) // 2
    
    """ Averages in the second half of the simulation (removing the 
        potentially transient first half"""
    avg_employment_rate = np.mean(simulation_history["history_v"][-n:])
    avg_wage_share = np.mean(simulation_history["history_w"][-n:])
    avg_banking_share = np.mean(simulation_history["history_d"][-n:])
    
    """ Euclidian distance from target"""
    goodness = -((tv - avg_employment_rate)**2 + \
                                (tw - avg_wage_share)**2 + \
                                (td - avg_banking_share)**2)**0.5
    return goodness

""" Main entry point"""
if __name__ == '__main__':
    MM = MorrisMethod()
    MM.run()