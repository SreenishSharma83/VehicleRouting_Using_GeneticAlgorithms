"""
Solve the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) using DEAP for optimization.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import graphviz as gv
import yaml
from ortools.constraint_solver import pywrapcp as cp
from ortools.constraint_solver.routing_enums_pb2 import (
    FirstSolutionStrategy,
    LocalSearchMetaheuristic,
)
from ortools.constraint_solver.routing_parameters_pb2 import RoutingSearchParameters
from deap import base, creator, tools

def main() -> None:
    """Entry point of the program."""

    # Parse command line arguments
    args = _parse_args()

    # Load a problem
    data = _load_problem(args.path)

    # Create a Routing Index Manager and Routing Model
    manager = cp.RoutingIndexManager(data.num_locations, data.num_vehicles, data.depot)
    routing = cp.RoutingModel(manager)

    # Define weights of edges
    _set_edge_weights(routing, manager, data)

    # Add capacity constraints
    _add_capacity_constraints(routing, manager, data)

    # Add time window constraints
    _add_time_window_constraints(routing, manager, data)

    # Configure routing search parameters
    search_params: RoutingSearchParameters = cp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = args.time_limit
    if args.gls:
        search_params.local_search_metaheuristic = LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    if args.verbose:
        search_params.log_search = True

    # Solve the problem to get initial solution
    assignment: cp.Assignment = routing.SolveWithParameters(search_params)
    if not assignment:
        print("No initial solution found.")
        return

    # Generate initial route
    initial_solution = _get_initial_solution(data, routing, manager, assignment)

    # Set up DEAP
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, initial_solution)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function, data=data)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)

    # Evolve the population
    for gen in range(100):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    # Extract and print the best solution
    fits = [ind.fitness.values[0] for ind in population]
    best_idx = fits.index(min(fits))
    best_solution = population[best_idx]

    print("Best solution:", best_solution)

    # Print the initial solution
    _print_solution(data, routing, manager, assignment)

def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", help="JSON or YAML file that represents a vehicle routing problem.")
    parser.add_argument("-t", "--time-limit", type=int, default=30)
    parser.add_argument("--gls", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()

@dataclass(frozen=True)
class Problem:
    weights: list[list[int]]
    service_times: list[int]
    demands: list[int]
    time_windows: list[list[int]]
    max_time: int
    vehicle_capacities: list[int]
    depot: int

    @property
    def num_locations(self) -> int:
        return len(self.time_windows)

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicle_capacities)

def _load_problem(path: str) -> Problem:
    """Load the data for the problem from path."""
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return Problem(**data)

def _set_edge_weights(routing: cp.RoutingModel, manager: cp.RoutingIndexManager, data: Problem) -> None:
    """Set weights of edges defined in the problem to the routing model."""
    def _weight_callback(from_index: int, to_index: int) -> int:
        from_node: int = manager.IndexToNode(from_index)
        to_node: int = manager.IndexToNode(to_index)
        return data.weights[from_node][to_node]
    weight_callback_index: int = routing.RegisterTransitCallback(_weight_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(weight_callback_index)

def _add_capacity_constraints(routing: cp.RoutingModel, manager: cp.RoutingIndexManager, data: Problem) -> None:
    """Add capacity constraints defined in the problem to the routing model."""
    def _demand_callback(from_index: int) -> int:
        from_node: int = manager.IndexToNode(from_index)
        return data.demands[from_node]
    demand_callback_index: int = routing.RegisterUnaryTransitCallback(_demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        slack_max=0,
        vehicle_capacities=data.vehicle_capacities,
        fix_start_cumul_to_zero=True,
        name="Capacity",
    )

def _add_time_window_constraints(routing: cp.RoutingModel, manager: cp.RoutingIndexManager, data: Problem) -> None:
    """Add time window constraints defined in the problem to the routing model."""
    def _time_callback(from_index: int, to_index: int) -> int:
        from_node: int = manager.IndexToNode(from_index)
        to_node: int = manager.IndexToNode(to_index)
        serv_time = data.service_times[from_node]
        trav_time = data.weights[from_node][to_node]
        return serv_time + trav_time

    time_callback_index: int = routing.RegisterTransitCallback(_time_callback)
    horizon = data.max_time
    routing.AddDimension(
        time_callback_index,
        slack_max=horizon,
        capacity=horizon,
        fix_start_cumul_to_zero=False,
        name="Time",
    )
    time_dimension: cp.RoutingDimension = routing.GetDimensionOrDie("Time")
    for loc_idx, (open_time, close_time) in enumerate(data.time_windows):
        index: int = manager.NodeToIndex(loc_idx)
        time_dimension.CumulVar(index).SetRange(open_time, close_time)

def _get_initial_solution(data: Problem, routing: cp.RoutingModel, manager: cp.RoutingIndexManager, assignment: cp.Assignment):
    """Extract the initial solution as a list of routes."""
    routes = []
    for vehicle_id in range(data.num_vehicles):
        route = []
        index: int = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index: int = manager.IndexToNode(index)
            route.append(node_index)
            next_var: cp.IntVar = routing.NextVar(index)
            index = assignment.Value(next_var)
        routes.append(route)
    return routes

def create_individual(initial_route):
    """Create a DEAP individual from an initial route."""
    return creator.Individual(initial_route)

def fitness_function(route, data):
    """Calculate the total cost of the route."""
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += data.weights[route[i]][route[i + 1]]
    return total_cost,

def mutate(individual):
    """Mutate a route by swapping two nodes."""
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual,

def crossover(ind1, ind2):
    """Perform order crossover between two routes."""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size - 1)
    cxpoint2 = random.randint(1, size - 1)

    if cxpoint1 > cxpoint2:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    temp1 = ind1[cxpoint1:cxpoint2]
    temp2 = [item for item in ind2 if item not in temp1]
    ind1[:] = temp1 + temp2
    ind2[:] = [item for item in ind1 if item not in temp2]
    return ind1, ind2

def _print_solution(data: Problem, routing: cp.RoutingModel, manager: cp.RoutingIndexManager, assignment: cp.Assignment) -> None:
    """Print the solution."""
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    time_dimension = routing.GetDimensionOrDie("Time")
    print("Solution:")
    for vehicle_id in range(data.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = assignment.Value(routing.NextVar(index))
        print(f"Vehicle {vehicle_id} route: {route}")
        print(f"Capacity used: {capacity_dimension.CumulVar(routing.Start(vehicle_id)).Value()}")
        print(f"Time: [{time_dimension.MinVar(routing.Start(vehicle_id)).Value()}, {time_dimension.MaxVar(routing.Start(vehicle_id)).Value()}]")

if __name__ == "__main__":
    main()
