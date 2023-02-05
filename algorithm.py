import argparse
from time import perf_counter, localtime
import pandas as pd
import folium

import numpy as np


def calc_distance(x1: float, y1: float, x2: float, y2: float):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calc_route_len(route: np.ndarray, distances: np.ndarray):
    route_distances = [
        distances[node, route[(idx + 1) % distances.shape[0]]]
        for idx, node in enumerate(route)
    ]
    return sum(route_distances)


def aco(
    distances: np.ndarray,
    generations: int = 500,
    ants: int = 40,
    alpha: int = 1,
    beta: int = 2,
    vapour: float = .1,
    entry: int = 0,
) -> np.ndarray:
    size = distances.shape[0]
    pheromone = np.ones((size, size))

    ant_routes = np.zeros((ants, size)).astype(int)
    route_lengths = np.zeros(ants)

    best_routes = []
    best_lengths = []

    gen_space = len(str(generations))

    for generation in range(generations):
        probs = (pheromone ** alpha) * (1 / distances ** beta)

        # Route construction
        for ant, ant_route in enumerate(ant_routes):
            ant_route[0] = entry

            for next_node in range(1, size):
                visited = set(ant_route[:next_node])
                unvisited = list(set(range(size)) - visited)

                prob = probs[ant_route[next_node - 1], unvisited]
                prob = prob / prob.sum()

                node = np.random.choice(unvisited, size=1, p=prob)[0]
                ant_route[next_node] = node

        # Calculation of route lengths
        route_lengths = np.array([
            calc_route_len(route, distances) for route in ant_routes
        ])

        # Setting the best route of generation
        best_idx = route_lengths.argmin()
        best_route = ant_routes[best_idx, :].copy()
        best_length = route_lengths[best_idx]

        best_routes.append(best_route)
        best_lengths.append(best_length)

        # Updating pheromones
        pheromone = (1 - vapour) * pheromone
        for ant, ant_route in enumerate(ant_routes):
            for node in range(size):
                n1, n2 = ant_route[node % size], ant_route[(node + 1) % size]
                route_length = 1 / route_lengths[ant]
                pheromone[n1, n2] += route_length

        # Log
        print(
            f'Generation: {generation:{gen_space}}',
            '|',
            f'Best route length: {best_length:{12}.{10}}'
        )

    best_generation = np.array(best_lengths).argmin()

    return best_routes[best_generation]


def tsp(data: pd.DataFrame, coord1: str, coord2: str, **options) -> np.ndarray:
    size = data.shape[0]

    distances = np.zeros((size, size))
    for (x, y), _ in np.ndenumerate(distances):
        point1, point2 = data.iloc[x], data.iloc[y]
        distances[x, y] = calc_distance(
            point1[coord1],
            point1[coord2],
            point2[coord1],
            point2[coord2],
        )

    start = perf_counter()
    route = aco(distances, **options)
    print(f'Finished in {perf_counter() - start}s')

    return route


def draw_map(data: pd.DataFrame, route: np.ndarray) -> folium.Map:
    world_map = folium.Map(
        location=[64.0914, 101.6016],
        tiles='Stamen Toner',
        zoom_start=3
    )

    for index, node in enumerate(route):
        next_node = route[(index + 1) % len(route)]

        loc1 = data.iloc[node]
        loc2 = data.iloc[next_node]

        locations = [[
            loc1['latitude_dd'] / 100,
            loc1['longitude_dd'] / 100,
        ], [
            loc2['latitude_dd'] / 100,
            loc2['longitude_dd'] / 100,
        ]]
        folium.PolyLine(
            locations,
            radius=1000,
        ).add_to(world_map)

    for index, row in data.iterrows():
        folium.Circle(
            radius=100,
            location=[
                row['latitude_dd'] / 100,
                row['longitude_dd'] / 100
            ],
            popup=row['settlement'] + ' ' + str(index),
            color='crimson',
            fill=True,
        ).add_to(world_map)

    return world_map


def main():
    parser = argparse.ArgumentParser(prog='TSP-ACO')
    parser.add_argument('generations', type=int, nargs='?', default=200)
    parser.add_argument('ants', type=int, nargs='?', default=30)
    parser.add_argument('entries', type=int, nargs='?', default=40)
    parser.add_argument('offset', type=int, nargs='?', default=3753)

    args = parser.parse_args()

    ants = args.ants
    generations = args.generations
    entries = args.entries
    offset = args.offset

    data = pd.read_csv('./data.csv', delimiter=',', index_col='id')
    data = data[offset: offset + entries]
    route = tsp(data, 'latitude_dd', 'longitude_dd',
                ants=ants, generations=generations)

    world_map = draw_map(data, route)

    save_time = localtime()
    save_time = f'{save_time[3]:02}.{save_time[4]:02}.{save_time[5]:02}'
    map_name = f'{data.shape[0]}.{ants}.{generations}-{save_time}.html'

    print('Map saved as ' + map_name)
    world_map.save(map_name)


if __name__ == '__main__':
    main()
