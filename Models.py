import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from shapely.geometry import Point, LineString
import math
import copy


plt.rcParams['animation.writer'] = 'ffmpeg'
MAX_DENSITY = 2  # the max capacity of a 100m road is 200 person
MAX_CAPACITY = 300  # the max capacity of a refuge

"""
Evacuation Simulation model 
By: Zhesong Wu, Contact: zhesong2@illinois.edu

Classes Overview

1. Road:
    Properties:
    
    Methods
2. Person:
    Properties:
    
    Methods

3. Refuge:
    Properties:
    
    Methods

4. Model:
    Properties:
    
    Methods
"""


def get_death_rate(crowd_index):
    """
    get the death rate from stomping for a road, if a road is twice more crowded than
    its max capacity, we assign a 5% death rate from stomping
    :param crowd_index:
    :return:
    """
    if crowd_index >= 2:
        return 0.05
    else:
        return 0


def get_speed(speed, crowd_index):
    """
    get the speed of a person by his fixed speed and the crowd index

    :param speed: float
    :param crowd_index: float
    :return: the true speed affected by crowds: float
    """
    if crowd_index <= 1:
        return speed
    else:
        return math.pow(0.5, (crowd_index - 1)) * speed


def fill_phantom(g):
    """
    add geometry information if there is no geometry tag
    :param g: Graph
    :return: g: Graph
    """
    for e in g.edges(keys=True, data=True):
        if 'geometry' not in e[3]:
            u = e[0]
            v = e[1]
            e[3]['geometry'] = LineString(
                [Point(g.node[u]['x'], g.node[u]['y']), Point(g.node[v]['x'], g.node[v]['y'])])
    return g


def adjust_length(g):
    """
    calculate the length by geometry, to avoid inconsistency
    :param g: Graph
    :return: g: Graph
    """
    for e in g.edges(keys=True, data=True):
        if 'geometry' in e[3]:
            e[3]['length'] = e[3]['geometry'].length
    return g


def clean(g):
    """
    clean up a graph, fill the geometry information for every edge
    :param g: Graph
    :return:
    """
    return adjust_length(fill_phantom(g))


class Road:
    def __init__(self, graph, node_i, node_j, info):
        self.graph = graph
        self.node_i = node_i
        self.node_j = node_j
        self.info = info
        self.geom = self.info['geometry']
        self.length = self.geom.length
        self.people = {}
        self.requests = []
        self.crowd_index = 0
        self.fixed_list = []
        self.refuge = None

    def add_person(self, person, pos=0):
        """
        add a person to the road with a relative position
        :param person: Person
        :param pos: float
        :return:
        """
        person.road = self
        self.people[person.pid] = person
        person.pos = pos

    def get_crowd_index(self, people=None):
        """
        get the crowd index for a road, basically the people number / road max capacity

        expand: in real situation, two directions would be considered, simplified here
        :param people: dic
        :return: the crowd index of the road: float
        """
        if people is None:
            people = self.people
        max_capacity = MAX_DENSITY * self.length
        my_people_len = len(people)

        # other_road = self.roads[(self.node_j, self.node_i, 0)]
        # other_people_len = len(other_road.people)
        # return (my_people_len + other_people_len)/max_capacity
        return my_people_len / max_capacity

    def move(self, people=None, timestep=1):
        """
        move all the people in the road forward for specific timestep
        :param people: dict
        :param timestep: int
        :return:
        """
        if people is None:
            people = self.people
        # six situation:
        # 1. Die from stomping
        # 2. Still on the road
        # 3. Successfully arrive
        # 4. Arrived but full, find another raod
        # 5. Arrived but full, cannot find another road, no where to go
        # 6. Transit to Another road, may change route if using strategy 1
        crowd_index = self.crowd_index
        death_rate = get_death_rate(crowd_index)
        leave_list = []
        for p in people:
            person = people[p]
            # 1. dead by stomping
            # death rate by stomping is calculated from the crowd index of the road.
            # when the crowd rate > 3, I assume the death rate is 0.01
            # generate a random number for each person in the road, if the number less than 0.01,
            # the people would die because of stomping
            random_death = np.random.uniform(0, 1)
            if random_death < death_rate:
                leave_list.append(p)
                self.fixed_list.append((person, -1))
                person.status = 4  # person is dead because of too many people
                # print(person.pid, ' is death because of stomping on the place', person.xy())
                continue
            # the speed of a person is calculated from its fixed speed and crowd index of the road
            speed = get_speed(person.speed, crowd_index)
            distance = timestep * speed
            person.new_pos = person.pos + distance
            # 2. Still on the road, keep evacuating
            if person.new_pos < self.length:
                person.pos = person.new_pos
                person.status = 1
                continue
            else:
                person.pnode = self.node_j
                person.new_pos = 0
                leave_list.append(p)
                # 3. Arrived
                if person.ifArrived():
                    person.refuge.add_person(person)
                # 4, 5, 6 situation
                else:
                    # strategy 1: flexible-plan person dynamically adjust his plan
                    if person.strategy == 1:
                        person.find_next()
                    # strategy 2: fixed_plan person always stick to its origin plan
                    else:
                        person.easy_find_next()
        for p in leave_list:
            self.people.pop(p)


class Refuge:
    def __init__(self, graph, rid, rnode, road=None, capacity=MAX_CAPACITY, arrive_list=None):
        self.graph = graph
        self.rid = rid
        self.rnode = rnode
        self.capacity = capacity
        self.road = road
        self.isFull = False
        if arrive_list is None:
            self.arrive_list = {}

    def show(self):
        """
        get the refuge's coordinates
        :return:
        """
        return self.road.geom.coords[0]

    def add_person(self, person):
        """
        add a person into the refuge
        :param person:
        :return:
        """
        person.road = self.road
        person.road.fixed_list.append((person, -1))
        person.status = 2
        person.pos = 0
        self.arrive_list[person.pid] = person
        if len(self.arrive_list) == self.capacity:
            self.isFull = True


class Person:
    def __init__(self, graph, pid, pnode, roads, strategy, road=None, pos=0, targets=None):
        self.graph = graph
        self.pid = pid
        self.pnode = pnode
        self.roads = roads
        self.road = road
        self.pos = pos
        self.new_pos = pos
        self.strategy = strategy
        self.route = None
        self.targets = targets
        if self.targets is None:
            self.targets = {}
        self.speed = np.random.uniform(4, 5)
        self.pref = np.random.uniform(0.5, 2)
        self.refuge = None
        self.route = None
        self.status = 0
        # 0:'Begin_To_Evacuate' -> 1:'Evacuating' -> 2:'Arrived'
        # 3:'No_Where_To_Go' 4:'Dead_From_Stomping'

    def ifArrived(self, refuge=None):
        """
        find if the person arrive a specific target place. the default is the person's target refuge
        :param refuge: Refuge
        :return: Boolean
        """
        if refuge is None:
            refuge = self.refuge
        if refuge is None:
            return False
        if self.pnode == refuge.rnode and not self.refuge.isFull:
            return True
        else:
            return False

    def get_target(self, graph=None, targets=None):
        """
        get the person's target refuge, and its route,
        if no refuge found or no routes found, person status = 3 'No where to go'
        :param graph:
        :param targets:
        :return: route: list, refuge: Refuge
        """
        # 3 situations
        if graph is None:
            graph = self.graph
        if targets is None:
            targets = self.targets
        # 1. no targets to go
        if targets is None or len(targets) == 0:
            self.status = 3
            # print('The person ', self.pid, 'has no where to go.')
            return
        ans = [float('inf'), None, None]  # length, target, path
        for t in targets:
            refuge = targets[t]
            if nx.has_path(graph, self.pnode, refuge.rnode):
                paths = nx.shortest_path(graph, self.pnode, refuge.rnode)
                length = 0
                for i in range(len(paths) - 1):
                    length += graph[paths[i]][paths[i + 1]][0]['length']
                if length < ans[0]:
                    ans[0] = length
                    ans[1] = refuge
                    ans[2] = paths
        # 2. has target, but has no way to arrive
        if ans[1] is None:
            self.status = 3
            # print('The person ', self.pid, 'has no where to go.')
        # 3. find the way
        else:
            self.route = ans[2]
            self.refuge = ans[1]
            # print('The person ', self.pid, 'finds the way to the refuge ', self.refuge.rid,
            #      ' And he has ', len(self.targets), ' in his plan')

    def find_next(self):
        """
        find the next road for a person, people would use strategy 1: flexible plan
        :return: person has a new road, or person go to the status 3 "No where to go"
        """
        # three situations that need to find the next road
        # 4. Arrived but full, find another road
        # 5. Arrived but full, cannot find another road, no where to go
        # 6. Transit to Another road, may change route if using strategy 1
        self.route = self.route[1:]
        # arrive but is full
        if len(self.route) <= 1:
            # print(self.pid, ' has ', len(self.targets), 'refuges in the plan, the refuge',
            #      self.refuge.rid, 'is full')
            self.targets.pop(self.refuge.rid)
            self.refuge = None
            self.get_target()
            # 5. Arrived but full, no where to go
            if self.status == 3:
                self.stay()
            # 4. Arrived but full, find another route
            else:
                road = self.roads[(self.route[0], self.route[1], 0)]
                road.add_person(self, 0)
                self.status = 1

        # 6. Transit to Another road
        else:
            next_road = self.roads[(self.route[0], self.route[1],0)]
            # 6.1 Keep the origin road since it is not crowded
            if next_road.crowd_index < self.pref:
               next_road.add_person(self, 0)
               self.status = 1
            # 6.2 Change Plan since the next road is crowded
            else:
                tmp = self.route
                # person would no longer taking that crowded road into consideration
                g2 = self.graph.copy()
                g2 = g2.remove_edge(next_road.node_i, next_road.node_j)
                self.get_target(graph=g2, targets={self.refuge.rid: self.refuge})

                # successfully find another path
                if self.status != 3:
                    other_road = self.roads[(self.route[0], self.route[1],0)]
                    if other_road.crowd_index < next_road.crowd_index:
                        self.graph = g2
                        other_road.add_person(self,0)
                        self.status = 1
                        return
                else:
                    self.route = tmp
                    next_road.add_person(self,0)
                    self.status = 1
                    return

    def easy_find_next(self):
        """
        find the next road for a person, person would use strategy 0: fixed plan
        :return: person has a new road, or person go to the status 3 "No where to go"
        """
        # three situations that need to find the next road
        # 4. Arrived but full, find another road
        # 5. Arrived but full, cannot find another road, no where to go
        # 6. Transit to Another road, always stick to fix plan
        self.route = self.route[1:]
        # arrive but is full
        if len(self.route) <= 1:
            # print(self.pid, ' has ', len(self.targets), 'refuges in the plan, the refuge',
            #      self.refuge.rid, 'is full')
            self.targets.pop(self.refuge.rid)
            self.refuge = None
            self.get_target()
            # 5. Arrived but full, no where to go
            if self.status == 3:
                self.stay()
            # 4. Arrived but full, find another route
            else:
                road = self.roads[(self.route[0], self.route[1], 0)]
                road.add_person(self, 0)
                self.status = 1

        # 6. Transit to Another road
        else:
            next_road = self.roads[(self.route[0], self.route[1],0)]
            next_road.add_person(self, 0)
            self.status = 1

    def stay(self):
        """
        process the command that a individual would no longer move
        :return:
        """
        for e in self.roads:
            if e[0] == self.pnode:
                self.road = self.roads[e]
                self.pos = 0
                self.road.fixed_list.append((self, -1))
                self.status = 3
                break

    def xy(self):
        """
        get the coordinates for a individual
        :return: coordinates tuple
        """
        return self.road.geom.interpolate(self.pos).coords[0]


class Model:
    def __init__(self, graph, num_people, num_refuge, strategy, refuge_capacity=MAX_CAPACITY):
        """
        Initialize the model, with the following inputs:

        :param graph: the base graph, MultiDiGraph from osmnx
        :param num_people: int: number of people
        :param num_refuge: int: number of refuges
        :param strategy: int: 1 means a flexible-plan evacuate, 0 means a fixed-plan evacuate
        :param refuge_capacity: int: max capacity of refuge
        """
        self.graph = graph
        self.isfinished = False
        self.strategy = strategy    # 0 or 1
        self.nodes = self.graph.nodes()  # nodes dict: {id:{'x':1，'y':1, 'osmid':1}}
        self.edges = self.graph.edges(keys=True, data=True)  # edges list: [(u, v, k, info)]
        self.roads = {}
        self.people = {}
        self.refuges = {}
        # initial roads dict {(u,v,k):Road()}
        for e in self.edges:
            info = e[3]
            self.roads[(e[0], e[1], e[2])] = Road(self.graph, e[0], e[1], info)
        for e in self.roads:
            self.roads[e].roads = self.roads
        i = 0
        j = 0
        # initialize people dic {id:Person()} and refuges dic {id: Refuge()}
        for r in np.random.choice(list(self.nodes.keys()), num_refuge):
            self.refuges[j] = Refuge(self.graph, j, r, refuge_capacity)
            j += 1
        # assign road to refuge
        for r in self.refuges:
            refuge = self.refuges[r]
            for e in self.roads:
                if refuge.rnode == e[0]:
                    refuge.road = self.roads[e]
                    break

        for p in np.random.choice(list(self.nodes.keys()), num_people):
            self.people[i] = Person(self.graph, i, p, self.roads, self.strategy, targets=self.refuges.copy())
            i += 1

    def copy(self):
        """
        a deep copy of the origin model
        :return: Model
        """
        model2 = Model(self.graph, 0, 0, self.strategy)
        model2.roads = copy.deepcopy(self.roads)
        model2.people = copy.deepcopy(self.people)
        model2.refuges = copy.deepcopy(self.refuges)
        return model2

    def find_refuge(self, people=None, show_map=0):
        """
        for each person, assign the refuge, route, and road
        :param people:
        :return:
        """
        if people is None:
            people = self.people
        all_routes = []
        for p in people:
            person = people[p]
            person.get_target()  # find refuges and routes
            if person.refuge is not None:
                # person is generated at the exactly place as the refuge,
                if len(person.route) == 1:
                    person.refuge.add_person(person)
                else:
                    road = self.roads[(person.route[0], person.route[1], 0)]
                    pos = road.length * np.random.uniform(0, 1)
                    road.add_person(person, pos)
                    person.status = 1
                all_routes.append(person.route)
            else:
                person.stay()
        if show_map == 1:
            fig, ax = ox.plot_graph_routes(self.graph, all_routes, fig_height=12)
            self.show(ax)
            self.show_refuge(ax)

    def move(self, timesteps=1):
        """
        move all the person in the model in timesteps
        :param timesteps: int
        :return:
        """
        for r in self.roads.values():
            r.move(timestamp=timesteps)

    def record(self, file):
        """
        record each person's status, id, location, and time when one finishes evacuating,
        then output it in a csv file
        :param file: str
        :return: output csv file
        """
        d = {0: 'Not Begin', 1: 'Evacuating', 2:'Arrived', 3:'No Place To Go', 4:'Dead By Crowd'}
        columns = ['id', 'status', 'coord_x', 'coord_y', 'time']
        df = pd.DataFrame(columns=columns)
        df = df.set_index('id')
        for r in self.roads.values():
            if len(r.fixed_list) > 0:
                for (person, t) in r.fixed_list:
                    coords = person.xy()
                    df.loc[person.pid] = [d[person.status], coords[0], coords[1], t]
        df.to_csv(file)

    def cal_index(self):
        """
        calculate each road's crowd index
        :return:
        """
        for road in self.roads.values():
            road.crowd_index = road.get_crowd_index()

    def assign_time(self, time):
        """
        record the time of each individual when one finish evacuating
        :param time: int
        :return:
        """
        for road in self.roads.values():
            nl = []
            for pair in road.fixed_list:
                if pair[1] == -1:
                    pair = (pair[0], time)
                nl.append(pair)
            road.fixed_list = nl

    def show(self, ax):
        """
        show individuals' location
        :param ax:
        :return:
        """
        return ax.scatter(*zip(*[self.people[p].xy() for p in self.people]))

    def show_refuge(self, ax):
        """
        show refuges' location
        :param ax:
        :return:
        """
        return ax.scatter(*zip(*[self.refuges[r].show() for r in self.refuges]), c='r', s=200)

    def run(self, nsteps=None, save_args=None):
        """
        run the model, two modes of running: a fixed time frame, or a complete run
        :param nsteps: int
        :param save_args: tuple, (fig, ax, filename)
        :return:
        """
        self.find_refuge()
        # if no save_args, do not save
        if save_args is None or len(save_args) < 3:
            for i in range(nsteps):
                self.cal_index()
                self.move()
        else:
            # show the initial position
            fig, ax, filename = save_args
            self.show_refuge(ax)
            r = self.show(ax)

            def runFrame():
                """
                frame generator
                :return:
                """
                i = 0
                while not self.isfinished:
                    i += 1
                    yield i

            def update(frame_number):
                """
                frame update, in each frame, update all the information and plot positions
                :param frame_number:
                :return:
                """
                if self.isfinished:
                    return
                self.cal_index()
                self.move()
                self.assign_time(frame_number)
                # update individuals' position
                r.set_offsets([self.people[p].xy() for p in self.people])
                # finish the run
                if not any(self.people[p].status < 2 for p in self.people):
                    self.isfinished = True
                    print('Evacuation completed at time: %d' % frame_number)
            # 1. Mode 1: if nsteps exist: run a fixed number of time frames
            # 2. Mode 2: if no nsteps: complete run
            if nsteps is None:
                frame = runFrame
            else:
                frame = nsteps
            anim = animation.FuncAnimation(fig, update, frames=frame, interval=100, save_count=10000)
            vid_file = filename + '.mp4'
            csv_file = filename + '.csv'
            anim.save(vid_file)
            self.record(csv_file)
            return HTML('<video width="800" controls><source src="%s" type="video/mp4"></video>' % filename)











