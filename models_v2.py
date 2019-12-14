import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from shapely.geometry import Point, LineString
import math
import time

plt.rcParams['animation.writer'] = 'ffmpeg'
MAX_DENSITY = 7  #7 people per m for the road

# define personal speed by age, need to be improved


def get_death_rate(crowd_index):
    if crowd_index >= 1:
        return 0.01
    else:
        return 0


def get_speed(speed, crowd_index):
    """

    speed is influenced by crowd index
    :param speed: float
    :param crowd_index: float
    :return: the true speed affected by crowds: float
    """
    if crowd_index <= 1:
        return speed
    else:
        return math.pow(0.5, (crowd_index - 1)) * speed


def fill_phantom(g):
    for e in g.edges(keys=True, data=True):
        if 'geometry' not in e[3]:
            u = e[0]
            v = e[1]
            e[3]['geometry'] = LineString(
                [Point(g.node[u]['x'], g.node[u]['y']), Point(g.node[v]['x'], g.node[v]['y'])])
    return g


def adjust_length(g):
    for e in g.edges(keys=True, data=True):
        if 'geometry' in e[3]:
            e[3]['length'] = e[3]['geometry'].length
    return g


def clean(g):
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

    def get_crowd_index(self, people=None):
        """
        the function is to get the crowd index of a road, if there is no external people input,
        it will calculate the self.people

        expand: in real situation, two directions would be considered, simplified here
        :param people: list or dic
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

    def move(self, people=None, timestamp=1):
        if people is None:
            people = self.people
        self.crowd_index = self.get_crowd_index()
        crowd_index = self.crowd_index

        # four situation:
        # 1. Still on the road
        # 2. Transform to the next road
        # 3. Arrive the Refuge
        # 4. Die from stomping because of crowd
        leave_list = []
        for p in people:
            person = people[p]
            death_rate = get_death_rate(crowd_index)
            random_death = np.random.uniform(0,1)
            if random_death < death_rate:
                leave_list.append(person)
                self.fixed_list.append((person, time.time()))
                person.status = 4  # person is dead because of too many people
                continue
            speed = get_speed(person.speed, crowd_index)
            distance = timestamp * speed
            person.new_pos = person.pos + distance
            # still on the road
            if person.new_pos < self.length:
                person.pos = person.new_pos
                person.status = 1  # keep evacuating
            # go to the next road or arrived
            else:
                person.pnode = self.node_j
                leave_list.append(person)
                if not person.ifArrived():
                    # strategy 1: try to find a less crowded road  0: go for straight
                    if person.strategy == 1:
                        person.find_next()
                    else:
                        person.easy_find_next()
        for person in leave_list:
            self.people.pop(person.pid)


class Refuge:
    def __init__(self, graph, rid, rnode, road=None, arrive_list=None):
        self.graph = graph
        self.rid = rid
        self.rnode = rnode
        self.road = road
        if arrive_list is None:
            self.arrive_list = {}

    def show(self):
        return self.road.geom.coords[0]

    def add_person(self, person):
        person.road = self.road
        person.road.fixed_list.append((person, time.time()))
        person.status = 2
        person.pos = 0
        self.arrive_list[person.pid] = person


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
        self.isArrived = False
        self.pref = np.random.uniform(0.5, 2)
        self.refuge = None
        self.status = 0
        # 0:'Begin_To_Evacuate' -> 1:'Evacuating' -> 2:'Arrived'
        # 3:'No_Where_To_Go' 4:'Dead_From_Stomping'

    def ifArrived(self, target=None):
        """
        find if the person arrive a specific target place. the default is the person's target refuge
        :param target: Refuge
        :return: Boolean
        """
        if target is None:
            target = self.refuge
        if self.pnode == target.rnode:
            target.add_person(self)
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
        if graph is None:
            graph = self.graph
        if targets is None:
            targets = self.targets
        if targets is None or len(targets) == 0:
            self.status = 3
            print('No Destination found')
            return

        ans = [float('inf'), None, None]  # length, target, path
        for t in targets:
            refuge = targets[t]
            if nx.has_path(graph, self.pnode, refuge.rnode):
                paths = nx.shortest_path(graph, self.pnode, refuge.rnode)
                for i in range(len(paths) - 1):
                    l = graph[paths[i]][paths[i + 1]][0]['length']
                    if l < ans[0]:
                        ans[0] = l
                        ans[1] = refuge
                        ans[2] = paths
        if ans[1] is None:
            self.status = 3  # No where to go
        else:
            self.route = ans[2]
            self.refuge = ans[1]
            self.route_length = ans[0]

    def find_next(self):
        """
        find the next road for a person.
        :return: person has a new road, or person go to the status 3 "No where to go"
        """
        # two situations that need to find the next road
        # 1. on the plan way: find the next road
        # 2. the road is too crowded, find another road
        if self.status >= 2:
            return
        else:
            self.route = self.route[1:]
            if len(self.route) > 1:
                next_road = self.roads[(self.route[0], self.route[1],0)]
                # 1. keep on the plan
                if next_road.crowd_index < self.pref:
                    self.road = next_road
                    self.pos = 0
                    self.road.people[self.pid] = self
                    self.status = 1
                    return
                # 2. Try to find another road
                else:
                    tmp = [self.route, self.route_length]
                    g2 = self.graph.remove_edge(next_road.node_i, next_road.node_j)
                    self.get_target(graph=g2,targets=[self.refuge])
                    # successfully find another path
                    if self.status <= 1:
                        other_road = self.roads[(self.route[0], self.route[1],0)]
                        if other_road.crowd_index < next_road.crowd_index:
                            self.graph = g2
                            self.road = other_road
                            self.pos = 0
                            self.road.people[self.pid] = self
                            self.status = 1
                            return
                        # the other road is even more crowded, keep the plan
                        else:
                            self.route =tmp[0]
                            self.road = next_road
                            self.pos = 0
                            self.road.people[self.pid] = self
                            self.status = 1
                            return
                    # no other road, choose the origin road,
                    # route and refuge not change
                    else:
                        self.road = next_road
                        self.pos = 0
                        self.road.people[self.pid] = self
                        self.status = 1
                        return

    def easy_find_next(self):
        """
        find the next road for a person, person would only follow the origin rule
        :return: person has a new road, or person go to the status 3 "No where to go"
        """
        # two situations that need to find the next road
        # 1. on the plan way: find the next road
        # 2. the refuge is full, find the path to another refuge
        if self.status >= 2:
            return
        else:
            self.route = self.route[1:]
            if len(self.route) > 1:
                next_road = self.roads[(self.route[0], self.route[1], 0)]
                # 1. keep on the plan
                self.road = next_road
                self.pos = 0
                self.road.people[self.pid] = self
                self.status = 1

    def stay(self):
        for e in self.roads:
            if e[0] == self.pnode:
                self.road = self.roads[e]
                self.pos = 0
                self.road.fixed_list.append((self, time.time()))
                break

    def xy(self):
        return self.road.geom.interpolate(self.pos).coords[0]


class Model:
    def __init__(self, graph, num_people, num_refuge, strategy):
        self.graph = graph
        self.isfinished = False
        self.strategy = strategy   # 0 or 1
        self.nodes = self.graph.nodes() # {id:{'x':1，'y':1, 'osmid':1}}
        self.edges = self.graph.edges(keys=True, data=True) #{(u,v,k):{info}}
        self.roads ={}
        self.people ={}
        self.refuges = {}
        # initial roads dic {(u,v,k):Road()}
        for e in self.edges:
            info = e[3]
            r = Road(self.graph, e[0], e[1], info)
            self.roads[(e[0], e[1], e[2])] = r
        for e in self.roads:
            self.roads[e].roads = self.roads
        i = 0
        j = 0
        # initialize people dic {id:Person()} and refuges dic {id: Refuge()}
        for r in np.random.choice(list(self.nodes.keys()), num_refuge):
            self.refuges[j] = Refuge(self.graph, j, r)
            j += 1
        # assign road to refuge
        for r in self.refuges:
            refuge = self.refuges[r]
            for e in self.roads:
                if refuge.rnode == e[0]:
                    refuge.road = self.roads[e]
                    break

        for p in np.random.choice(list(self.nodes.keys()), num_people):
            self.people[i] = Person(self.graph, i, p, self.roads, strategy=self.strategy, targets=self.refuges)
            i += 1

    def find_refuge(self, people=None):
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
            if person.refuge and person.route:
                # person is generated at the exactly place as the refuge,
                # assign it to a road
                if len(person.route) <= 1:
                    person.refuge.add_person(person)
                # assign start road to person
                else:
                    person.road = self.roads[(person.route[0], person.route[1], 0)]
                    person.road.people[person.pid] = person
                all_routes.append(person.route)
            else:
                person.stay()

        fig, ax = ox.plot_graph_routes(self.graph, all_routes, fig_height=12)

    def move(self, timesteps=1):
        for r in self.roads.values():
            r.move(timestamp=timesteps)

    def record(self, t1, file):
        d = {0: 'Not Begin', 1: 'Evacuating', 2:'Arrived', 3:'No Place To Go', 4:'Dead By Crowd'}
        columns = ['id', 'status', 'coord_x', 'coord_y', 'time']
        df = pd.DataFrame(columns=columns)
        df = df.set_index('id')
        for r in self.roads.values():
            if len(r.fixed_list) > 0:
                for (person, t) in r.fixed_list:
                    coords = person.xy()
                    df.loc[person.pid] = [d[person.status], coords[0], coords[1], t - t1]
        df.to_csv(file)

    def show(self, ax):
        return ax.scatter(*zip(*[self.people[p].xy() for p in self.people]))

    def show_refuge(self, ax):
        return ax.scatter(*zip(*[self.refuges[r].show() for r in self.refuges]), c='r')

    def run(self, nsteps=None, save_args=None):
        t1 = time.time()
        save = True
        self.find_refuge()
        if save_args is None or len(save_args) < 3:
            save = False
            for i in range(nsteps):
                self.move()
                self.show(save_args[1])
                fig, ax = plt.plot()
        else:
            fig, ax, filename = save_args
            self.show_refuge(ax)
            r = self.show(ax)

            def runFrame():
                i = 0
                while not self.isfinished:
                    i += 1
                    yield i

            def update(frame_number):
                if self.isfinished:
                    return
                # self.relocate()
                self.move()
                r.set_offsets([self.people[p].xy() for p in self.people])
                if not any(self.people[p].status < 2 for p in self.people):
                    self.isfinished = True
                    print('Evacuation completed at time: %d' % frame_number)

            if nsteps is None:
                frame = runFrame
            else:
                frame = nsteps
            anim = animation.FuncAnimation(fig, update, frames=frame, interval=100, save_count=1000)
            vid_file = filename + '.mp4'
            csv_file = filename + '.csv'
            anim.save(vid_file)
            self.record(t1, csv_file)
            return HTML('<video width="800" controls><source src="%s" type="video/mp4"></video>' % filename)











