import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import shapely
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.collections as collections
from IPython.display import HTML
import queue
from shapely.geometry import Point, LineString
import math

plt.rcParams['animation.writer'] = 'ffmpeg'
MAX_DENSITY = 7 #7 people per m for the road
MAX_CAPACITY = 100
# define personal speed by age, need to be improved


def get_death_rate(crowd_index):
    if crowd_index >= 3:
        return 0.01
    else:
        return 0


def get_speed(speed, crowd_index):
    """


    :param speed: float
    :param crowd_index: float
    :return: the true speed affected by crowds: float
    """
    if crowd_index <= 1:
        return speed
    else:
        return math.pow(0.5, (crowd_index - 1)) * speed


def avgspeed(age):
    if age<15:
        return 3
    elif age < 40:
        return 4
    elif age < 60:
        return 3
    else:
        return 2

def fillPhantom(g):
    for e in g.edges(keys=True, data=True):
        if 'geometry' not in e[3]:
            u = e[0]
            v = e[1]
            e[3]['geometry'] = LineString(
                [Point(g.node[u]['x'], g.node[u]['y']), Point(g.node[v]['x'], g.node[v]['y'])])
    return g


def adjustLength(g):
    for e in g.edges(keys=True, data=True):
        if 'geometry' in e[3]:
            e[3]['length'] = e[3]['geometry'].length
    return g


def cleanUp(g):
    return adjustLength(fillPhantom(g))


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
        notice: the road density has to take the inverse direction road into consideration
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

    def show(self, people=None):  # need to decide if necessary
        """
        get the real-time people location
        :param people: dic
        :return: the locations of people: geometrycollection
        """
        if people is None:
            people = self.people
        return shapely.geometry.GeometryCollection([self.geom] + [self.geom.interpolate(people[_].pos) for _ in people])

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
                self.fixed_list.append(person)
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
                person.pos = self.length
                leave_list.append(person)
                if not person.ifArrived():
                    person.find_next()
        for person in leave_list:
            self.people.pop(person.pid)


class Refuge:
    def __init__(self, graph, rid, rnode, road=None, capacity=MAX_CAPACITY, arrive_list=None):
        self.graph = graph
        self.rid = rid
        self.rnode = rnode
        self.capacity = capacity
        self.road = road
        if arrive_list is None:
            self.arrive_list = {}

    def show(self):
        return self.road.geom.coord[0]

class Person:
    def __init__(self, graph, pid, pnode, roads, road=None, pos=0, targets=None):
        self.graph = graph
        self.pid = pid
        self.pnode = pnode
        self.roads = roads
        self.road = road
        self.pos = pos
        self.new_pos = pos
        self.route = None
        self.targets = targets
        if self.targets is None:
            self.targets = {}
        self.speed = np.random.uniform(4,5)
        self.isArrived = False
        self.pref = np.random.uniform(0.5, 2)
        self.refuge = None
        self.status = 0 # 0:'Begin_To_Evacuate' -> 1:'Evacuating' -> 2:'Arrived' 3:'No_Where_To_Go' 4:'Dead_From_Stomping'

    def ifArrived(self, target=None):
        """
        find if the person arrive a specific target place. the default is the person's target refuge
        :param target: Refuge
        :return: Boolean
        """
        if target is None:
            target = self.refuge
        if self.pnode == target.rnode:
            if len(target.arrive_list) < target.capacity:
                self.road = target.road
                self.road.fixed_list.append(self)
                target.arrive_list[self.pid] = self
                self.status = 2
                return True
            else:
                self.targets.pop(target.rid)
                self.refuge = None

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
            targets = self.targets  #If no input targets, it would get the build-in targets group
        if targets is None or len(targets) == 0:
            self.status = 3
            print('No Destination found')

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
        # three situations that need to find the next road
        # 1. on the plan way: find the next road
        # 2. the road is too crowded, find another road
        # 3. the refuge is full, find the path to another refuge
        if self.status >= 2:
            return
        else:
            self.route = self.route[1:]
            if len(self.route) > 0:
                next_road = self.roads[(self.route[0], self.route[1],0)]
                # 1. keep on the plan
                if next_road.crowd_index < self.pref:
                    self.road = next_road
                    self.pos = 0
                    self.road.people[self.pid] = self
                    self.status = 1
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
                        # the other road is even more crowded, keep the plan
                        else:
                            self.route =tmp[0]
                            self.road = next_road
                            self.pos = 0
                            self.road.people[self.pid] = self
                            self.status = 1
                    # no other road, choose the origin road,
                    # route and refuge not change
                    else:
                        self.road = next_road
                        self.pos = 0
                        self.road.people[self.pid] = self
                        self.status = 1
            # 3. the refuge is null, find another target
            else:
                self.get_target()
                if self.status != 3:
                    self.road = self.roads[(self.route[0], self.route[1],0)]
                    self.pos = 0
                    self.road.people[self.pid] = self
                    self.status = 1
                else:
                    self.stay()

    def stay(self):
        for e in self.roads:
            if e[0] == self.pnode:
                self.road = self.roads[e]
                break
        self.road.fixed_list.append(self)

    def go_to_next_road(self, road=None):
        """
        if person has a road attribute, assign person to the road's people dict
        :param road:
        :return:
        """
        if road is None:
            road = self.road
        if road is None:
            return
        # go to the road when person is still evacuating
        if self.status <= 1:
            self.pnode = road.node_i
            road.people[self.pid] = self
            self.status = 1
        # people stay at his place when he arrived, has no place to go or dead

    def xy(self):
        return self.road.geom.interpolate(self.pos).coords[0]


class Model:
    def __init__(self, graph, num_people, num_refuge, refuge_capacity=MAX_CAPACITY):
        self.graph = graph
        self.isfinished = False
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
            self.people[i] = Person(self.graph, i, p, self.roads, targets=self.refuges)
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
                    person.road = person.refuge.road
                    person.status = 2
                    person.road.fixed_list.append(person)
                    person.refuge.arrive_list[person.pid] = person
                # assign start road to person
                else:
                    person.road = self.roads[(person.route[0], person.route[1], 0)]
                    person.road.people[person.pid] = person
                all_routes.append(person.route)
            else:
                person.stay()

        # fig, ax = ox.plot_graph_routes(self.graph, all_routes)

    def move(self, timesteps=1):
        for r in self.roads.values():
            r.move(timestamp=timesteps)

    def relocate(self):
        for p in self.people:
            person = self.people[p]
            if person.pos == 0:
                person.go_to_next_road()

    def record(self):
        for p in self.people:
            person = self.people[p]
            print('person: ',person.pid,' status:',person.status,
                  ' Road:',(person.road.node_i, person.road.node_j)
                  , ' pos',person.pos,' index',person.road.crowd_index,
                  ' Person Speed',person.speed)

    def show(self, ax):
        return ax.scatter(*zip(*[self.people[p].xy() for p in self.people]))

    def show_refuge(self, ax):
        return ax.scatter(*zip(*[self.refuges[r].show() for r in self.refuges]), c='r')

    def run(self, nsteps=None, save_args=None):
        save = True
        self.find_refuge()
        if save_args is None or len(save_args) < 3:
            save = False
            for i in range(nsteps):
                self.relocate()
                self.move()
                self.show(save_args[1])
                fig, ax = plt.plot()
        else:
            fig, ax, filename = save_args
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
                if not any(self.people[p].status <= 2 for p in self.people):
                    self.isfinished = True
                    print('Evacuation completed at time: %d' % frame_number)
                    return

            if nsteps is None:
                frame = runFrame
            else:
                frame = nsteps
            anim = animation.FuncAnimation(fig, update, frames=frame, interval=100, save_count=1000)
            anim.save(filename)
            return HTML('<video width="800" controls><source src="%s" type="video/mp4"></video>' % filename)

    def run2(self, nsteps):
        self.find_refuge()
        self.record()
        for i in range(nsteps):
            self.move()
            self.relocate()
            self.record()










