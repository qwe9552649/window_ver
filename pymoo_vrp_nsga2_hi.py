#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
pymoo_vrp_nsga2_hi.py

NSGA-II-HI (Hybrid Insertion) 기반 Multi-Objective CVRPTW 솔버
- pymoo 프레임워크 사용
- VRP 특화 인코딩: 구분자(0=depot) 포함 시퀀스
  [0, c3, c7, 0, c1, c5, 0] → Route1=[c3,c7], Route2=[c1,c5]
  NSGA-II가 구분자 위치를 최적화 → 차량당 적재량 자동 결정
- 교차: Route Exchange Crossover + HI 재삽입 (논문 Section III-B)
- 돌연변이: Swap/Inversion/Move/Split (경로 분할 직접 변경)
- 수리 연산자: Hybrid Insertion (HI) - 3가지 삽입 전략
  1) Nearest Path: 삽입비용 최소화 위치에 삽입
  2) Cheapest/Shortest Path: 가장 짧은 경로에 삽입
  3) Random Path: 다양성 확보를 위한 랜덤 삽입
- 초기 해: Nearest Neighbor + 제약 인식 시퀀스 생성
- 목적함수: [f1(기업), f2(고객), f3(사회)]

Chen et al. (IEEE CEC 2024) 논문 기반

Usage:
  python3 pymoo_vrp_nsga2_hi.py --input problem.json --output result.json
═══════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import argparse
import time
import warnings
import numpy as np
from copy import deepcopy

# pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# Constants - MOO Cost Parameters
# ═══════════════════════════════════════════════════════════════════════════

# f1: Enterprise cost parameters
MOO_FUEL_COST_PER_KM = 0.20            # Fuel cost (EUR/km)
MOO_VEHICLE_DAILY_COST = 7.0           # Vehicle daily cost (EUR/vehicle/day)

# f3: Society cost parameters (CO2)
MOO_VEHICLE_CO2_PER_KM = 0.12          # Delivery vehicle CO2 (kg/km)
MOO_CUSTOMER_VEHICLE_CO2_PER_KM = 0.12 # Passenger car CO2 (kg/km) — same as delivery van
MOO_LOCKER_CO2_PER_UNIT_PER_DAY = 0.5  # Locker CO2 (kg/day)

# ═══════════════════════════════════════════════════════════════════════════
# MODE_SHARE_TABLE (distance-based mode share)
# (one_way_km, walk%, bicycle%, vehicle_dedicated%, vehicle_linked%)
# ═══════════════════════════════════════════════════════════════════════════

MODE_SHARE_TABLE = [
    (0.0, 77.8, 15.8, 2.3, 4.1),
    (0.2, 70.1, 20.6, 3.4, 6.0),
    (0.4, 60.9, 25.9, 4.8, 8.4),
    (0.6, 50.8, 31.3, 6.6, 11.3),
    (0.8, 40.6, 36.1, 8.6, 14.6),
    (1.0, 31.1, 40.0, 10.9, 18.0),
    (1.2, 22.9, 42.6, 13.1, 21.4),
    (1.4, 16.3, 43.8, 15.3, 24.6),
    (1.6, 11.3, 43.8, 17.4, 27.5),
    (1.8, 7.6, 42.9, 19.4, 30.1),
    (2.0, 5.1, 41.3, 21.2, 32.4),
    (2.2, 3.3, 39.3, 22.9, 34.4),
    (2.4, 2.2, 37.1, 24.5, 36.2),
    (2.6, 1.4, 34.7, 26.1, 37.8),
    (2.8, 0.9, 32.3, 27.5, 39.3),
    (3.0, 0.6, 29.9, 28.9, 40.6),
    (3.2, 0.4, 27.5, 30.3, 41.8),
    (3.4, 0.2, 25.3, 31.6, 42.9),
    (3.6, 0.1, 23.1, 32.8, 43.9),
    (3.8, 0.1, 21.1, 34.1, 44.7),
    (4.0, 0.1, 19.2, 35.2, 45.5),
    (4.2, 0.0, 17.5, 36.3, 46.2),
    (4.4, 0.0, 15.8, 37.4, 46.7),
    (4.6, 0.0, 14.3, 38.4, 47.2),
    (4.8, 0.0, 12.9, 39.4, 47.6),
    (5.0, 0.0, 11.7, 40.4, 48.0),
]


def get_mode_share(one_way_distance_km: float):
    """편도 거리에 따른 이동수단 분담률 반환 (선형 보간)"""
    dist = max(0.0, one_way_distance_km)
    
    if dist >= 5.0:
        return (0.0, 11.7, 40.4, 48.0)  # as percentages (will be /100)
    
    for i in range(len(MODE_SHARE_TABLE) - 1):
        d1, w1, b1, vd1, vl1 = MODE_SHARE_TABLE[i]
        d2, w2, b2, vd2, vl2 = MODE_SHARE_TABLE[i + 1]
        
        if d1 <= dist < d2:
            t = (dist - d1) / (d2 - d1)
            walk = w1 + t * (w2 - w1)
            bicycle = b1 + t * (b2 - b1)
            vehicle_ded = vd1 + t * (vd2 - vd1)
            vehicle_link = vl1 + t * (vl2 - vl1)
            return (walk / 100.0, bicycle / 100.0, vehicle_ded / 100.0, vehicle_link / 100.0)
    
    return (77.8 / 100.0, 15.8 / 100.0, 2.3 / 100.0, 4.1 / 100.0)


def get_co2_weighted_vehicle_share(distance_km: float) -> float:
    """CO2 가중 차량 이용 비율"""
    walk, bicycle, dedicated, linked = get_mode_share(distance_km)
    return dedicated * 1.0 + linked * 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Route Decoding
# ═══════════════════════════════════════════════════════════════════════════

def decode_sequence_to_routes(seq):
    """
    구분자(0=depot) 포함 시퀀스를 경로 리스트로 디코딩
    
    논문 Section III-A 방식:
    seq = [0, c3, c7, 0, c1, c5, 0, c2, 0]
    → routes = [[c3, c7], [c1, c5], [c2]]
    
    고객 인덱스는 1-based (시퀀스에서), 반환은 0-based (내부용)
    시퀀스 내 0은 depot 구분자.
    
    Args:
        seq: 구분자 포함 시퀀스 (numpy array 또는 list)
            - 값 0: depot 구분자
            - 값 1~N: 고객 ID (1-based → 내부 0-based로 변환)
    
    Returns:
        routes: list of lists (각 경로의 고객 인덱스, 0-indexed)
    """
    routes = []
    current_route = []
    
    for val in seq:
        val = int(val)
        if val == 0:
            # depot 구분자: 현재 경로 저장 및 새 경로 시작
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            # 고객 ID: 1-based → 0-based
            current_route.append(val - 1)
    
    if current_route:
        routes.append(current_route)
    
    return routes


def decode_routes_from_perm(perm, demands, capacity,
                            time_matrix=None, time_windows=None,
                            service_times=None, depot_tw=None):
    """
    레거시 호환용: 순열을 경로로 디코딩 (용량 + 시간창 제약 기반 분할)
    f2 정규화 초기화에서 사용.
    """
    routes = []
    current_route = []
    current_load = 0
    
    tw_aware = (time_matrix is not None and time_windows is not None 
                and service_times is not None and depot_tw is not None)
    
    if tw_aware:
        depot_start_h = depot_tw[0] / 3600.0
        current_time = depot_start_h
    
    for idx in perm:
        if idx < 0 or idx >= len(demands):
            continue
        demand = demands[idx]
        
        if current_load + demand > capacity:
            if current_route:
                routes.append(current_route)
            current_route = [idx]
            current_load = demand
            if tw_aware:
                current_time = depot_start_h
                travel_h = time_matrix[0][idx + 1] / 3600.0
                arrival_h = current_time + travel_h
                tw_early_h = time_windows[idx][0] / 3600.0
                current_time = max(arrival_h, tw_early_h) + service_times[idx] / 3600.0
            continue
        
        if tw_aware:
            if len(current_route) == 0:
                prev_matrix_idx = 0
            else:
                prev_matrix_idx = current_route[-1] + 1
            travel_h = time_matrix[prev_matrix_idx][idx + 1] / 3600.0
            arrival_h = current_time + travel_h
            tw_late_h = time_windows[idx][1] / 3600.0
            
            if arrival_h > tw_late_h:
                if current_route:
                    routes.append(current_route)
                current_route = [idx]
                current_load = demand
                current_time = depot_start_h
                travel_from_depot_h = time_matrix[0][idx + 1] / 3600.0
                arrival_from_depot_h = current_time + travel_from_depot_h
                tw_early_h = time_windows[idx][0] / 3600.0
                current_time = max(arrival_from_depot_h, tw_early_h) + service_times[idx] / 3600.0
                continue
            
            tw_early_h = time_windows[idx][0] / 3600.0
            current_time = max(arrival_h, tw_early_h) + service_times[idx] / 3600.0
        
        current_route.append(idx)
        current_load += demand
    
    if current_route:
        routes.append(current_route)
    
    return routes


def build_feasible_sequence(customers_0based, demands, capacity, time_matrix, 
                             time_windows, service_times, depot_tw):
    """
    고객 리스트(0-based)로부터 제약을 만족하는 구분자 포함 시퀀스 생성.
    용량/시간창 위반 시 depot(0)을 삽입하여 새 경로를 시작.
    
    Returns:
        seq: list of int (0=depot, 1~N=customer 1-based)
    """
    seq = [0]  # 시작 depot
    current_load = 0
    depot_start_h = depot_tw[0] / 3600.0
    current_time = depot_start_h
    
    for cust_0 in customers_0based:
        cust_1 = cust_0 + 1  # 1-based (시퀀스용)
        demand = demands[cust_0]
        
        # 이전 노드 → 이 고객 이동시간
        if len(seq) == 0 or seq[-1] == 0:
            prev_mi = 0  # depot
        else:
            prev_mi = seq[-1]  # 이미 1-based
        
        travel_h = time_matrix[prev_mi][cust_1] / 3600.0
        arrival_h = current_time + travel_h
        tw_late_h = time_windows[cust_0][1] / 3600.0
        
        # 용량 초과 또는 시간창 초과 → 새 경로
        if current_load + demand > capacity or arrival_h > tw_late_h:
            seq.append(0)  # depot 구분자
            current_load = 0
            current_time = depot_start_h
            # depot에서 새로 출발
            travel_h = time_matrix[0][cust_1] / 3600.0
            arrival_h = current_time + travel_h
        
        seq.append(cust_1)
        current_load += demand
        tw_early_h = time_windows[cust_0][0] / 3600.0
        current_time = max(arrival_h, tw_early_h) + service_times[cust_0] / 3600.0
    
    seq.append(0)  # 종료 depot
    return seq


# ═══════════════════════════════════════════════════════════════════════════
# Customer Satisfaction (Chen et al., IEEE CEC 2024)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_customer_satisfaction(arrival_time_h: float, desired_time_h: float):
    """
    고객 만족도 계산
    S = 1 / (1 + waiting_time)²
    """
    waiting_time = max(0.0, arrival_time_h - desired_time_h)
    satisfaction = 1.0 / (1.0 + waiting_time) ** 2
    return satisfaction, waiting_time


# ═══════════════════════════════════════════════════════════════════════════
# Route Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_routes(routes, dist_matrix, time_matrix, time_windows, 
                    service_times, depot_tw, customer_types=None, 
                    demands=None, node_individual_desired_times=None):
    """
    경로 평가: 거리, 시간, D2D 고객 만족도, 시간창 위반 계산
    
    D2D 고객만 만족도/불만족도 측정 (락커 고객은 이동거리로 측정)
    
    Args:
        routes: list of lists (0-indexed 고객 인덱스)
        dist_matrix: NxN 거리 행렬 (0=depot, 1..=customers) - km
        time_matrix: NxN 시간 행렬 (0=depot, 1..=customers) - seconds
        time_windows: N-1 vector of (early, late) in seconds
        service_times: N-1 vector in seconds
        depot_tw: (early, late) in seconds
        customer_types: list of 'locker' or 'd2d'
        demands: list of demands per customer
        node_individual_desired_times: list of list of desired times (seconds)
    
    Returns:
        dict with evaluation metrics
    """
    total_distance = 0.0
    total_time_hours = 0.0
    total_wait_time_hours = 0.0
    tw_violations = 0.0
    
    # D2D customer satisfaction
    d2d_total_satisfaction = 0.0
    d2d_num_customers = 0
    d2d_total_delay_hours = 0.0
    
    use_individual_times = (node_individual_desired_times is not None 
                           and len(node_individual_desired_times) > 0)
    
    for route in routes:
        if not route:
            continue
        
        route_dist = 0.0
        current_time = depot_tw[0] / 3600.0  # hours
        
        # depot(0) → first customer
        first_idx = route[0] + 1  # matrix index (depot=0)
        route_dist += dist_matrix[0][first_idx]
        current_time += time_matrix[0][first_idx] / 3600.0
        
        for i, cust_idx in enumerate(route):
            matrix_idx = cust_idx + 1
            
            # Time window info
            tw_early_h = time_windows[cust_idx][0] / 3600.0
            tw_late_h = time_windows[cust_idx][1] / 3600.0
            
            # Customer type check
            is_locker = (customer_types is not None 
                        and cust_idx < len(customer_types) 
                        and customer_types[cust_idx] == 'locker')
            
            arrival_time = current_time
            
            # Wait if early
            if current_time < tw_early_h:
                wait_time = tw_early_h - current_time
                total_wait_time_hours += wait_time
                current_time = tw_early_h
            
            # D2D customer satisfaction (locker excluded)
            if not is_locker:
                if (use_individual_times 
                    and cust_idx < len(node_individual_desired_times) 
                    and node_individual_desired_times[cust_idx]):
                    for desired_sec in node_individual_desired_times[cust_idx]:
                        desired_h = desired_sec / 3600.0
                        sat, delay = calculate_customer_satisfaction(arrival_time, desired_h)
                        d2d_total_satisfaction += sat
                        d2d_total_delay_hours += delay
                        d2d_num_customers += 1
                else:
                    node_weight = 1
                    if demands is not None and cust_idx < len(demands):
                        node_weight = max(1, demands[cust_idx])
                    sat, delay = calculate_customer_satisfaction(arrival_time, tw_early_h)
                    d2d_total_satisfaction += sat * node_weight
                    d2d_total_delay_hours += delay * node_weight
                    d2d_num_customers += node_weight
            
            # Time window violation (D2D only, locker = 24h)
            if not is_locker and current_time > tw_late_h:
                tw_violations += (current_time - tw_late_h)
            
            # Service time
            current_time += service_times[cust_idx] / 3600.0
            
            # Travel to next node
            if i < len(route) - 1:
                next_idx = route[i + 1] + 1
                route_dist += dist_matrix[matrix_idx][next_idx]
                current_time += time_matrix[matrix_idx][next_idx] / 3600.0
        
        # Last customer → depot
        last_idx = route[-1] + 1
        route_dist += dist_matrix[last_idx][0]
        current_time += time_matrix[last_idx][0] / 3600.0
        
        total_distance += route_dist
        total_time_hours += (current_time - depot_tw[0] / 3600.0)
    
    avg_d2d_satisfaction = (d2d_total_satisfaction / d2d_num_customers 
                           if d2d_num_customers > 0 else 1.0)
    avg_d2d_delay = (d2d_total_delay_hours / d2d_num_customers 
                     if d2d_num_customers > 0 else 0.0)
    
    return {
        'total_distance': total_distance,
        'total_time_hours': total_time_hours,
        'total_wait_time_hours': total_wait_time_hours,
        'tw_violations': tw_violations,
        'avg_satisfaction': avg_d2d_satisfaction,
        'avg_customer_delay': avg_d2d_delay,
        'num_d2d_customers': d2d_num_customers,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MOO Objective Function Evaluation
# ═══════════════════════════════════════════════════════════════════════════

# Upper-Lower-Bound Min-Max (Marler & Arora 2005, Ishibuchi et al. 2017), range≈0 → z'=0
F2_NORM_EPSILON = 1e-6
F2_NORM_RANGE_MIN = 1e-10


def evaluate_moo_objectives(routes, dist_matrix, time_matrix, time_windows,
                            service_times, depot_tw, customer_locker_distances=None,
                            num_active_lockers=0, customer_types=None, demands=None,
                            node_individual_desired_times=None,
                            f2_norm_params=None):
    """
    MOO 목적함수 평가 (f1, f2, f3)
    f2: Min-Max 정규화, range≈0이면 z'=0 (변별력 없음, D2D 전용 등)
    """
    n_customers = len(dist_matrix) - 1
    num_vehicles = len(routes)
    
    # Route evaluation
    eval_result = evaluate_routes(
        routes, dist_matrix, time_matrix, time_windows, service_times, depot_tw,
        customer_types=customer_types, demands=demands,
        node_individual_desired_times=node_individual_desired_times
    )
    
    total_distance = eval_result['total_distance']
    tw_violations = eval_result['tw_violations']
    avg_d2d_satisfaction = eval_result['avg_satisfaction']
    avg_d2d_delay = eval_result['avg_customer_delay']
    num_d2d_customers = eval_result['num_d2d_customers']
    
    # ─────────────────────────────────────────────────────────────────────
    # f1: Enterprise cost (EUR)
    # ─────────────────────────────────────────────────────────────────────
    f1_fuel = total_distance * MOO_FUEL_COST_PER_KM
    f1_vehicle = num_vehicles * MOO_VEHICLE_DAILY_COST
    f1 = f1_fuel + f1_vehicle
    
    # ─────────────────────────────────────────────────────────────────────
    # f2: Customer cost (normalized)
    # ─────────────────────────────────────────────────────────────────────
    
    # 1. Locker customer actual travel distance (MODE_SHARE based)
    total_customer_actual_dist = 0.0
    num_locker_customers = 0
    total_customer_vehicle_co2 = 0.0
    
    if customer_locker_distances is not None:
        for dist in customer_locker_distances:
            if dist > 0:
                num_locker_customers += 1
                walk, bicycle, dedicated, linked = get_mode_share(dist)
                round_trip_dist = dist * 2.0
                one_way_dist = dist
                actual_dist = (walk + bicycle + dedicated) * round_trip_dist + linked * one_way_dist
                total_customer_actual_dist += actual_dist
                
                # f3: CO2 calculation
                dedicated_co2 = dedicated * round_trip_dist * MOO_CUSTOMER_VEHICLE_CO2_PER_KM
                linked_co2 = linked * one_way_dist * MOO_CUSTOMER_VEHICLE_CO2_PER_KM
                total_customer_vehicle_co2 += dedicated_co2 + linked_co2
    
    avg_customer_actual_dist = (total_customer_actual_dist / num_locker_customers 
                                if num_locker_customers > 0 else 0.0)
    
    # 2. D2D customer dissatisfaction
    d2d_total_dissatisfaction = num_d2d_customers * (1.0 - avg_d2d_satisfaction)
    
    # f2 = Min-Max, range≈0이면 z'=0 (변별력 없음)
    if f2_norm_params is None:
        f2_norm_params = {'mobility_L': 0, 'mobility_U': 1, 'dissatisfaction_L': 0, 'dissatisfaction_U': 1}
    mob_L = f2_norm_params.get('mobility_L', 0)
    mob_U = f2_norm_params.get('mobility_U', 1)
    dis_L = f2_norm_params.get('dissatisfaction_L', 0)
    dis_U = f2_norm_params.get('dissatisfaction_U', 1)
    range_mob = mob_U - mob_L
    range_dis = dis_U - dis_L
    # range=0이면 변별력 없으므로 0 처리 (부동소수점 -0 방지)
    mobility_normalized = ((avg_customer_actual_dist - mob_L) / (range_mob + F2_NORM_EPSILON)
                           if range_mob >= F2_NORM_RANGE_MIN else 0.0)
    dissatisfaction_normalized = ((d2d_total_dissatisfaction - dis_L) / (range_dis + F2_NORM_EPSILON)
                                  if range_dis >= F2_NORM_RANGE_MIN else 0.0)
    f2 = mobility_normalized + dissatisfaction_normalized
    
    # ─────────────────────────────────────────────────────────────────────
    # f3: Society cost (kg CO2)
    # ─────────────────────────────────────────────────────────────────────
    vehicle_co2 = total_distance * MOO_VEHICLE_CO2_PER_KM
    locker_co2 = num_active_lockers * MOO_LOCKER_CO2_PER_UNIT_PER_DAY
    f3 = vehicle_co2 + total_customer_vehicle_co2 + locker_co2
    
    detail = {
        'f1': f1, 'f2': f2, 'f3': f3,
        'f1_fuel_cost': f1_fuel,
        'f1_vehicle_cost': f1_vehicle,
        'f1_driver_cost': 0.0,
        'f2_mobility_inconvenience': mobility_normalized,
        'f2_dissatisfaction': dissatisfaction_normalized,
        'avg_customer_satisfaction': avg_d2d_satisfaction,
        'avg_customer_delay': avg_d2d_delay,
        'avg_customer_actual_dist': avg_customer_actual_dist,
        'mobility_raw_km': avg_customer_actual_dist,
        'dissatisfaction_raw': d2d_total_dissatisfaction,
        'f3_vehicle_co2': vehicle_co2,
        'f3_customer_co2': total_customer_vehicle_co2,
        'f3_locker_co2': locker_co2,
        'total_distance': total_distance,
        'num_vehicles': num_vehicles,
        'tw_violations': tw_violations,
    }
    
    return [f1, f2, f3], [tw_violations], total_distance, avg_d2d_satisfaction, detail


# ═══════════════════════════════════════════════════════════════════════════
# F2 Normalization Initialization
# ═══════════════════════════════════════════════════════════════════════════

class _SingleObjF2Problem(ElementwiseProblem):
    """f2 하위 목적함수 단일 최적화용 문제 (Marler & Arora 2005, ideal point 탐색)"""
    
    def __init__(self, n_customers, dist_matrix, time_matrix, demands,
                 time_windows, service_times, depot_tw, capacity,
                 customer_locker_distances, customer_types, node_individual_desired_times,
                 target='dissatisfaction'):
        self.n_customers = n_customers
        total_demand = sum(demands) if demands else n_customers
        theoretical_min = max(1, int(np.ceil(total_demand / capacity)))
        self.max_vehicles = min(n_customers, max(5, theoretical_min * 4))
        seq_len = n_customers + self.max_vehicles + 1
        
        super().__init__(n_var=seq_len, n_obj=1, n_ieq_constr=1, vtype=int)
        
        self.dist_matrix = dist_matrix
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.depot_tw = depot_tw
        self.capacity = capacity
        self.customer_locker_distances = customer_locker_distances
        self.customer_types = customer_types
        self.node_individual_desired_times = node_individual_desired_times
        self.target = target  # 'mobility' or 'dissatisfaction'
    
    def _evaluate(self, x, out, *args, **kwargs):
        seq = x.astype(int)
        routes = decode_sequence_to_routes(seq)
        routes = [r for r in routes if len(r) > 0]
        
        if not routes:
            out["F"] = [1e9]
            out["G"] = [1e6]
            return
        
        eval_result = evaluate_routes(
            routes, self.dist_matrix, self.time_matrix,
            self.time_windows, self.service_times, self.depot_tw,
            customer_types=self.customer_types, demands=self.demands,
            node_individual_desired_times=self.node_individual_desired_times
        )
        
        if self.target == 'mobility':
            num_locker_cust = 0
            total_locker_dist = 0.0
            if self.customer_locker_distances is not None:
                for d in self.customer_locker_distances:
                    if d > 0:
                        num_locker_cust += 1
                        walk, bicycle, dedicated, linked = get_mode_share(d)
                        actual_dist = (walk + bicycle + dedicated) * d * 2.0 + linked * d
                        total_locker_dist += actual_dist
            obj = total_locker_dist / num_locker_cust if num_locker_cust > 0 else 0.0
        else:  # dissatisfaction
            obj = eval_result['num_d2d_customers'] * (1.0 - eval_result['avg_satisfaction'])
        
        # 제약: 시간창 위반 + 용량 위반
        tw_viol = eval_result['tw_violations']
        cap_viol = 0.0
        for route in routes:
            rd = sum(self.demands[c] for c in route)
            if rd > self.capacity:
                cap_viol += rd - self.capacity
        
        out["F"] = [obj]
        out["G"] = [tw_viol + cap_viol]


def _compute_f2_component(routes, dist_matrix, time_matrix, time_windows,
                           service_times, depot_tw, customer_locker_distances,
                           customer_types, demands, node_individual_desired_times):
    """경로에서 mobility, dissatisfaction 계산 (ideal point 평가용)"""
    eval_result = evaluate_routes(
        routes, dist_matrix, time_matrix, time_windows, service_times, depot_tw,
        customer_types=customer_types, demands=demands,
        node_individual_desired_times=node_individual_desired_times
    )
    num_locker_cust = 0
    total_locker_dist = 0.0
    if customer_locker_distances is not None:
        for d in customer_locker_distances:
            if d > 0:
                num_locker_cust += 1
                walk, bicycle, dedicated, linked = get_mode_share(d)
                actual_dist = (walk + bicycle + dedicated) * d * 2.0 + linked * d
                total_locker_dist += actual_dist
    mobility = total_locker_dist / num_locker_cust if num_locker_cust > 0 else 0.0
    dissatisfaction = eval_result['num_d2d_customers'] * (1.0 - eval_result['avg_satisfaction'])
    return mobility, dissatisfaction


def initialize_f2_normalization(n_customers, dist_matrix, time_matrix, 
                                 time_windows, service_times, depot_tw,
                                 demands, capacity, customer_locker_distances=None,
                                 customer_types=None, node_individual_desired_times=None,
                                 n_gen=20, pop_size=20):
    """
    f2 Min-Max 파라미터 — 단일 목적 최적화로 ideal point 탐색 (Marler & Arora 2005, Eq.7).
    
    mobility, dissatisfaction 각각을 단독 minimize하여 ideal point x_mob*, x_dis* 결정.
    z_L = min, z_U = Pareto maximum (2개 ideal 해에서의 max).
    range≈0이면 z'=0 (변별력 없음).
    """
    from pymoo.algorithms.soo.nonconvex.ga import GA
    
    ideal_results = {}  # target -> (mobility, dissatisfaction)
    
    for target in ['mobility', 'dissatisfaction']:
        problem = _SingleObjF2Problem(
            n_customers, dist_matrix, time_matrix, demands,
            time_windows, service_times, depot_tw, capacity,
            customer_locker_distances, customer_types, node_individual_desired_times,
            target=target
        )
        
        sampling = SequenceSampling()
        crossover = RouteExchangeCrossover()
        mutation = SequenceMutation()
        repair = SequenceRepair()
        
        algorithm = GA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
        
        res = minimize(
            problem, algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=42, verbose=False
        )
        
        # 최적해에서 경로 추출 → mobility, dissatisfaction 평가
        best_seq = res.X.astype(int)
        best_routes = decode_sequence_to_routes(best_seq)
        best_routes = [r for r in best_routes if len(r) > 0]
        
        mob, dis = _compute_f2_component(
            best_routes, dist_matrix, time_matrix, time_windows,
            service_times, depot_tw, customer_locker_distances,
            customer_types, demands, node_individual_desired_times
        )
        ideal_results[target] = (mob, dis)
    
    # z_L = ideal, z_U = Pareto maximum (2개 ideal 해에서의 min/max)
    mob_mob, dis_mob = ideal_results['mobility']       # x_mob*에서의 (mobility, dissatisfaction)
    mob_dis, dis_dis = ideal_results['dissatisfaction'] # x_dis*에서의 (mobility, dissatisfaction)
    
    mob_L = min(mob_mob, mob_dis)
    mob_U = max(mob_mob, mob_dis)
    dis_L = min(dis_mob, dis_dis)
    dis_U = max(dis_mob, dis_dis)
    return {'mobility_L': mob_L, 'mobility_U': mob_U,
            'dissatisfaction_L': dis_L, 'dissatisfaction_U': dis_U}


# ═══════════════════════════════════════════════════════════════════════════
# VRP Problem Definition (pymoo ElementwiseProblem)
# ═══════════════════════════════════════════════════════════════════════════

class CVRPTWProblem(ElementwiseProblem):
    """
    CVRPTW - 논문(Chen et al. 2024) 방식 구분자 포함 시퀀스 인코딩
    
    Encoding: [0, c3, c7, 0, c1, c5, 0, c2, 0]
      - 0 = depot 구분자 (경로 분할점)
      - 1~N = 고객 ID (1-based)
      - 두 0 사이의 고객들 = 한 차량의 경로
      
    NSGA-II가 구분자 위치도 최적화 → 차량당 적재량 자동 결정
    
    max_vehicles 자동 계산:
      - 이론적 최소 차량 = 총수요 / 용량
      - 시간창/거리 제약 고려 × 탐색 여유 = 4배
      - 2033명 고객 예시: 677대(기존) → 244대(최적화) [16% 효율 개선]
    
    n_var = n_customers + max_vehicles + 1
      (고객 N개 + depot 구분자 최대 max_vehicles+1개)
    """
    
    def __init__(self, n_customers, dist_matrix, time_matrix, demands,
                 time_windows, service_times, depot_tw, capacity,
                 customer_locker_distances=None, num_active_lockers=0,
                 customer_types=None, node_individual_desired_times=None,
                 f2_norm_params=None,
                 max_vehicles=None):
        
        self.n_customers = n_customers
        # 최대 차량 수: 수요 기반 자동 계산 (속도 최적화)
        # theoretical_min × 4배 여유 = 충분한 탐색 공간 + 효율적 탐색
        if max_vehicles is None:
            total_demand = sum(demands) if demands else n_customers
            theoretical_min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
            # 시간창/거리 제약 고려 + 탐색 여유 = 4배, 최소 5대
            max_vehicles = min(n_customers, max(5, theoretical_min_vehicles * 4))
        self.max_vehicles = max_vehicles
        
        # 시퀀스 길이: 고객 N개 + depot 구분자 (max_vehicles+1)개
        seq_len = n_customers + max_vehicles + 1
        
        super().__init__(
            n_var=seq_len,
            n_obj=3,
            n_ieq_constr=1,  # time window violation (hard constraint)
            vtype=int
        )
        
        self.dist_matrix = dist_matrix
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.depot_tw = depot_tw
        self.capacity = capacity
        self.customer_locker_distances = customer_locker_distances
        self.num_active_lockers = num_active_lockers
        self.customer_types = customer_types
        self.node_individual_desired_times = node_individual_desired_times
        self.f2_norm_params = f2_norm_params or {'mobility_L': 0, 'mobility_U': 1,
                                                'dissatisfaction_L': 0, 'dissatisfaction_U': 1}
    
    def _evaluate(self, x, out, *args, **kwargs):
        """구분자 포함 시퀀스를 경로로 디코딩 후 목적함수 평가"""
        seq = x.astype(int)
        routes = decode_sequence_to_routes(seq)
        
        # 빈 경로 제거
        routes = [r for r in routes if len(r) > 0]
        
        if not routes:
            out["F"] = [1e9, 1e9, 1e9]
            out["G"] = [1e6]
            return
        
        f_vec, g_vec, _, _, _ = evaluate_moo_objectives(
            routes, self.dist_matrix, self.time_matrix,
            self.time_windows, self.service_times, self.depot_tw,
            customer_locker_distances=self.customer_locker_distances,
            num_active_lockers=self.num_active_lockers,
            customer_types=self.customer_types,
            demands=self.demands,
            node_individual_desired_times=self.node_individual_desired_times,
            f2_norm_params=self.f2_norm_params
        )
        
        # 추가 제약: 용량 위반 체크
        capacity_violation = 0.0
        for route in routes:
            route_demand = sum(self.demands[c] for c in route)
            if route_demand > self.capacity:
                capacity_violation += route_demand - self.capacity
        
        # 모든 고객이 포함되었는지 확인
        all_customers = set()
        for r in routes:
            all_customers.update(r)
        missing = self.n_customers - len(all_customers)
        
        total_violation = g_vec[0] + capacity_violation * 10.0 + missing * 100.0
        
        out["F"] = f_vec
        out["G"] = [total_violation]


# ═══════════════════════════════════════════════════════════════════════════
# VRP-Specialized Operators
# ═══════════════════════════════════════════════════════════════════════════

class SequenceSampling(Sampling):
    """
    논문 방식 초기 해 생성: 구분자 포함 시퀀스
    
    1/3: Nearest Neighbor 기반 (시간창/용량 인식 경로 구성)
    1/3: Randomized Nearest Neighbor 기반
    1/3: 랜덤 순서 + 제약 기반 자동 분할
    """
    
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        nc = problem.n_customers
        
        for i in range(n_samples):
            if i < n_samples // 3:
                customers = self._nearest_neighbor_order(nc, problem.dist_matrix)
            elif i < 2 * n_samples // 3:
                customers = self._randomized_nn_order(nc, problem.dist_matrix, k=3)
            else:
                customers = list(np.random.permutation(nc))
            
            # 고객 순서를 제약 인식 시퀀스로 변환
            seq = build_feasible_sequence(
                customers, problem.demands, problem.capacity,
                problem.time_matrix, problem.time_windows,
                problem.service_times, problem.depot_tw
            )
            
            # 시퀀스를 n_var 길이로 패딩 (남는 자리는 0=depot)
            arr = np.zeros(problem.n_var, dtype=int)
            for j in range(min(len(seq), problem.n_var)):
                arr[j] = seq[j]
            X[i] = arr
        
        return X
    
    @staticmethod
    def _nearest_neighbor_order(n, dist):
        """Nearest Neighbor 순서로 고객 정렬 (0-based 반환)"""
        visited = set()
        order = []
        start = int(np.argmin([dist[0][j + 1] for j in range(n)]))
        order.append(start)
        visited.add(start)
        while len(order) < n:
            last_mi = order[-1] + 1
            best_d, best_j = float('inf'), -1
            for j in range(n):
                if j not in visited and dist[last_mi][j + 1] < best_d:
                    best_d = dist[last_mi][j + 1]
                    best_j = j
            if best_j >= 0:
                order.append(best_j)
                visited.add(best_j)
            else:
                break
        for j in range(n):
            if j not in visited:
                order.append(j)
        return order
    
    @staticmethod
    def _randomized_nn_order(n, dist, k=3):
        """Randomized Nearest Neighbor 순서 (0-based 반환)"""
        visited = set()
        order = []
        start = np.random.randint(0, n)
        order.append(start)
        visited.add(start)
        while len(order) < n:
            last_mi = order[-1] + 1
            cands = [(dist[last_mi][j + 1], j) for j in range(n) if j not in visited]
            cands.sort()
            top = cands[:min(k, len(cands))]
            if top:
                _, chosen = top[np.random.randint(0, len(top))]
                order.append(chosen)
                visited.add(chosen)
            else:
                break
        for j in range(n):
            if j not in visited:
                order.append(j)
        return order


class RouteExchangeCrossover(Crossover):
    """
    논문 Section III-B: 경로 단위 교환 + HI 재삽입
    
    1) p1에서 랜덤 경로 Ω1, p2에서 랜덤 경로 Ω2 선택
    2) p1에서 Ω1 제거 → Ω2 삽입, p2에서 Ω2 제거 → Ω1 삽입
    3) 중복 고객 제거, 누락 고객을 HI 전략으로 재삽입
    """
    
    def __init__(self, prob=0.8, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)
        self.crossover_prob = float(prob)
    
    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        n = problem.n_var
        Y = np.full((self.n_offsprings, n_matings, n), 0, dtype=int)
        
        for k in range(n_matings):
            p1 = list(X[0, k].astype(int))
            p2 = list(X[1, k].astype(int))
            
            if np.random.random() > self.crossover_prob:
                Y[0, k] = np.array(self._pad(p1, n), dtype=int)
                Y[1, k] = np.array(self._pad(p2, n), dtype=int)
                continue
            
            c1, c2 = self._route_exchange(p1, p2, problem)
            Y[0, k] = np.array(self._pad(c1, n), dtype=int)
            Y[1, k] = np.array(self._pad(c2, n), dtype=int)
        
        return Y
    
    @staticmethod
    def _pad(seq, n):
        """시퀀스를 n 길이로 패딩 (0=depot)"""
        arr = [0] * n
        for i in range(min(len(seq), n)):
            arr[i] = seq[i]
        return arr
    
    @staticmethod
    def _get_routes_from_seq(seq):
        """시퀀스에서 경로 리스트 추출 (depot 기준 분할)"""
        routes = []
        current = []
        for v in seq:
            v = int(v)
            if v == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(v)
        if current:
            routes.append(current)
        return routes
    
    @staticmethod
    def _routes_to_seq(routes):
        """경로 리스트를 시퀀스로 변환"""
        seq = [0]
        for route in routes:
            seq.extend(route)
            seq.append(0)
        return seq
    
    def _route_exchange(self, p1, p2, problem):
        """경로 단위 교환 후 HI 재삽입"""
        routes1 = self._get_routes_from_seq(p1)
        routes2 = self._get_routes_from_seq(p2)
        
        if not routes1 or not routes2:
            return p1[:], p2[:]
        
        # 랜덤 경로 선택
        r1_idx = np.random.randint(0, len(routes1))
        r2_idx = np.random.randint(0, len(routes2))
        omega1 = routes1[r1_idx]  # p1에서 선택된 경로
        omega2 = routes2[r2_idx]  # p2에서 선택된 경로
        
        # p1에서 Ω1 제거 → Ω2 삽입
        new_routes1 = [r[:] for r in routes1]
        new_routes1[r1_idx] = omega2[:]
        
        # p2에서 Ω2 제거 → Ω1 삽입
        new_routes2 = [r[:] for r in routes2]
        new_routes2[r2_idx] = omega1[:]
        
        # 중복 고객 제거 및 누락 고객 재삽입
        c1_seq = self._fix_duplicates_and_missing(
            new_routes1, problem.n_customers, problem)
        c2_seq = self._fix_duplicates_and_missing(
            new_routes2, problem.n_customers, problem)
        
        return c1_seq, c2_seq
    
    def _fix_duplicates_and_missing(self, routes, n_customers, problem):
        """중복 고객 제거, 누락 고객을 HI 전략으로 재삽입"""
        all_custs = set(range(1, n_customers + 1))  # 1-based
        
        # 중복 제거: 먼저 나온 것만 유지
        seen = set()
        for route in routes:
            to_remove = []
            for i, c in enumerate(route):
                if c in seen:
                    to_remove.append(i)
                else:
                    seen.add(c)
            for i in reversed(to_remove):
                route.pop(i)
        
        # 빈 경로 제거
        routes = [r for r in routes if len(r) > 0]
        
        # 누락 고객 찾기
        missing = all_custs - seen
        
        # HI 전략으로 누락 고객 삽입
        dist_matrix = problem.dist_matrix
        for cust_1based in missing:
            strategy = np.random.choice(['nearest', 'shortest', 'random'],
                                        p=[0.5, 0.3, 0.2])
            inserted = False
            
            if strategy == 'nearest' and routes:
                # Nearest Path: 가장 가까운 고객이 있는 경로에 삽입
                best_route_idx, best_pos, best_cost = -1, 0, float('inf')
                cust_mi = cust_1based  # matrix index = 1-based
                for ri, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        prev_mi = 0 if pos == 0 else route[pos - 1]
                        next_mi = 0 if pos == len(route) else route[pos]
                        cost = (dist_matrix[prev_mi][cust_mi] + 
                                dist_matrix[cust_mi][next_mi] - 
                                dist_matrix[prev_mi][next_mi])
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = ri
                            best_pos = pos
                if best_route_idx >= 0:
                    routes[best_route_idx].insert(best_pos, cust_1based)
                    inserted = True
                    
            elif strategy == 'shortest' and routes:
                # Shortest Path: 가장 짧은 경로에 삽입
                shortest_ri = min(range(len(routes)), key=lambda ri: len(routes[ri]))
                route = routes[shortest_ri]
                cust_mi = cust_1based
                best_pos, best_cost = 0, float('inf')
                for pos in range(len(route) + 1):
                    prev_mi = 0 if pos == 0 else route[pos - 1]
                    next_mi = 0 if pos == len(route) else route[pos]
                    cost = (dist_matrix[prev_mi][cust_mi] + 
                            dist_matrix[cust_mi][next_mi] - 
                            dist_matrix[prev_mi][next_mi])
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                routes[shortest_ri].insert(best_pos, cust_1based)
                inserted = True
                
            elif strategy == 'random' and routes:
                # Random Path: 랜덤 경로의 랜덤 위치에 삽입
                ri = np.random.randint(0, len(routes))
                pos = np.random.randint(0, len(routes[ri]) + 1)
                routes[ri].insert(pos, cust_1based)
                inserted = True
            
            if not inserted:
                # 새 경로 생성
                routes.append([cust_1based])
        
        return self._routes_to_seq(routes)


class SequenceRepair(Repair):
    """
    논문 NSGA-II-HI: 시퀀스 수리 연산자
    
    교차/돌연변이 후 시퀀스의 유효성을 보장:
    1) 모든 고객(1~N)이 정확히 한 번 포함되는지 확인
    2) 중복 고객 제거, 누락 고객 HI 재삽입
    3) 일부 해에 대해 worst-placed 고객 제거 + HI 재삽입 (개선)
    """
    
    def __init__(self, improvement_rate=0.3):
        super().__init__()
        self.improvement_rate = improvement_rate
    
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            seq = list(X[i].astype(int))
            
            # 1) 유효성 보장
            seq = self._ensure_valid_sequence(seq, problem)
            
            # 2) 일부 해에 HI 개선 적용
            if np.random.random() < self.improvement_rate:
                seq = self._hi_improvement(seq, problem)
            
            # 패딩
            arr = np.zeros(problem.n_var, dtype=int)
            for j in range(min(len(seq), problem.n_var)):
                arr[j] = seq[j]
            X[i] = arr
        
        return X
    
    @staticmethod
    def _ensure_valid_sequence(seq, problem):
        """
        시퀀스에서 모든 고객이 정확히 1번 등장하도록 수리
        """
        nc = problem.n_customers
        all_custs = set(range(1, nc + 1))  # 1-based
        
        # 현재 시퀀스에서 고객 추출 (중복 제거)
        seen = set()
        clean_seq = []
        for v in seq:
            v = int(v)
            if v == 0:
                clean_seq.append(0)
            elif 1 <= v <= nc:
                if v not in seen:
                    clean_seq.append(v)
                    seen.add(v)
                # 중복은 스킵
            # 범위 밖 값도 스킵
        
        # 빈 depot 제거 (연속된 0)
        result = []
        prev_was_depot = False
        for v in clean_seq:
            if v == 0:
                if not prev_was_depot:
                    result.append(0)
                    prev_was_depot = True
            else:
                result.append(v)
                prev_was_depot = False
        
        # 시작이 depot이 아니면 추가
        if not result or result[0] != 0:
            result.insert(0, 0)
        # 끝이 depot이 아니면 추가
        if result[-1] != 0:
            result.append(0)
        
        # 누락 고객 찾기
        missing = all_custs - seen
        
        if missing:
            # HI 전략으로 누락 고객 삽입
            dist_matrix = problem.dist_matrix
            # 시퀀스에서 경로 추출 → 삽입 → 시퀀스 재구성
            routes = []
            current = []
            for v in result:
                if v == 0:
                    if current:
                        routes.append(current)
                        current = []
                else:
                    current.append(v)
            if current:
                routes.append(current)
            
            for cust in missing:
                if not routes:
                    routes.append([cust])
                    continue
                best_ri, best_pos, best_cost = 0, 0, float('inf')
                cust_mi = cust  # 1-based = matrix index
                for ri, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        prev_mi = 0 if pos == 0 else route[pos - 1]
                        next_mi = 0 if pos == len(route) else route[pos]
                        cost = (dist_matrix[prev_mi][cust_mi] +
                                dist_matrix[cust_mi][next_mi] -
                                dist_matrix[prev_mi][next_mi])
                        if cost < best_cost:
                            best_cost = cost
                            best_ri = ri
                            best_pos = pos
                routes[best_ri].insert(best_pos, cust)
            
            # 경로 → 시퀀스
            result = [0]
            for route in routes:
                result.extend(route)
                result.append(0)
        
        return result
    
    @staticmethod
    def _hi_improvement(seq, problem):
        """
        논문 Section III-C: worst-placed 고객 제거 + HI 재삽입
        """
        dist_matrix = problem.dist_matrix
        nc = problem.n_customers
        
        # 시퀀스에서 경로 추출
        routes = []
        current = []
        for v in seq:
            v = int(v)
            if v == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(v)
        if current:
            routes.append(current)
        
        if not routes:
            return seq
        
        # 각 고객의 제거 절약량 계산
        savings = {}
        for route in routes:
            for j, cust in enumerate(route):
                prev_mi = 0 if j == 0 else route[j - 1]
                next_mi = 0 if j == len(route) - 1 else route[j + 1]
                current_cost = dist_matrix[prev_mi][cust] + dist_matrix[cust][next_mi]
                removal_cost = dist_matrix[prev_mi][next_mi]
                savings[cust] = current_cost - removal_cost
        
        if not savings:
            return seq
        
        # Worst-placed 2개 제거
        k = min(2, len(savings))
        worst = sorted(savings.keys(), key=lambda c: savings[c], reverse=True)[:k]
        worst_set = set(worst)
        
        # 제거
        for route in routes:
            for j in reversed(range(len(route))):
                if route[j] in worst_set:
                    route.pop(j)
        routes = [r for r in routes if len(r) > 0]
        
        # HI 재삽입 (3전략 중 선택)
        for cust in worst:
            strategy = np.random.choice(['nearest', 'cheapest', 'random'],
                                        p=[0.4, 0.4, 0.2])
            
            if not routes:
                routes.append([cust])
                continue
            
            if strategy == 'nearest' or strategy == 'cheapest':
                best_ri, best_pos, best_cost = 0, 0, float('inf')
                cust_mi = cust
                for ri, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        prev_mi = 0 if pos == 0 else route[pos - 1]
                        next_mi = 0 if pos == len(route) else route[pos]
                        cost = (dist_matrix[prev_mi][cust_mi] +
                                dist_matrix[cust_mi][next_mi] -
                                dist_matrix[prev_mi][next_mi])
                        if cost < best_cost:
                            best_cost = cost
                            best_ri = ri
                            best_pos = pos
                routes[best_ri].insert(best_pos, cust)
            else:
                ri = np.random.randint(0, len(routes))
                pos = np.random.randint(0, len(routes[ri]) + 1)
                routes[ri].insert(pos, cust)
        
        # 경로 → 시퀀스
        result = [0]
        for route in routes:
            result.extend(route)
            result.append(0)
        return result


class SequenceMutation(Mutation):
    """
    논문 Section III-C: 구분자 포함 시퀀스 돌연변이
    
    4가지 전략:
    1. Swap (25%): 두 고객 위치 교환
    2. Inversion (25%): 한 경로 내 부분역순
    3. Move (25%): 고객을 다른 경로로 이동 (경로 분할 변경!)
    4. Split (25%): depot 구분자 이동 (경로 분할 직접 변경!)
    """
    
    def __init__(self, prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.mutation_prob = float(prob)
    
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if np.random.random() < self.mutation_prob:
                seq = list(X[i].astype(int))
                
                strategy = np.random.choice(
                    ['swap', 'inversion', 'move', 'split'],
                    p=[0.25, 0.25, 0.25, 0.25])
                
                if strategy == 'swap':
                    seq = self._swap_customers(seq)
                elif strategy == 'inversion':
                    seq = self._inversion_in_route(seq)
                elif strategy == 'move':
                    seq = self._move_customer(seq, problem)
                else:
                    seq = self._split_mutation(seq, problem)
                
                # 패딩
                arr = np.zeros(problem.n_var, dtype=int)
                for j in range(min(len(seq), problem.n_var)):
                    arr[j] = seq[j]
                X[i] = arr
        
        return X
    
    @staticmethod
    def _get_customer_positions(seq):
        """시퀀스에서 고객(>0) 위치 인덱스 목록"""
        return [i for i, v in enumerate(seq) if int(v) > 0]
    
    @staticmethod
    def _swap_customers(seq):
        """두 고객 위치 교환"""
        positions = [i for i, v in enumerate(seq) if int(v) > 0]
        if len(positions) < 2:
            return seq
        i, j = np.random.choice(len(positions), 2, replace=False)
        pi, pj = positions[i], positions[j]
        seq[pi], seq[pj] = seq[pj], seq[pi]
        return seq
    
    @staticmethod
    def _inversion_in_route(seq):
        """한 경로 내 부분역순"""
        # 경로 식별
        routes_pos = []  # [(start_idx, end_idx), ...] 각 경로의 고객 인덱스 범위
        current = []
        for i, v in enumerate(seq):
            if int(v) == 0:
                if current:
                    routes_pos.append(current[:])
                    current = []
            else:
                current.append(i)
        if current:
            routes_pos.append(current)
        
        # 2명 이상인 경로 선택
        valid_routes = [r for r in routes_pos if len(r) >= 2]
        if not valid_routes:
            return seq
        
        route_positions = valid_routes[np.random.randint(0, len(valid_routes))]
        if len(route_positions) < 2:
            return seq
        
        # 부분역순
        i, j = sorted(np.random.choice(len(route_positions), 2, replace=False))
        segment_positions = route_positions[i:j + 1]
        values = [seq[p] for p in segment_positions]
        values.reverse()
        for k, p in enumerate(segment_positions):
            seq[p] = values[k]
        
        return seq
    
    @staticmethod
    def _move_customer(seq, problem):
        """
        고객을 한 경로에서 다른 경로로 이동 (경로 분할 변경)
        이것이 논문 핵심: 차량별 고객 수가 변함
        """
        # 경로 추출
        routes = []
        current = []
        for v in seq:
            v = int(v)
            if v == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(v)
        if current:
            routes.append(current)
        
        if len(routes) < 2:
            return seq
        
        # 소스 경로에서 고객 1명 제거
        src_ri = np.random.randint(0, len(routes))
        if not routes[src_ri]:
            return seq
        src_pos = np.random.randint(0, len(routes[src_ri]))
        cust = routes[src_ri].pop(src_pos)
        
        # 대상 경로에 삽입
        dst_ri = np.random.randint(0, len(routes))
        while dst_ri == src_ri and len(routes) > 1:
            dst_ri = np.random.randint(0, len(routes))
        
        dst_pos = np.random.randint(0, len(routes[dst_ri]) + 1)
        routes[dst_ri].insert(dst_pos, cust)
        
        # 빈 경로 제거
        routes = [r for r in routes if len(r) > 0]
        
        # 경로 → 시퀀스
        result = [0]
        for route in routes:
            result.extend(route)
            result.append(0)
        return result
    
    @staticmethod
    def _split_mutation(seq, problem):
        """
        Depot 구분자 삽입/제거: 경로를 분할하거나 합치기
        - 50% 확률로 기존 depot 구분자 제거 (경로 합침)
        - 50% 확률로 새 depot 구분자 삽입 (경로 분할)
        """
        routes = []
        current = []
        for v in seq:
            v = int(v)
            if v == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(v)
        if current:
            routes.append(current)
        
        if not routes:
            return seq
        
        if np.random.random() < 0.5 and len(routes) > 1:
            # 두 경로 합치기
            r1 = np.random.randint(0, len(routes))
            r2 = r1
            while r2 == r1 and len(routes) > 1:
                r2 = np.random.randint(0, len(routes))
            # r1에 r2를 추가
            routes[r1].extend(routes[r2])
            routes.pop(r2)
        else:
            # 경로 분할: 2명 이상인 경로를 선택하여 중간에 depot 삽입
            valid = [ri for ri in range(len(routes)) if len(routes[ri]) >= 2]
            if valid:
                ri = valid[np.random.randint(0, len(valid))]
                split_pos = np.random.randint(1, len(routes[ri]))
                new_route1 = routes[ri][:split_pos]
                new_route2 = routes[ri][split_pos:]
                routes[ri] = new_route1
                routes.insert(ri + 1, new_route2)
        
        # 경로 → 시퀀스
        result = [0]
        for route in routes:
            result.extend(route)
            result.append(0)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Progress Callback
# ═══════════════════════════════════════════════════════════════════════════

class VRPCallback(Callback):
    """Progress reporting callback"""
    
    def __init__(self):
        super().__init__()
        self.gen_count = 0
    
    def notify(self, algorithm):
        self.gen_count += 1
        if self.gen_count % 20 == 0 or self.gen_count == 1:
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                f1_min = np.min(F[:, 0])
                f2_min = np.min(F[:, 1])
                f3_min = np.min(F[:, 2])
                n_nds = len(algorithm.opt) if algorithm.opt is not None else 0
                print(f"  [Gen {self.gen_count:4d}] "
                      f"f1={f1_min:.2f} f2={f2_min:.4f} f3={f3_min:.2f} "
                      f"| Pareto={n_nds}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main NSGA-II-HI Solver
# ═══════════════════════════════════════════════════════════════════════════

def solve_cvrptw_nsga2_hi(dist_matrix, time_matrix, demands, time_windows,
                           service_times, depot_tw, capacity,
                           customer_locker_distances=None, num_active_lockers=0,
                           customer_types=None, node_individual_desired_times=None,
                           pop_size=100, n_gen=200, seed=42,
                           f2_norm_params=None,
                           crossover_type='ox', verbose=True):
    """
    NSGA-II-HI 기반 CVRPTW 다목적 최적화 솔버
    f2: Min-Max 정규화, range=0 fallback (D2D 전용 방지)
    """
    np.random.seed(seed)
    n_customers = len(demands)
    
    if n_customers == 0:
        return {
            'pareto_front': [],
            'selected_routes': [],
            'total_cost': 0.0,
            'moo_detail': {},
            'all_solutions': [],
        }
    
    if verbose:
        print(f"  🧬 NSGA-II-HI 시작: {n_customers}개 고객, "
              f"pop={pop_size}, gen={n_gen}, crossover={crossover_type}")
    
    # F2 Normalization (Min-Max, range=0 fallback)
    if f2_norm_params is not None and f2_norm_params.get('preset_mobility_only', False):
        # 시나리오 4: mobility range가 사전설정됨 → dissatisfaction만 ideal point 탐색
        preset_mob_L = f2_norm_params['mobility_L']
        preset_mob_U = f2_norm_params['mobility_U']
        
        # dissatisfaction만 단일목적 최적화로 탐색
        dis_params = initialize_f2_normalization(
            n_customers, dist_matrix, time_matrix, time_windows,
            service_times, depot_tw, demands, capacity,
            customer_locker_distances=customer_locker_distances,
            customer_types=customer_types,
            node_individual_desired_times=node_individual_desired_times
        )
        
        # mobility는 사전설정값 사용, dissatisfaction은 ideal point 탐색 결과 사용
        f2_norm_params = {
            'mobility_L': preset_mob_L,
            'mobility_U': preset_mob_U,
            'dissatisfaction_L': dis_params['dissatisfaction_L'],
            'dissatisfaction_U': dis_params['dissatisfaction_U'],
        }
        if verbose:
            p = f2_norm_params
            print(f"  ✅ f2 정규화 (사전설정 mobility): 이동불편[{p['mobility_L']:.3f}, {p['mobility_U']:.3f}], 불만족도[{p['dissatisfaction_L']:.3f}, {p['dissatisfaction_U']:.3f}]")
    elif f2_norm_params is None:
        f2_norm_params = initialize_f2_normalization(
            n_customers, dist_matrix, time_matrix, time_windows,
            service_times, depot_tw, demands, capacity,
            customer_locker_distances=customer_locker_distances,
            customer_types=customer_types,
            node_individual_desired_times=node_individual_desired_times
        )
        if verbose:
            p = f2_norm_params
            print(f"  ✅ f2 정규화 (Min-Max): 이동불편[{p['mobility_L']:.3f}, {p['mobility_U']:.3f}], 불만족도[{p['dissatisfaction_L']:.3f}, {p['dissatisfaction_U']:.3f}]")
    
    problem = CVRPTWProblem(
        n_customers=n_customers,
        dist_matrix=dist_matrix,
        time_matrix=time_matrix,
        demands=demands,
        time_windows=time_windows,
        service_times=service_times,
        depot_tw=depot_tw,
        capacity=capacity,
        customer_locker_distances=customer_locker_distances,
        num_active_lockers=num_active_lockers,
        customer_types=customer_types,
        node_individual_desired_times=node_individual_desired_times,
        f2_norm_params=f2_norm_params,
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Operator Configuration (논문 Section III)
    # ─────────────────────────────────────────────────────────────────
    
    # Adjust population size for small problems
    actual_pop_size = min(pop_size, max(20, n_customers * 2))
    actual_n_gen = min(n_gen, max(50, n_gen))
    
    if verbose:
        print(f"  📐 인코딩: 구분자 포함 시퀀스 (n_var={problem.n_var}, "
              f"max_vehicles={problem.max_vehicles})")
    
    algorithm = NSGA2(
        pop_size=actual_pop_size,
        sampling=SequenceSampling(),
        crossover=RouteExchangeCrossover(prob=0.8),
        mutation=SequenceMutation(prob=0.3),
        repair=SequenceRepair(improvement_rate=0.3),
        eliminate_duplicates=True,
    )
    
    termination = get_termination("n_gen", actual_n_gen)
    
    # ─────────────────────────────────────────────────────────────────
    # Run Optimization
    # ─────────────────────────────────────────────────────────────────
    t_start = time.time()
    
    # Callback 설정 (pymoo 버그 회피: verbose=False일 때는 callback 자체를 전달하지 않음)
    if verbose:
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
            callback=VRPCallback(),
        )
    else:
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
        )
    
    t_elapsed = time.time() - t_start
    
    # ─────────────────────────────────────────────────────────────────
    # Extract Results
    # ─────────────────────────────────────────────────────────────────
    if res.F is None or len(res.F) == 0:
        if verbose:
            print("  ⚠️ No feasible solutions found!")
        return {
            'pareto_front': [],
            'selected_routes': [],
            'total_cost': 0.0,
            'moo_detail': {},
            'all_solutions': [],
        }
    
    pareto_X = res.X
    n_pareto = len(res.F)
    
    # 각 Pareto 해를 구분자 시퀀스에서 디코딩하여 재평가
    pareto_front = []
    for pi in range(n_pareto):
        sol_seq = pareto_X[pi].astype(int)
        sol_routes = decode_sequence_to_routes(sol_seq)
        sol_routes = [r for r in sol_routes if len(r) > 0]
        if not sol_routes:
            pareto_front.append([1e9, 1e9, 1e9])
            continue
        f_vec_clean, _, _, _, _ = evaluate_moo_objectives(
            sol_routes, dist_matrix, time_matrix, time_windows,
            service_times, depot_tw,
            customer_locker_distances=customer_locker_distances,
            num_active_lockers=num_active_lockers,
            customer_types=customer_types,
            demands=demands,
            node_individual_desired_times=node_individual_desired_times,
            f2_norm_params=f2_norm_params,
        )
        pareto_front.append(f_vec_clean)
    
    if verbose:
        print(f"  ✅ NSGA-II-HI 완료: {t_elapsed:.1f}초, "
              f"Pareto 해 {len(pareto_front)}개")
        if pareto_front:
            print(f"     f1: [{min(r[0] for r in pareto_front):.2f}, "
                  f"{max(r[0] for r in pareto_front):.2f}] EUR")
            print(f"     f2: [{min(r[1] for r in pareto_front):.4f}, "
                  f"{max(r[1] for r in pareto_front):.4f}]")
            print(f"     f3: [{min(r[2] for r in pareto_front):.2f}, "
                  f"{max(r[2] for r in pareto_front):.2f}] kg CO2")
    
    # ─────────────────────────────────────────────────────────────────
    # Select best solution (f1 minimum - compatible with existing system)
    # ─────────────────────────────────────────────────────────────────
    f1_values = [f[0] for f in pareto_front]
    best_idx = int(np.argmin(f1_values))
    best_seq = pareto_X[best_idx].astype(int)
    best_routes = decode_sequence_to_routes(best_seq)
    best_routes = [r for r in best_routes if len(r) > 0]
    
    # Evaluate selected solution with full detail
    _, _, total_dist, _, moo_detail = evaluate_moo_objectives(
        best_routes, dist_matrix, time_matrix, time_windows,
        service_times, depot_tw,
        customer_locker_distances=customer_locker_distances,
        num_active_lockers=num_active_lockers,
        customer_types=customer_types,
        demands=demands,
        node_individual_desired_times=node_individual_desired_times,
        f2_norm_params=f2_norm_params,
    )
    
    # Convert routes to output format (with depot=0)
    routes_output = []
    for route in best_routes:
        route_with_depot = [0] + [c + 1 for c in route] + [0]  # 1-indexed customers
        routes_output.append(route_with_depot)
    
    # Build all_solutions for each Pareto point
    all_solutions = []
    for sol_idx in range(len(pareto_front)):
        sol_seq = pareto_X[sol_idx].astype(int)
        sol_routes = decode_sequence_to_routes(sol_seq)
        sol_routes = [r for r in sol_routes if len(r) > 0]
        if not sol_routes:
            continue
        _, _, sol_dist, _, sol_detail = evaluate_moo_objectives(
            sol_routes, dist_matrix, time_matrix, time_windows,
            service_times, depot_tw,
            customer_locker_distances=customer_locker_distances,
            num_active_lockers=num_active_lockers,
            customer_types=customer_types,
            demands=demands,
            node_individual_desired_times=node_individual_desired_times,
            f2_norm_params=f2_norm_params,
        )
        all_solutions.append({
            'idx': sol_idx,
            'f1': pareto_front[sol_idx][0],
            'f2': pareto_front[sol_idx][1],
            'f3': pareto_front[sol_idx][2],
            'detail': sol_detail,
            'total_distance': sol_dist,
        })
    
    return {
        'pareto_front': pareto_front,
        'selected_idx': best_idx,
        'selected_routes': routes_output,
        'total_cost': total_dist,
        'moo_detail': moo_detail,
        'all_solutions': all_solutions,
        'f2_norm_params': f2_norm_params,
        'elapsed_seconds': t_elapsed,
        'n_gen': actual_n_gen,
        'pop_size': actual_pop_size,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI Interface (Julia ↔ Python bridge)
# ═══════════════════════════════════════════════════════════════════════════

def load_problem_from_json(input_path):
    """JSON 파일에서 문제 데이터 로드"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists to proper types
    dist_matrix = [list(map(float, row)) for row in data['dist_matrix']]
    time_matrix = [list(map(float, row)) for row in data['time_matrix']]
    demands = list(map(int, data['demands']))
    time_windows = [tuple(map(int, tw)) for tw in data['time_windows']]
    service_times = list(map(int, data['service_times']))
    depot_tw = tuple(map(int, data['depot_tw']))
    capacity = int(data['capacity'])
    
    # Optional fields
    customer_locker_distances = None
    if 'customer_locker_distances' in data and data['customer_locker_distances']:
        customer_locker_distances = list(map(float, data['customer_locker_distances']))
    
    num_active_lockers = int(data.get('num_active_lockers', 0))
    
    customer_types = data.get('customer_types', None)
    
    node_individual_desired_times = None
    if 'node_individual_desired_times' in data and data['node_individual_desired_times']:
        node_individual_desired_times = [
            list(map(int, times)) if times else []
            for times in data['node_individual_desired_times']
        ]
    
    # Algorithm parameters
    pop_size = int(data.get('pop_size', 100))
    n_gen = int(data.get('n_gen', 200))
    seed = int(data.get('seed', 42))
    crossover_type = data.get('crossover_type', 'ox')
    
    # Pre-computed f2 normalization (from Julia)
    f2_norm_params = data.get('f2_norm_params', None)
    if f2_norm_params is not None:
        f2_norm_params = {
            'mobility_L': float(f2_norm_params.get('mobility_L', 0)),
            'mobility_U': float(f2_norm_params.get('mobility_U', 1)),
            'dissatisfaction_L': float(f2_norm_params.get('dissatisfaction_L', 0)),
            'dissatisfaction_U': float(f2_norm_params.get('dissatisfaction_U', 1)),
        }
    
    # Cost parameter overrides (from Julia sensitivity tests)
    cost_overrides = data.get('cost_overrides', None)
    if cost_overrides is not None:
        global MOO_FUEL_COST_PER_KM, MOO_VEHICLE_DAILY_COST, MOO_VEHICLE_CO2_PER_KM
        MOO_FUEL_COST_PER_KM = float(cost_overrides.get('fuel_cost_per_km', MOO_FUEL_COST_PER_KM))
        MOO_VEHICLE_DAILY_COST = float(cost_overrides.get('vehicle_daily_cost', MOO_VEHICLE_DAILY_COST))
        MOO_VEHICLE_CO2_PER_KM = float(cost_overrides.get('vehicle_co2_per_km', MOO_VEHICLE_CO2_PER_KM))
    
    return {
        'dist_matrix': dist_matrix,
        'time_matrix': time_matrix,
        'demands': demands,
        'time_windows': time_windows,
        'service_times': service_times,
        'depot_tw': depot_tw,
        'capacity': capacity,
        'customer_locker_distances': customer_locker_distances,
        'num_active_lockers': num_active_lockers,
        'customer_types': customer_types,
        'node_individual_desired_times': node_individual_desired_times,
        'pop_size': pop_size,
        'n_gen': n_gen,
        'seed': seed,
        'crossover_type': crossover_type,
        'f2_norm_params': f2_norm_params,
    }


def save_result_to_json(result, output_path):
    """결과를 JSON 파일로 저장"""
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert(v) for v in obj]
        return obj
    
    result_clean = convert(result)
    
    with open(output_path, 'w') as f:
        json.dump(result_clean, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='NSGA-II-HI CVRPTW Solver')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--quiet', action='store_true', default=False)
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("═" * 60)
        print("🧬 NSGA-II-HI CVRPTW Solver (pymoo)")
        print("═" * 60)
    
    # Load problem
    problem_data = load_problem_from_json(args.input)
    
    if verbose:
        n = len(problem_data['demands'])
        print(f"  문제 크기: {n}개 고객")
        print(f"  차량 용량: {problem_data['capacity']}")
        print(f"  팝 크기: {problem_data['pop_size']}, 세대: {problem_data['n_gen']}")
    
    # Solve
    result = solve_cvrptw_nsga2_hi(
        dist_matrix=problem_data['dist_matrix'],
        time_matrix=problem_data['time_matrix'],
        demands=problem_data['demands'],
        time_windows=problem_data['time_windows'],
        service_times=problem_data['service_times'],
        depot_tw=problem_data['depot_tw'],
        capacity=problem_data['capacity'],
        customer_locker_distances=problem_data['customer_locker_distances'],
        num_active_lockers=problem_data['num_active_lockers'],
        customer_types=problem_data['customer_types'],
        node_individual_desired_times=problem_data['node_individual_desired_times'],
        pop_size=problem_data['pop_size'],
        n_gen=problem_data['n_gen'],
        seed=problem_data['seed'],
        f2_norm_params=problem_data['f2_norm_params'],
        crossover_type=problem_data['crossover_type'],
        verbose=verbose,
    )
    
    # Save result
    save_result_to_json(result, args.output)
    
    if verbose:
        print(f"\n  📁 결과 저장: {args.output}")
        print("═" * 60)


if __name__ == '__main__':
    main()
