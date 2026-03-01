# MOO Multi-Trip CVRP Budapest

## 개요
부다페스트 5구역을 대상으로 **NSGA-II 기반 Multi-Objective Optimization (MOO)** VRP를 수행하는 Julia 시뮬레이터입니다.

## 목적함수 (3개 벡터, 가중합 아님!)

| 목적함수 | 구성요소 | 단위 |
|----------|----------|------|
| **f1 (기업)** | (이동거리 × 유류비) + (차량수 × 일일운행비) + (운행시간 × 운전자임금) | EUR |
| **f2 (고객)** | (평균실제이동거리 × 불편비용) + 총 불만족도 | EUR + 무차원 |
| **f3 (사회)** | (배송차량 CO2 + 고객 차량이용분 CO2 + 락커 CO2) × 탄소비용 | EUR |

### f1 기업 비용

| 구성요소 | 파라미터 | 근거 |
|----------|----------|------|
| 유류비 | 0.15 EUR/km | 유럽 경유 가격 기준 |
| 일일 운행비 | **50 EUR/대/일** | 감가상각(25) + 보험(10) + 유지보수(15) |
| 운전자 임금 | 12 EUR/h | 헝가리 배송기사 시급 |

### f2 고객 비용

**1. 평균 실제 이동거리** (MODE_SHARE 기반):
```
실제이동거리 = (도보+자전거+전용차량) × 왕복 + 연계차량 × 편도
```
- 도보/자전거/전용차량: **왕복** (편도 × 2)
- 연계차량: **편도만** (다른 목적 이동 중 들르므로)

**2. 고객 만족도** (Chen et al., IEEE CEC 2024 논문 그대로):
```
S_j = 1 / (1 + w_j)²
총 불만족도 = Σ (1 - S_j)
```
- `w_j = max(0, arrival_time - desired_time)` (대기시간)
- 논문 공식 그대로 적용, 스케일 조정 없음

**3. f2 정규화** (Upper-Lower-Bound Min-Max, Marler & Arora 2005, Eq.7):
- mobility, dissatisfaction 각각 단일 목적 최적화(GA)로 ideal point x_mob*, x_dis* 탐색
- z_L = ideal point (min), z_U = Pareto maximum (ideal 해에서의 max)
- z' = (z - z_L) / (z_U - z_L + ε), range≈0이면 z'=0 (변별력 없음, D2D 전용 등)
- 시나리오 전환 시 reset → 시나리오별 재계산

### f3 사회 비용 (CO2)

**거리별 이동수단 비율** (MODE_SHARE):
| 편도거리(km) | 도보(%) | 자전거(%) | 전용차량(%) | 연계차량(%) |
|--------------|---------|-----------|-------------|-------------|
| 0.0 | 77.8 | 15.8 | 2.3 | 4.1 |
| 1.0 | 31.1 | 40.0 | 10.9 | 18.0 |
| 2.0 | 5.1 | 41.3 | 21.2 | 32.4 |
| 5.0 | 0.0 | 11.7 | 40.4 | 48.0 |

**CO2 계산** (이동거리 기반):
- 도보/자전거: CO2 = 0
- 전용차량: **왕복거리** × 0.15 kg/km
- 연계차량: **편도거리** × 0.15 kg/km (다른 목적 이동에 연계)

## 주요 특징

- **NSGA-II-HI**: pymoo 기반 다목적 최적화
- **Pareto Front**: 3개 목적함수의 비지배 해 집합 생성
- **시나리오 1-5**: 배송사별 락커 공유 규칙
- **시간창 지원**: D2D 배송 및 락커 서비스 시간창
- **OSRM**: 실제 도로망 기반 거리/시간 계산
- **Monte Carlo**: 다중 시뮬레이션 (omega)

## 파일 구조

```
├── moo_multitrip_cvrp_budapest.jl    # 메인 시뮬레이터
├── nsga2_moo_backend.jl              # NSGA-II MOO 백엔드
├── moo_pareto_visualization.jl       # Pareto front 시각화
├── osrm_client.jl                    # OSRM 도로망 클라이언트
├── Project.toml                      # Julia 의존성
└── README.md
```

## Pareto Front 조회

NSGA-II 최적화 후 Pareto front는 백엔드에 저장됩니다:

```julia
using Main.nsga2_moo_backend

# Pareto front 조회 (n_solutions × 3 matrix)
pf = get_moo_pareto_front()

# 각 열: [f1(기업), f2(고객), f3(사회)]
println("Pareto front size: $(size(pf))")
println("f1 range: $(minimum(pf[:, 1])) ~ $(maximum(pf[:, 1])) EUR")
println("f2 range: $(minimum(pf[:, 2])) ~ $(maximum(pf[:, 2])) EUR")
println("f3 range: $(minimum(pf[:, 3])) ~ $(maximum(pf[:, 3])) EUR")
```

## 실행 방법

```powershell
# 기본 실행 (Windows PowerShell)
julia --project=. moo_multitrip_cvrp_budapest.jl

# 특정 시나리오 실행
julia --project=. -e 'include("moo_multitrip_cvrp_budapest.jl"); run_scenarios_v2([1,2,3,5], 10)'
```

## 시나리오

| 번호 | 이름 | 설명 |
|------|------|------|
| 1 | D2D | Door-to-Door 배송만 |
| 2 | DPL | Dedicated Private Locker (자사 락커만) |
| 3 | SPL | Shared Private Locker (모든 락커 공유) |
| 4 | OPL | Optimized Public Locker (SLRP 최적화) |
| 5 | PSPL | Partially Shared Private Locker |

## 의존성

- Julia 1.9+
- pymoo (NSGA-II-HI, Python)
- PyCall.jl
- Distances.jl
- OSRM Docker (도로망 거리 계산)

---

## Windows 환경 설정 가이드

### 1. Julia 설치
[https://julialang.org/downloads/](https://julialang.org/downloads/) 에서 Windows 인스톨러 다운로드 후 설치

### 2. Python 설치
[https://www.python.org/downloads/](https://www.python.org/downloads/) 에서 Python 3.10 이상 설치  
> **설치 시 "Add Python to PATH" 반드시 체크**

```powershell
# pymoo 및 의존 패키지 설치
pip install pymoo numpy
```

### 3. Julia 패키지 설치

```powershell
# PowerShell에서 Julia REPL 실행
julia

# Julia REPL에서 실행
]add PyCall JuMP Gurobi Plots DataFrames CSV JSON3 HTTP Clustering Distances HypothesisTests StatsPlots

# PyCall이 현재 Python을 사용하도록 설정
using PyCall
```

### 4. Gurobi 설치 (선택사항 — 최적화 솔버)
[https://www.gurobi.com/downloads/](https://www.gurobi.com/downloads/) 에서 Windows용 설치 후 라이선스 활성화

### 5. OSRM 도로망 서버 실행 (Docker Desktop 필요)

[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 설치 후:

```powershell
# 차량 프로파일 (포트 5001)
docker run -p 5001:5000 osrm/osrm-backend osrm-routed --algorithm mld /data/hungary-latest.osrm

# 도보 프로파일 (포트 5002)
docker run -p 5002:5000 osrm/osrm-backend osrm-routed --algorithm mld /data/hungary-latest.osrm
```

> OSRM 없이 실행 시 자동으로 유클리드 거리로 fallback 됩니다.

### 6. 실행

```powershell
cd C:\path\to\project
julia --project=. moo_multitrip_cvrp_budapest.jl
```

### 결과 위치
실행 후 결과 파일은 `%USERPROFILE%\Desktop\runs\` 폴더에 저장됩니다.  
(`C:\Users\사용자명\Desktop\runs\`)
