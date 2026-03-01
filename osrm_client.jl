# ═══════════════════════════════════════════════════════════════════════════════
# OSRM Client for Julia
# 도로망 기반 거리 및 경로 계산
# ═══════════════════════════════════════════════════════════════════════════════

using HTTP
using JSON3

# ═══════════════════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════════════════

const OSRM_CAR_URL = "http://localhost:5001"   # 차량 프로파일
const OSRM_FOOT_URL = "http://localhost:5002"  # 도보 프로파일

const OSRM_CACHE = Dict{String, Any}()  # 캐시 (key: query, value: result)
const OSRM_CACHE_LOCK = ReentrantLock()

# ═══════════════════════════════════════════════════════════════════════════════
# OSRM 서버 연결 필수 확인 (시작 시 호출)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    assert_osrm_available()

OSRM 차량(5001) 및 도보(5002) 서버가 응답하는지 확인.
하나라도 오프라인이면 즉시 에러로 프로그램을 중단한다.
유클리드/Haversine 대체는 허용하지 않는다.
"""
function assert_osrm_available()
    errors = String[]
    for (label, url, port) in [("차량(car)", OSRM_CAR_URL, 5001),
                                ("도보(foot)", OSRM_FOOT_URL, 5002)]
        try
            resp = HTTP.get("$url/health"; connect_timeout=3, readtimeout=5)
            if resp.status == 200
                println("  ✅ OSRM $label 서버 정상 (포트 $port)")
            else
                push!(errors, "OSRM $label 서버 비정상 응답 (HTTP $(resp.status), 포트 $port)")
            end
        catch e
            push!(errors, "OSRM $label 서버 오프라인 (포트 $port): $e")
        end
    end

    if !isempty(errors)
        println("\n" * "═"^70)
        println("  ❌ OSRM 서버에 연결할 수 없습니다. 실행을 중단합니다.")
        println("═"^70)
        for err in errors
            println("  • $err")
        end
        println("""
\n  📌 해결 방법:
  Docker Desktop을 실행한 뒤 아래 명령어로 OSRM 서버를 시작하세요:

  docker run -d -p 5001:5000 osrm/osrm-backend \\
      osrm-routed --algorithm mld /data/hungary-latest.osrm

  docker run -d -p 5002:5000 osrm/osrm-backend \\
      osrm-routed --algorithm mld /data/hungary-latest.osrm

  또는: .\\setup_windows.ps1 을 실행하면 OSRM 상태를 확인할 수 있습니다.
""")
        println("═"^70)
        error("OSRM 서버 미연결 — 실행 불가. 위 안내에 따라 OSRM을 먼저 실행하세요.")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 기본 API 함수
# ═══════════════════════════════════════════════════════════════════════════════

"""
OSRM 서버 URL 반환
- profile: "car" 또는 "foot"
"""
function get_osrm_url(profile::String)
    if profile == "car"
        return OSRM_CAR_URL
    elseif profile == "foot"
        return OSRM_FOOT_URL
    else
        error("Unknown profile: $profile. Use 'car' or 'foot'")
    end
end

"""
OSRM nearest 서비스 - 가장 가까운 도로 위 점 찾기
- lat, lon: 원래 좌표
- profile: "car" 또는 "foot"
반환: (snapped_lat, snapped_lon) 또는 nothing
"""
function osrm_nearest(lat::Float64, lon::Float64; profile::String="foot")
    cache_key = "nearest_$(profile)_$(lat)_$(lon)"
    
    lock(OSRM_CACHE_LOCK) do
        if haskey(OSRM_CACHE, cache_key)
            return OSRM_CACHE[cache_key]
        end
    end
    
    base_url = get_osrm_url(profile)
    url = "$base_url/nearest/v1/$profile/$lon,$lat?number=1"
    
    try
        response = HTTP.get(url; connect_timeout=5, readtimeout=10)
        data = JSON3.read(String(response.body))
        
        if data["code"] == "Ok" && length(data["waypoints"]) > 0
            waypoint = data["waypoints"][1]
            snapped_lon, snapped_lat = waypoint["location"]
            result = (Float64(snapped_lat), Float64(snapped_lon))
            
            lock(OSRM_CACHE_LOCK) do
                OSRM_CACHE[cache_key] = result
            end
            
            return result
        end
    catch e
        @warn "OSRM nearest failed: $e"
    end
    
    return nothing
end

"""
OSRM route 서비스 - 두 점 사이의 도로 거리 계산
- from_pos: (lat, lon)
- to_pos: (lat, lon)
- profile: "car" 또는 "foot"
반환: 거리 (km) 또는 Inf (경로 없음)
"""
function osrm_route_distance(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64}; profile::String="car")
    from_lat, from_lon = from_pos
    to_lat, to_lon = to_pos
    
    cache_key = "route_$(profile)_$(from_lat)_$(from_lon)_$(to_lat)_$(to_lon)"
    
    lock(OSRM_CACHE_LOCK) do
        if haskey(OSRM_CACHE, cache_key)
            return OSRM_CACHE[cache_key]
        end
    end
    
    base_url = get_osrm_url(profile)
    url = "$base_url/route/v1/$profile/$from_lon,$from_lat;$to_lon,$to_lat?overview=false"
    
    try
        response = HTTP.get(url; connect_timeout=5, readtimeout=10)
        data = JSON3.read(String(response.body))
        
        if data["code"] == "Ok" && length(data["routes"]) > 0
            distance_m = data["routes"][1]["distance"]
            distance_km = Float64(distance_m) / 1000.0
            
            lock(OSRM_CACHE_LOCK) do
                OSRM_CACHE[cache_key] = distance_km
            end
            
            return distance_km
        end
    catch e
        @warn "OSRM route failed: $e"
    end
    
    return Inf
end

"""
OSRM table 서비스 - 여러 점 사이의 거리 행렬 계산
- locations: [(lat, lon), ...]
- profile: "car" 또는 "foot"
반환: 거리 행렬 (km) NxN
"""
function osrm_distance_matrix(locations::Vector{Tuple{Float64,Float64}}; profile::String="car")
    dist_matrix, _ = osrm_distance_and_duration_matrix(locations; profile=profile)
    return dist_matrix
end

"""
OSRM table 서비스 - 여러 점 사이의 거리 및 시간 행렬 계산
- locations: [(lat, lon), ...]
- profile: "car" 또는 "foot"
반환: (거리 행렬 (미터), 시간 행렬 (초)) 튜플
"""
function osrm_distance_and_duration_matrix(locations::Vector{Tuple{Float64,Float64}}; profile::String="car")
    n = length(locations)
    if n == 0
        return Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0)
    end
    
    # 좌표 문자열 생성
    coords_str = join(["$(lon),$(lat)" for (lat, lon) in locations], ";")
    
    base_url = get_osrm_url(profile)
    # distance와 duration 둘 다 요청
    url = "$base_url/table/v1/$profile/$coords_str?annotations=distance,duration"
    
    try
        response = HTTP.get(url; connect_timeout=30, readtimeout=60)
        data = JSON3.read(String(response.body))
        
        if data["code"] == "Ok"
            distances = data["distances"]
            durations = data["durations"]
            
            dist_matrix = zeros(Float64, n, n)
            time_matrix = zeros(Float64, n, n)
            
            for i in 1:n
                for j in 1:n
                    # 거리 (미터 → km)
                    dist_m = distances[i][j]
                    if dist_m === nothing
                        dist_matrix[i, j] = Inf
                    else
                        dist_matrix[i, j] = Float64(dist_m) / 1000.0  # km로 변환
                    end
                    
                    # 시간 (초)
                    dur_s = durations[i][j]
                    if dur_s === nothing
                        time_matrix[i, j] = Inf
                    else
                        time_matrix[i, j] = Float64(dur_s)  # 초 단위 유지
                    end
                end
            end
            
            return dist_matrix, time_matrix
        end
    catch e
        @warn "OSRM table failed: $e"
    end
    
    # 실패 시 Inf 행렬 반환
    return fill(Inf, n, n), fill(Inf, n, n)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 스냅 함수
# ═══════════════════════════════════════════════════════════════════════════════

"""
고객 위치 스냅 (도보망 기준)
"""
function snap_customer_location(lat::Float64, lon::Float64)
    result = osrm_nearest(lat, lon; profile="foot")
    if result === nothing
        @warn "Customer snap failed, using original location: ($lat, $lon)"
        return (lat, lon)
    end
    return result
end

"""
락커 위치 스냅 (양쪽 검증: car + foot)
- 둘 다 접근 가능한지 확인
"""
function snap_locker_location(lat::Float64, lon::Float64)
    # 차량 스냅
    car_snap = osrm_nearest(lat, lon; profile="car")
    # 도보 스냅
    foot_snap = osrm_nearest(lat, lon; profile="foot")
    
    if car_snap === nothing && foot_snap === nothing
        @warn "Locker snap failed completely, using original: ($lat, $lon)"
        return (lat, lon)
    end
    
    if car_snap === nothing
        return foot_snap
    end
    
    if foot_snap === nothing
        return car_snap
    end
    
    # 두 스냅 위치 차이 확인 (간단한 유클리드 거리)
    diff = sqrt((car_snap[1] - foot_snap[1])^2 + (car_snap[2] - foot_snap[2])^2) * 111.0  # km
    
    if diff < 0.05  # 50m 이내
        # 둘 다 가능 → foot 기준 사용 (고객 도보 우선)
        return foot_snap
    else
        # 차이가 크면 foot 위치에서 car 경로 가능한지 검증
        # (간단히 foot 기준 사용)
        @warn "Car/Foot snap differ by $(round(diff*1000, digits=0))m, using foot snap"
        return foot_snap
    end
end

"""
데포 위치 스냅 (차량 도로 기준)
"""
function snap_depot_location(lat::Float64, lon::Float64)
    result = osrm_nearest(lat, lon; profile="car")
    if result === nothing
        @warn "Depot snap failed, using original location: ($lat, $lon)"
        return (lat, lon)
    end
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# 거리 계산 함수
# ═══════════════════════════════════════════════════════════════════════════════

"""
차량 이동 거리 (car profile)
"""
function road_distance_car(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    return osrm_route_distance(from_pos, to_pos; profile="car")
end

"""
도보 이동 거리 (foot profile)
"""
function road_distance_foot(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    return osrm_route_distance(from_pos, to_pos; profile="foot")
end

"""
차량 경로 거리 행렬 (여러 위치)
"""
function road_distance_matrix_car(locations::Vector{Tuple{Float64,Float64}})
    return osrm_distance_matrix(locations; profile="car")
end

"""
도보 경로 거리 행렬 (여러 위치)
"""
function road_distance_matrix_foot(locations::Vector{Tuple{Float64,Float64}})
    return osrm_distance_matrix(locations; profile="foot")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 경로 Geometry 함수 (시각화용)
# ═══════════════════════════════════════════════════════════════════════════════

"""
OSRM route 서비스 - 경로 geometry 가져오기
- from_pos: (lat, lon)
- to_pos: (lat, lon)
- profile: "car" 또는 "foot"
반환: Vector{Tuple{Float64,Float64}} (lat, lon 순서) 또는 빈 벡터
"""
function osrm_route_geometry(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64}; profile::String="car")
    from_lat, from_lon = from_pos
    to_lat, to_lon = to_pos
    
    cache_key = "geometry_$(profile)_$(from_lat)_$(from_lon)_$(to_lat)_$(to_lon)"
    
    lock(OSRM_CACHE_LOCK) do
        if haskey(OSRM_CACHE, cache_key)
            return OSRM_CACHE[cache_key]
        end
    end
    
    base_url = get_osrm_url(profile)
    # overview=full: 전체 경로, geometries=geojson: GeoJSON 형식
    url = "$base_url/route/v1/$profile/$from_lon,$from_lat;$to_lon,$to_lat?overview=full&geometries=geojson"
    
    try
        response = HTTP.get(url; connect_timeout=5, readtimeout=10)
        data = JSON3.read(String(response.body))
        
        if data["code"] == "Ok" && length(data["routes"]) > 0
            geometry = data["routes"][1]["geometry"]
            coordinates = geometry["coordinates"]
            
            # GeoJSON은 [lon, lat] 순서 → (lat, lon)으로 변환
            path = Tuple{Float64,Float64}[]
            for coord in coordinates
                lon, lat = coord
                push!(path, (Float64(lat), Float64(lon)))
            end
            
            lock(OSRM_CACHE_LOCK) do
                OSRM_CACHE[cache_key] = path
            end
            
            return path
        end
    catch e
        @warn "OSRM geometry failed: $e"
    end
    
    # 실패 시 직선 경로 반환
    return [from_pos, to_pos]
end

"""
차량 경로 geometry
"""
function road_geometry_car(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    return osrm_route_geometry(from_pos, to_pos; profile="car")
end

"""
도보 경로 geometry
"""
function road_geometry_foot(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    return osrm_route_geometry(from_pos, to_pos; profile="foot")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════════

"""
캐시 초기화
"""
function clear_osrm_cache()
    lock(OSRM_CACHE_LOCK) do
        empty!(OSRM_CACHE)
    end
    println("OSRM cache cleared")
end

"""
캐시 크기 확인
"""
function osrm_cache_size()
    lock(OSRM_CACHE_LOCK) do
        return length(OSRM_CACHE)
    end
end

"""
OSRM 서버 연결 테스트
"""
function test_osrm_connection()
    println("Testing OSRM connection...")
    
    # 부다페스트 중심부 테스트 좌표
    test_lat, test_lon = 47.4979, 19.0402
    
    # Car 테스트
    car_snap = osrm_nearest(test_lat, test_lon; profile="car")
    if car_snap !== nothing
        println("✅ Car profile OK: ($test_lat, $test_lon) → $(car_snap)")
    else
        println("❌ Car profile FAILED")
    end
    
    # Foot 테스트
    foot_snap = osrm_nearest(test_lat, test_lon; profile="foot")
    if foot_snap !== nothing
        println("✅ Foot profile OK: ($test_lat, $test_lon) → $(foot_snap)")
    else
        println("❌ Foot profile FAILED")
    end
    
    # 거리 테스트
    from_pos = (47.4979, 19.0402)
    to_pos = (47.5025, 19.0515)
    
    car_dist = road_distance_car(from_pos, to_pos)
    foot_dist = road_distance_foot(from_pos, to_pos)
    
    println("Distance test ($from_pos → $to_pos):")
    println("  Car: $(round(car_dist, digits=3)) km")
    println("  Foot: $(round(foot_dist, digits=3)) km")
    
    return car_snap !== nothing && foot_snap !== nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# 거리 행렬 사전 계산 시스템
# ═══════════════════════════════════════════════════════════════════════════════

"""
전역 거리 및 시간 행렬 저장소 (이중 스냅 지원)
- car_matrix: 차량 거리 행렬 (km, car 스냅 위치 기준)
- car_time_matrix: 차량 이동 시간 행렬 (초, car 스냅 위치 기준)
- foot_matrix: 도보 거리 행렬 (km, foot 스냅 위치 기준)
- foot_time_matrix: 도보 이동 시간 행렬 (초, foot 스냅 위치 기준)
- car_node_index: car 스냅 좌표 → 인덱스 매핑
- foot_node_index: foot 스냅 좌표 → 인덱스 매핑
- car_snapped_locations: 원래 → car 스냅 위치
- foot_snapped_locations: 원래 → foot 스냅 위치
"""
mutable struct DistanceMatrixStore
    car_matrix::Matrix{Float64}
    car_time_matrix::Matrix{Float64}      # NEW: 차량 이동 시간 (초)
    foot_matrix::Matrix{Float64}
    foot_time_matrix::Matrix{Float64}     # NEW: 도보 이동 시간 (초)
    car_node_index::Dict{Tuple{Float64,Float64}, Int}
    foot_node_index::Dict{Tuple{Float64,Float64}, Int}
    car_snapped_locations::Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}  # 원래 → car 스냅
    foot_snapped_locations::Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}} # 원래 → foot 스냅
    initialized::Bool
end

const DISTANCE_STORE = DistanceMatrixStore(
    Matrix{Float64}(undef, 0, 0),
    Matrix{Float64}(undef, 0, 0),  # car_time_matrix
    Matrix{Float64}(undef, 0, 0),
    Matrix{Float64}(undef, 0, 0),  # foot_time_matrix
    Dict{Tuple{Float64,Float64}, Int}(),
    Dict{Tuple{Float64,Float64}, Int}(),
    Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}(),
    Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}(),
    false
)
const DISTANCE_STORE_LOCK = ReentrantLock()

"""
거리 행렬 초기화 (모든 노드에 대해 사전 계산)
- depot_locations: 데포 위치들 [(lat, lon), ...]
- locker_locations: 락커 위치들 [(lat, lon), ...]
- customer_locations: 고객 위치들 [(lat, lon), ...]
"""
function initialize_distance_matrix(
    depot_locations::Vector{Tuple{Float64,Float64}},
    locker_locations::Vector{Tuple{Float64,Float64}},
    customer_locations::Vector{Tuple{Float64,Float64}}
)
    println("🗺️ 거리 행렬 사전 계산 시작 (이중 스냅 모드)...")
    start_time = time()
    
    # 모든 위치 수집
    all_original = vcat(depot_locations, locker_locations, customer_locations)
    n_total = length(all_original)
    
    println("   📍 총 노드 수: $(n_total) (데포: $(length(depot_locations)), 락커: $(length(locker_locations)), 고객: $(length(customer_locations)))")
    
    # 이중 스냅 처리 (car + foot 둘 다)
    car_snapped_locations = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()
    foot_snapped_locations = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()
    all_car_snapped = Tuple{Float64,Float64}[]
    all_foot_snapped = Tuple{Float64,Float64}[]
    
    println("   🔄 이중 스냅 중 (car + foot)...")
    for (i, (lat, lon)) in enumerate(all_original)
        original = (lat, lon)
        
        # Car 프로파일 스냅
        car_snap = osrm_nearest(lat, lon; profile="car")
        if car_snap === nothing
            car_snap = original
        end
        car_snapped_locations[original] = car_snap
        push!(all_car_snapped, car_snap)
        
        # Foot 프로파일 스냅
        foot_snap = osrm_nearest(lat, lon; profile="foot")
        if foot_snap === nothing
            foot_snap = original
        end
        foot_snapped_locations[original] = foot_snap
        push!(all_foot_snapped, foot_snap)
        
        if i % 100 == 0
            print("\r   스냅 진행: $i / $(n_total)")
        end
    end
    println("\r   ✅ 이중 스냅 완료: $(n_total)개 위치")
    
    # 중복 제거된 고유 위치
    unique_car_snapped = unique(all_car_snapped)
    unique_foot_snapped = unique(all_foot_snapped)
    println("   📊 고유 위치 수: car=$(length(unique_car_snapped)), foot=$(length(unique_foot_snapped))")
    
    # Car 노드 인덱스 생성
    car_node_index = Dict{Tuple{Float64,Float64}, Int}()
    for (i, pos) in enumerate(unique_car_snapped)
        car_node_index[pos] = i
    end
    
    # Foot 노드 인덱스 생성
    foot_node_index = Dict{Tuple{Float64,Float64}, Int}()
    for (i, pos) in enumerate(unique_foot_snapped)
        foot_node_index[pos] = i
    end
    
    # 거리 및 시간 행렬 계산 (OSRM table API 사용)
    println("   🚗 차량 거리/시간 행렬 계산 중 (car 스냅 위치 기준)...")
    car_matrix, car_time_matrix = osrm_distance_and_duration_matrix(unique_car_snapped; profile="car")
    println("   ✅ 차량 거리/시간 행렬 완료: $(length(unique_car_snapped)) × $(length(unique_car_snapped))")
    
    println("   🚶 도보 거리/시간 행렬 계산 중 (foot 스냅 위치 기준)...")
    foot_matrix, foot_time_matrix = osrm_distance_and_duration_matrix(unique_foot_snapped; profile="foot")
    println("   ✅ 도보 거리/시간 행렬 완료: $(length(unique_foot_snapped)) × $(length(unique_foot_snapped))")
    
    # 저장
    lock(DISTANCE_STORE_LOCK) do
        DISTANCE_STORE.car_matrix = car_matrix
        DISTANCE_STORE.car_time_matrix = car_time_matrix
        DISTANCE_STORE.foot_matrix = foot_matrix
        DISTANCE_STORE.foot_time_matrix = foot_time_matrix
        DISTANCE_STORE.car_node_index = car_node_index
        DISTANCE_STORE.foot_node_index = foot_node_index
        DISTANCE_STORE.car_snapped_locations = car_snapped_locations
        DISTANCE_STORE.foot_snapped_locations = foot_snapped_locations
        DISTANCE_STORE.initialized = true
    end
    
    elapsed = round(time() - start_time, digits=2)
    println("🗺️ 거리 행렬 사전 계산 완료! (소요 시간: $(elapsed)초)")
    
    return true
end

"""
사전 계산된 거리 조회 (차량) - car 스냅 위치 사용
"""
function get_precomputed_car_distance(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        if !DISTANCE_STORE.initialized
            @warn "Distance matrix not initialized, using OSRM direct query"
            return road_distance_car(from_pos, to_pos)
        end
        
        # car 스냅된 위치로 변환
        from_snapped = get(DISTANCE_STORE.car_snapped_locations, from_pos, from_pos)
        to_snapped = get(DISTANCE_STORE.car_snapped_locations, to_pos, to_pos)
        
        # car 인덱스 조회
        from_idx = get(DISTANCE_STORE.car_node_index, from_snapped, 0)
        to_idx = get(DISTANCE_STORE.car_node_index, to_snapped, 0)
        
        if from_idx == 0 || to_idx == 0
            # 행렬에 없는 위치 → 직접 쿼리
            return road_distance_car(from_snapped, to_snapped)
        end
        
        return DISTANCE_STORE.car_matrix[from_idx, to_idx]
    end
end

"""
사전 계산된 이동 시간 조회 (차량) - car 스냅 위치 사용
반환: 이동 시간 (초)
"""
function get_precomputed_car_duration(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        if !DISTANCE_STORE.initialized
            @warn "Time matrix not initialized"
            # 폴백: 거리를 평균 속도로 나눈 시간 (30km/h 가정)
            dist_km = road_distance_car(from_pos, to_pos)
            return dist_km / 30.0 * 3600.0  # 초로 변환
        end
        
        # car 스냅된 위치로 변환
        from_snapped = get(DISTANCE_STORE.car_snapped_locations, from_pos, from_pos)
        to_snapped = get(DISTANCE_STORE.car_snapped_locations, to_pos, to_pos)
        
        # car 인덱스 조회
        from_idx = get(DISTANCE_STORE.car_node_index, from_snapped, 0)
        to_idx = get(DISTANCE_STORE.car_node_index, to_snapped, 0)
        
        if from_idx == 0 || to_idx == 0
            # 행렬에 없는 위치 → 거리 기반 추정
            dist_km = road_distance_car(from_snapped, to_snapped)
            return dist_km / 30.0 * 3600.0
        end
        
        return DISTANCE_STORE.car_time_matrix[from_idx, to_idx]
    end
end

"""
사전 계산된 거리 및 이동 시간 조회 (차량) - car 스냅 위치 사용
반환: (거리 km, 이동 시간 초)
"""
function get_precomputed_car_distance_and_duration(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        if !DISTANCE_STORE.initialized
            @warn "Distance/Time matrix not initialized"
            dist_km = road_distance_car(from_pos, to_pos)
            duration_s = dist_km / 30.0 * 3600.0
            return (dist_km, duration_s)
        end
        
        # car 스냅된 위치로 변환
        from_snapped = get(DISTANCE_STORE.car_snapped_locations, from_pos, from_pos)
        to_snapped = get(DISTANCE_STORE.car_snapped_locations, to_pos, to_pos)
        
        # car 인덱스 조회
        from_idx = get(DISTANCE_STORE.car_node_index, from_snapped, 0)
        to_idx = get(DISTANCE_STORE.car_node_index, to_snapped, 0)
        
        if from_idx == 0 || to_idx == 0
            dist_km = road_distance_car(from_snapped, to_snapped)
            duration_s = dist_km / 30.0 * 3600.0
            return (dist_km, duration_s)
        end
        
        return (DISTANCE_STORE.car_matrix[from_idx, to_idx], 
                DISTANCE_STORE.car_time_matrix[from_idx, to_idx])
    end
end

"""
사전 계산된 거리 조회 (도보) - foot 스냅 위치 사용 (도로망 거리만)
"""
function get_precomputed_foot_distance(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        if !DISTANCE_STORE.initialized
            @warn "Distance matrix not initialized, using OSRM direct query"
            return road_distance_foot(from_pos, to_pos)
        end
        
        # foot 스냅된 위치로 변환
        from_snapped = get(DISTANCE_STORE.foot_snapped_locations, from_pos, from_pos)
        to_snapped = get(DISTANCE_STORE.foot_snapped_locations, to_pos, to_pos)
        
        # foot 인덱스 조회
        from_idx = get(DISTANCE_STORE.foot_node_index, from_snapped, 0)
        to_idx = get(DISTANCE_STORE.foot_node_index, to_snapped, 0)
        
        if from_idx == 0 || to_idx == 0
            # 행렬에 없는 위치 → 직접 쿼리
            return road_distance_foot(from_snapped, to_snapped)
        end
        
        return DISTANCE_STORE.foot_matrix[from_idx, to_idx]
    end
end

"""
Haversine 거리 계산 (km 단위)
- pos1, pos2: (lat, lon) 형식
"""
function haversine_distance_km(pos1::Tuple{Float64,Float64}, pos2::Tuple{Float64,Float64})
    lat1, lon1 = pos1
    lat2, lon2 = pos2
    
    R = 6371.0  # 지구 반경 (km)
    
    φ1 = deg2rad(lat1)
    φ2 = deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    
    return R * c
end

"""
Hybrid 도보 거리 계산 (골목 내 이동 반영)

최종 거리 = 스냅 직선거리(출발) + 도로망 거리 + 스냅 직선거리(도착)

[원래 위치] ──(직선)──> [foot 스냅] ══(도로망)══> [목적지 foot 스냅] ──(직선)──> [원래 목적지]
"""
function get_hybrid_foot_distance(from_pos::Tuple{Float64,Float64}, to_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        if !DISTANCE_STORE.initialized
            error("OSRM 거리 행렬이 초기화되지 않았습니다. OSRM 서버가 실행 중인지 확인하세요 (포트 5001, 5002). Haversine 대체는 허용되지 않습니다.")
        end
        
        # foot 스냅된 위치 가져오기
        from_snapped = get(DISTANCE_STORE.foot_snapped_locations, from_pos, from_pos)
        to_snapped = get(DISTANCE_STORE.foot_snapped_locations, to_pos, to_pos)
        
        # 1. 출발지 → 출발 스냅 위치 (직선거리, 골목 내 이동)
        snap_dist_from = haversine_distance_km(from_pos, from_snapped)
        
        # 2. 출발 스냅 → 도착 스냅 (도로망 거리)
        from_idx = get(DISTANCE_STORE.foot_node_index, from_snapped, 0)
        to_idx = get(DISTANCE_STORE.foot_node_index, to_snapped, 0)
        
        if from_idx == 0 || to_idx == 0
            # 행렬에 없는 위치 → 직접 쿼리
            road_dist = road_distance_foot(from_snapped, to_snapped)
        else
            road_dist = DISTANCE_STORE.foot_matrix[from_idx, to_idx]
        end
        
        # 3. 도착 스냅 → 도착지 (직선거리, 골목 내 이동)
        snap_dist_to = haversine_distance_km(to_snapped, to_pos)
        
        # 합계
        total_dist = snap_dist_from + road_dist + snap_dist_to
        
        return total_dist
    end
end

"""
스냅된 위치 조회 (car 프로파일)
"""
function get_car_snapped_location(original_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        return get(DISTANCE_STORE.car_snapped_locations, original_pos, original_pos)
    end
end

"""
스냅된 위치 조회 (foot 프로파일)
"""
function get_foot_snapped_location(original_pos::Tuple{Float64,Float64})
    lock(DISTANCE_STORE_LOCK) do
        return get(DISTANCE_STORE.foot_snapped_locations, original_pos, original_pos)
    end
end

"""
스냅된 위치 조회 (기본: foot 프로파일, 호환성 유지)
"""
function get_snapped_location(original_pos::Tuple{Float64,Float64})
    return get_foot_snapped_location(original_pos)
end

"""
거리 행렬 초기화 여부 확인
"""
function is_distance_matrix_initialized()
    lock(DISTANCE_STORE_LOCK) do
        return DISTANCE_STORE.initialized
    end
end

"""
거리 행렬 초기화 해제
"""
function reset_distance_matrix()
    lock(DISTANCE_STORE_LOCK) do
        DISTANCE_STORE.car_matrix = Matrix{Float64}(undef, 0, 0)
        DISTANCE_STORE.car_time_matrix = Matrix{Float64}(undef, 0, 0)
        DISTANCE_STORE.foot_matrix = Matrix{Float64}(undef, 0, 0)
        DISTANCE_STORE.foot_time_matrix = Matrix{Float64}(undef, 0, 0)
        DISTANCE_STORE.car_node_index = Dict{Tuple{Float64,Float64}, Int}()
        DISTANCE_STORE.foot_node_index = Dict{Tuple{Float64,Float64}, Int}()
        DISTANCE_STORE.car_snapped_locations = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()
        DISTANCE_STORE.foot_snapped_locations = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()
        DISTANCE_STORE.initialized = false
    end
    println("Distance/Time matrix reset (dual-snap mode)")
end

# 모듈 로드 시 자동 테스트 (선택적)
# test_osrm_connection()

