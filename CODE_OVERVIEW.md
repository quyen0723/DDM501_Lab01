# DDM501 Lab 1 — Code Overview

> Báo cáo tổng hợp toàn bộ mã nguồn trong thư mục `ddm501-lab1-complete/`.
> Nguồn: BCC context collector + đọc trực tiếp 15 file (đã verify từng file).

---

## 1. Tổng quan dự án

**Mục đích**: Lab 1 của môn DDM501 — "First ML Product". Xây dựng một dịch vụ dự đoán rating phim (Movie Rating Prediction) production-ready: train mô hình Collaborative Filtering (SVD) trên dataset MovieLens 100K, bọc trong REST API (FastAPI), đóng gói Docker, có test suite pytest.

**Tech stack**:
- Python 3.10+
- **FastAPI** 0.104.1 + **uvicorn** 0.24.0 (REST API / ASGI server)
- **Pydantic** 2.5.2 (validation request/response)
- **scikit-surprise** 1.1.3 (thuật toán SVD — Matrix Factorization)
- **scikit-learn** 1.3.2, **pandas** 2.1.3, **numpy** 1.26.2 (dữ liệu)
- **pytest** 7.4.3 + **pytest-cov** 4.1.0 + **httpx** 0.25.2 (test)
- **Docker** + **docker-compose** (containerization)
- **python-dotenv** 1.0.0 (config)

**Dataset**: MovieLens 100K — 100.000 ratings, 943 users, 1.682 movies.

---

## 2. Cấu trúc thư mục

```
ddm501-lab1-complete/
└── lab1/
    ├── app/
    │   ├── __init__.py        # Package marker, __version__ = "1.0.0"
    │   ├── main.py            # FastAPI app + endpoints
    │   ├── model.py           # Wrapper load/predict model SVD
    │   ├── schemas.py         # Pydantic request/response models
    │   └── config.py          # Config đọc từ env vars
    ├── models/
    │   └── .gitkeep           # Nơi chứa svd_model.pkl (gitignored)
    ├── tests/
    │   ├── __init__.py        # Test package marker
    │   └── test_api.py        # 18 test cases
    ├── scripts/
    │   └── train_model.py     # Train + save SVD model
    ├── Dockerfile
    ├── docker-compose.yml
    ├── pytest.ini
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```

**Lưu ý**: `models/*.pkl` bị `.gitignore` (file model lớn), chỉ giữ `.gitkeep`. Phải chạy `train_model.py` trước khi chạy API/Docker.

---

## 3. Chi tiết từng file

### 3.1 `app/config.py` — Cấu hình
[VERIFIED] `lab1/app/config.py:1-24`

Đọc settings từ environment variables, có giá trị mặc định.

| Hằng số | Nguồn | Mặc định | Vai trò |
|---------|-------|----------|---------|
| `BASE_DIR` | computed | `lab1/` | Thư mục gốc để tính đường dẫn tương đối |
| `MODEL_PATH` | `os.getenv("MODEL_PATH")` | `models/svd_model.pkl` | Đường dẫn file model đã train |
| `MODEL_VERSION` | `os.getenv("MODEL_VERSION")` | `"1.0.0"` | Version model trả về trong response |
| `API_TITLE` | hardcode | `"Movie Rating Prediction API"` | Title Swagger |
| `API_DESCRIPTION` | hardcode | — | Mô tả API |
| `API_VERSION` | hardcode | `"1.0.0"` | Version API |
| `HOST` | `os.getenv("HOST")` | `"0.0.0.0"` | Bind address |
| `PORT` | `int(os.getenv("PORT"))` | `8000` | Port server |
| `DEBUG` | `os.getenv("DEBUG")` | `false` | Flag debug |

### 3.2 `app/schemas.py` — Pydantic models
[VERIFIED] `lab1/app/schemas.py:1-94`

Định nghĩa schema validation cho request/response. Dùng `Field(...)` (bắt buộc) + `json_schema_extra` cho examples Swagger.

| Class | Dùng cho | Trường chính |
|-------|----------|--------------|
| `PredictionRequest` | POST `/predict` body | `user_id: str`, `movie_id: str` (bắt buộc) |
| `PredictionResponse` | response `/predict` | `user_id`, `movie_id`, `predicted_rating` (`ge=1.0, le=5.0`), `model_version` |
| `HealthResponse` | response `/health` | `status: str`, `model_loaded: bool` |
| `PredictionItem` | item trong batch | `user_id: str`, `movie_id: str` |
| `BatchPredictionRequest` | POST `/predict/batch` body | `predictions: List[PredictionItem]` |
| `BatchPredictionResponse` | response batch | `predictions: List[PredictionResponse]`, `total_count: int` |

Lưu ý: `predicted_rating` có ràng buộc `ge=1.0, le=5.0` — ngoài khoảng này FastAPI trả 422. Các class response dùng `model_config = {"protected_namespaces": ()}` để tránh warning Pydantic cho field chứa tiền tố `model_`.

### 3.3 `app/model.py` — Model wrapper
[VERIFIED] `lab1/app/model.py:1-105`

Lớp `MovieRatingModel` bọc model SVD đã pickle.

| Method | Hành vi |
|--------|---------|
| `__init__(model_path=MODEL_PATH)` | Lưu path, gọi `_load_model()` |
| `_load_model()` | `pickle.load` từ `model_path`. `FileNotFoundError` → log + re-raise; lỗi khác → log + raise |
| `predict(user_id, movie_id) -> float` | Gọi `self.model.predict(...)`, trả `round(float(prediction.est), 2)`. Model `None` → `RuntimeError` |
| `predict_batch(pairs) -> List[float]` | List comprehension gọi `predict` cho từng cặp |
| `is_loaded() -> bool` | `self.model is not None` |

Module còn có **singleton pattern** (tuỳ chọn): `_model_instance` + `get_model()` — lazy khởi tạo một instance duy nhất. Tuy nhiên `main.py` không dùng `get_model()` mà tự quản lý `model` global.

SVD xử lý unknown user/movie bằng cách fallback về global mean rating → vẫn trả prediction hợp lệ (không raise).

### 3.4 `app/main.py` — FastAPI application
[VERIFIED] `lab1/app/main.py:1-178`

Khởi tạo app + định nghĩa endpoints. CORS mở toàn bộ (`allow_origins=["*"]`).

**Lifecycle**: `@app.on_event("startup")` tạo `MovieRatingModel()` gán vào `model` global. Nếu fail → log error, `model` stays `None` → health check báo unhealthy.

**Endpoints**:

| Method | Path | response_model | Hành vi |
|--------|------|----------------|---------|
| GET | `/` | — | Trả `{name, version, description, docs, health}` |
| GET | `/health` | `HealthResponse` | `status="healthy"` nếu `model.is_loaded()`, kèm `model_loaded` bool |
| GET | `/model/info` | — | `{model_version, model_type: "SVD (Collaborative Filtering)", is_loaded}` |
| POST | `/predict` | `PredictionResponse` | Check model loaded (503 nếu không) → `model.predict()` → trả rating + `MODEL_VERSION`. Lỗi predict → 500 |
| POST | `/predict/batch` | `BatchPredictionResponse` | Loop qua `request.predictions`, predict từng item, trả list + `total_count` |

**Error handling**:
- Model `None`/chưa load → HTTP **503** "Model not loaded"
- Exception trong predict → HTTP **500** + `str(e)`
- Validation Pydantic (thiếu field, sai type, body rỗng) → HTTP **422** (tự động bởi FastAPI)

Comment trong file đánh dấu `/health` là "PROVIDED - DO NOT MODIFY", `/predict` là "TODO 1", `/predict/batch` là "TODO 2 (BONUS)" → đây là lab có scaffold, học sinh implement TODO.

### 3.5 `app/__init__.py`
[VERIFIED] `lab1/app/__init__.py:1-7` — Docstring + `__version__ = "1.0.0"`.

### 3.6 `scripts/train_model.py` — Training pipeline
[VERIFIED] `lab1/scripts/train_model.py:1-119`

Hàm `main()` chạy pipeline 4 bước:

1. **Load data**: `Dataset.load_builtin('ml-100k')` — surprise tự download MovieLens 100K.
2. **Cross-validation**: Tạo `SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)`, chạy `cross_validate(model, data, measures=['RMSE','MAE'], cv=5, verbose=True)`. In mean RMSE/MAE (≈0.93 / ≈0.74).
3. **Train full**: `data.build_full_trainset()` → `model.fit(trainset)`.
4. **Save**: `pickle.dump(model)` → `models/svd_model.pkl`. Tự tạo `models/` nếu thiếu.

Cuối cùng test sample prediction (user "196", movie "242") rồi in next steps.

**Hyperparameters**: 100 latent factors, 20 epochs, lr=0.005, reg=0.02.

**Dead imports**: `train_test_split` và `accuracy` (dòng 19-20) được import nhưng không dùng trong script — chỉ `cross_validate` thực sự được gọi. Code smell nhỏ.

### 3.7 `tests/test_api.py` — Test suite (18 tests)
[VERIFIED] `lab1/tests/test_api.py:1-251`

Dùng `TestClient(app)` của FastAPI. **Quan trọng**: fixture `run_app_lifespan` (scope session, autouse) dùng `with client:` để trigger `@app.on_event("startup")` — nếu không, model không load khi test.

| Class test | Số test | Cases |
|------------|--------|-------|
| `TestHealthEndpoint` | 2 | 200 + format response |
| `TestRootEndpoint` | 2 | 200 + chứa api info |
| `TestPredictEndpoint` | 5 | valid input, response format, missing user_id (422), missing movie_id (422), empty body (422) |
| `TestEdgeCases` | 4 | unknown user (200, fallback mean), unknown movie (200), special chars in ID (200), wrong type (422) |
| `TestModelInfoEndpoint` | 2 | 200 + chứa version/type/is_loaded |
| `TestBatchPredictEndpoint` | 3 | multiple items, empty list (total_count=0), missing field (422) |

### 3.8 `tests/__init__.py`
Docstring rỗng — package marker.

### 3.9 `Dockerfile`
[VERIFIED] `lab1/Dockerfile:1-51`

- Base: `python:3.10-slim`
- Cài system deps: `build-essential` (compile scikit-surprise C extensions) + `curl` (cho HEALTHCHECK)
- Copy `requirements.txt` trước → `pip install` (tận dụng Docker cache layer)
- Copy app code (`COPY . .`)
- `EXPOSE 8000`
- `HEALTHCHECK` 30s interval, 10s timeout, 5s start-period, 3 retries → `curl -f http://localhost:8000/health`
- `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`

### 3.10 `docker-compose.yml`
[VERIFIED] `lab1/docker-compose.yml:1-39`

Service `api`:
- `build: .` (dùng Dockerfile)
- Port `8000:8000`
- Volume mount `./models:/app/models` — model được train trên host rồi mount vào container (không nạp model vào image)
- Env: `MODEL_PATH=/app/models/svd_model.pkl`, `MODEL_VERSION=1.0.0`
- `restart: unless-stopped`

Comment sẵn (chưa enable): redis + prometheus (BONUS monitoring).

### 3.11 `requirements.txt`
Pinned versions cho tất cả dependency (xem mục 1).

### 3.12 `pytest.ini`
- `testpaths = tests`, pattern `test_*.py` / `Test*` / `test_*`
- `addopts = -v --tb=short`
- `filterwarnings = ignore::DeprecationWarning` (ẩn warning từ scikit-surprise/Pydantic)

### 3.13 `.gitignore`
Bỏ `__pycache__`, venv, dist, IDE, `.pytest_cache`, `.coverage`, `htmlcov`, **`models/*.pkl`** (giữ `.gitkeep`), `data/`, `*.csv`, logs, `.docker`, OS files.

### 3.14 `README.md`
Tài liệu đầy đủ: features, structure, quick start (setup → train → run → test → Docker), bảng endpoints, error responses, config vars, model details.

---

## 4. Dataflow & dependencies giữa các file

```
config.py ──(MODEL_PATH, MODEL_VERSION, API_*)──┐
                                               ├──> main.py (FastAPI app)
schemas.py ──(PredictionRequest/Response, ...)─┤       │
                                               │       │ startup: MovieRatingModel()
model.py ──(MovieRatingModel)──────────────────┘       │
   │                                                  │ predict()
   └── pickle.load(MODEL_PATH) <── models/svd_model.pkl
                                       ▲
                                       │ pickle.dump
                            scripts/train_model.py
                            (surprise SVD, ml-100k)

tests/test_api.py ──TestClient──> app.main:app

Dockerfile ──build──> image ──run──> uvicorn app.main:app
docker-compose.yml ──mount ./models──> container /app/models
```

**Luồng runtime**:
1. Startup → `main.py` khởi tạo `MovieRatingModel(MODEL_PATH)` → `model.py._load_model()` pickle-load `models/svd_model.pkl`.
2. Request `/predict` → validate bằng `schemas.PredictionRequest` → `model.predict()` → `surprise.SVD.predict()` → `round(est, 2)` → wrap `PredictionResponse`.
3. Health check đọc `model.is_loaded()` để report trạng thái.
4. Docker `HEALTHCHECK` gọi `/health`; compose mount `./models` để container dùng model train trên host.

---

## 5. Cách chạy

```bash
# 1. Cài deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train model (tạo models/svd_model.pkl)
python scripts/train_model.py

# 3. Chạy API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs

# 4. Test
pytest tests/ -v
pytest tests/ -v --cov=app --cov-report=html

# 5. Docker
docker-compose build && docker-compose up -d
curl http://localhost:8000/health
```

---

## 6. Ghi chú kiến trúc

- **Scaffold lab**: comment `TODO 1/2/3` và `PROVIDED - DO NOT MODIFY` cho thấy đây là lab có template, học sinh implement các TODO trong `model.py`, `main.py`, `schemas.py`, `Dockerfile`.
- **Model lazy + resilient**: model load fail không crash app — health check báo unhealthy, endpoint trả 503.
- **SVD fallback**: unknown user/movie → global mean (không error) → API vẫn trả 200.
- **Tách biệt model & image**: model không nạp vào Docker image mà mount volume → train 1 lần trên host, dùng cho nhiều container.
- **Test phải trigger lifespan**: fixture `with client:` là chi tiết quan trọng — không có thì model `None` và mọi test predict fail.
- **Bonus features**: batch prediction, healthcheck Docker, và sẵn sàng cho redis/prometheus monitoring.

---

*Báo cáo sinh từ BCC (`bcc_basic_context_collector`, engine opencode/glm-5.2) + verify trực tiếp 15 file nguồn tại `ddm501-lab1-complete/lab1/`.*