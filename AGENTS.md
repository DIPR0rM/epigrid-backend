# EpiGrid: Autonomous Multi-Agent & Physics-Informed ST-GNN Specification

## 1. System Overview & Mission
EpiGrid (v7.0) is a 5-Phase Physics-Informed Spatio-Temporal Graph Neural Network (`PINN-ST-GNN`), Hybrid Graph RAG, and Explainable AI (`XAI`) Command Center for 14-day preemptive dengue defense across all 8 Divisions, 64 Districts, Municipal Wards (including all 129 Dhaka DNCC/DSCC wards), and Custom GPS Sectors of Bangladesh.

## 2. Autonomous Agent Pipeline & Orchestration (`FastAPI` + `MCP`)

### Agent 1: Orbital Hydrology, 30m DEM Ring & Shoreline-Safe Water Exclusion Agent (`services/spatiotemporal_pipeline.py`)
- **Inputs:** Target WGS84 `(lat, lon)`, 650m geodesic 5-subsector mesh (`CORE`, `NE`, `NW`, `SE`, `SW`).
- **Tools & APIs:** ESA Copernicus Sentinel-2 L2A STAC & Process API (`B03`, `B08`, `B11`, `SCL`) + Copernicus 30m DEM via Open-Meteo Elevation API (25-point query: center + 150m N/S/E/W ring per sub-sector) + OpenStreetMap (`OSM`) Overpass API water-body geometry.
- **Physics & Guardrails:**
  - Computes `MNDWI`, `NDWI`, `NDBI`, and `GNDVI`, masking out clouds (`SCL` 0, 3, 8, 9, 10).
  - Enforces a **3-Layer Shoreline-Safe Permanent Water Exclusion Mask**: (1) Sentinel-2 `SCL == 6` and turbid lake thresholds (`NDWI > 0.22 & NDBI < -0.16`), (2) live OSM Overpass water ways (`around:12m`) + major river/lake centerlines (Hatirjheel, Padma, Buriganga, Turag, Balu, Dhanmondi/Gulshan Lakes), and (3) a tight $3\times 3$ center-pixel pin check (`>= 67%`) so dry-land streets adjacent to lakes evaluate strictly on dry-land pixels.
  - Computes coordinate-shift-invariant **Topographic Wetness Index (`TWI`)**, **USDA-NRCS Runoff Curve Number ($CN \in [52, 96]$)**, and an **Urban Tree-Canopy Regime Switch** (recognizing high `GNDVI` in dense urban cores $I_{\text{urb}} \ge 0.56$ as shade-producing canopy rather than rural pervious soil).

### Agent 2: Thermodynamic Brière-1 & Ross-Macdonald Kinetics Agent (`services/spatiotemporal_pipeline.py` & `services/st_gnn_engine.py`)
- **Inputs:** Live 14-day NASA POWER agro-climatology (`T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR`).
- **Physics & Guardrails:** Solves daily non-linear Brière-1 thermal reaction norms $R(T_t) = c \cdot T_t(T_t - T_{\min})\sqrt{T_{\max} - T_t}$ across all 14 daily steps (correcting for diurnal Jensen's inequality) to derive Viral Extrinsic Incubation Period (`EIP`), Mosquito Development Rate (`MDR`), Gonotrophic Biting Frequency ($a$), Ross-Macdonald Vectorial Capacity ($V_c$), and Basic Reproduction Number ($R_0$).

### Agent 3: Autonomous Coordinate Hydrology & Geodesic Graph Topology Agent (`services/st_gnn_engine.py`)
- **Inputs:** Pure Coordinate Municipal Ward Registry or Custom `LAT/LON` 5-Subsector Geodesic Mesh + live DGHS hospital baseline (`services/dghs_client.py`).
- **Physics & Guardrails:**
  - Executes `compute_autonomous_coordinate_hydrology(lat, lon)` to derive `drainage_efficiency`, `retention_multiplier`, `scs_curve_number_cn`, `topographic_wetness_index`, `elevation_m`, and `default_structure` strictly from WGS84 coordinates without manual lookup tables.
  - Guarantees **100% cross-mode consistency** (Full-Ward Mode, Single-Ward 5-Subsector Mode, and Custom `LAT/LON` Mode produce identical physical hydrology at the same coordinate) and **100% coordinate shift-invariance**.
  - Constructs the Kipf-Welling symmetrically normalized spatial adjacency matrix $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$ for mosquito dispersal corridors ($d \le 4.5\text{ km}$, $\sigma = 1.6\text{ km}$).

### Agent 4: Warm-Start PINN-ST-GNN, MC-Dropout & Autograd XAI Agent (`services/st_gnn_engine.py`)
- **Architecture:** 2-Layer 48-dim Spectral GCN with skip-residual projections + `LayerNorm` -> `GRU` over 14-day sequences -> Temporal Self-Attention -> Multi-Horizon Sigmoid Decoder (`Day +1` to `Day +14`).
- **Optimization & Transfer Memory (`TRANSFER_SCHEMA_VERSION = 7`):** Trains via `AdamW` + 8-epoch linear warmup + cosine annealing on a **Zero-Floor Entropy-Normalized Bregman KL-Divergence + Horizon MSE + Graph Laplacian + PINN Temporal Velocity Objective**. Persists weights in `models/epigrid_stgnn_weights.pt` (retaining 85% weights intra-district and 72% encoder / 20% readout head cross-district) with a **Warm-Start Fast-Convergence Early-Stopping Gate** ($\mathcal{L}_{\text{total}} \le 0.00018$).
- **Explainability & Bayesian Uncertainty:** Computes PyTorch Autograd Input $\times$ Gradient Saliency ($\left|X \cdot \partial\hat{Y}/\partial X\right|$) summing to $100.0\%$ across 4 macro pillars, extracts Peak GRU Temporal Attention lag dates, and runs 8 stochastic Monte Carlo Dropout passes (`p=0.08`) for 95% confidence intervals (`±CI95%`).

### Agent 5: Hybrid Graph RAG, Biophysical Guardrail & AI Co-Pilot Dispatcher (`services/alert_dispatcher.py`)
- **Retrieval:** Encodes each sector into a 6D normalized state vector and executes dense cosine similarity + keyword boost + 1-hop graph traversal over `WHO_DGHS_KNOWLEDGE_GRAPH` (`WHO-DGHS-SOP-01` through `05`).
- **Self-Reflective Guardrail Audit:** Verifies larval metamorphosis viability ($\tau_{\text{retention}} \ge 0.88\,\tau_{\text{aquatic}}$), Ross-Macdonald $R_0$, permanent water-body exclusion, and WHO chemical dosing ceilings (`Temephos 50EC` $\le 14\text{ L}$ & `Deltamethrin` $\le 10\text{ L}$) before logging SHA-256 cryptographic dispatches and generating the **Executive AI Co-Pilot Situation Briefing**.

## 3. Model Context Protocol (`MCP`) & Live Documentation Interface
- **Interactive Geospatial Command Center:** `GET /dashboard`
- **Live Technical Whitepaper & Mathematical Proof Hub:** `GET /docs`
- **OpenAPI / Swagger Testing Console:** `GET /swagger`
- **MCP Discovery & Invocation Endpoints:** `GET /mcp`, `GET /mcp/tools`, and `POST /mcp/invoke`
- **Automated Regression & Guardrail Benchmark:** `GET /api/v1/ai/evaluate-benchmark` (`tests/eval_benchmark.py`)
