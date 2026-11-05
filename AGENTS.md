# EpiGrid: Autonomous Multi-Agent & Physics-Informed ST-GNN Specification (AGENTS.md)

## 1. System Overview & Mission
EpiGrid is a 5-Phase Physics-Informed Spatio-Temporal Graph Neural Network (PINN-ST-GNN), Hybrid Graph RAG, and Explainable AI (XAI) Command Center for 14-day preemptive dengue defense across all 8 Divisions, 64 Districts, Municipal Wards, and Custom GPS Sectors of Bangladesh.

## 2. Autonomous Agent Pipeline & Orchestration (`FastAPI` + `MCP`)

### Agent 1: Orbital Hydrology & 30m DEM Topography Agent (`services/spatiotemporal_pipeline.py`)
- **Inputs:** Target WGS84 `(lat, lon)`, 650m geodesic 5-subsector mesh (`CORE`, `NE`, `NW`, `SE`, `SW`).
- **Tools & APIs:** ESA Copernicus Sentinel-2 L2A STAC & Process API (`B03`, `B08`, `B11`, `SCL`) + Copernicus 30m DEM via Open-Meteo Elevation API.
- **Physics & Guardrails:** Computes `MNDWI`, `NDWI`, `NDBI`, and `GNDVI`; masks out clouds (`SCL` 0, 3, 8, 9, 10) and permanent open rivers/lakes (`SCL == 6` or `NDWI > 0.34`); derives Topographic Wetness Index (`TWI`) and dynamic sub-sector `drainage_efficiency`.

### Agent 2: Thermodynamic Brière-1 & Ross-Macdonald Kinetics Agent (`services/spatiotemporal_pipeline.py` & `services/st_gnn_engine.py`)
- **Inputs:** Live 14-day NASA POWER agro-climatology (`T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR`).
- **Physics & Guardrails:** Solves daily non-linear Brière-1 thermal reaction norms $R(T) = c \cdot T(T - T_{\min})\sqrt{T_{\max} - T}$ for Viral Extrinsic Incubation Period (`EIP`), Mosquito Development Rate (`MDR`), Gonotrophic Biting Frequency, Vectorial Capacity ($V_c$), and Basic Reproduction Number ($R_0$).

### Agent 3: Geodesic Graph & IDW Spatial Topology Builder (`services/st_gnn_engine.py`)
- **Inputs:** Official Municipal Ward Registry or Custom `LAT/LON` 5-Subsector Geodesic Mesh + live DGHS hospital baseline (`services/dghs_client.py`).
- **Physics & Guardrails:** Applies Shepherd's Inverse-Distance-Weighted (`IDW`) spatial interpolation fused with live Sentinel-2 + 30m DEM drainage efficiency and constructs the Kipf-Welling symmetrically normalized adjacency matrix $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$ ($d \le 4.5\text{ km}$).

### Agent 4: Warm-Start PINN-ST-GNN, MC-Dropout & Autograd XAI Agent (`services/st_gnn_engine.py`)
- **Architecture:** 2-Layer 48-dim Spectral GCN with residual connections + LayerNorm -> GRU over 14-day sequences -> Temporal Self-Attention -> Multi-Horizon Sigmoid Decoder (Day +1 to Day +14).
- **Optimization:** AdamW + CosineAnnealingLR on a Zero-Floor Entropy-Normalized Bregman KL-Divergence + Horizon MSE + Graph Laplacian + PINN Temporal Velocity objective (`models/epigrid_stgnn_weights.pt`).
- **Explainability & Uncertainty:** PyTorch Autograd Input x Gradient Saliency ($|X \cdot \partial\hat{Y}/\partial X|$) across 4 macro pillars + 8-pass stochastic Monte Carlo Dropout for 95% Bayesian confidence intervals (`±CI95%`).

### Agent 5: Hybrid Graph RAG & Self-Reflective Biophysical Guardrail Dispatcher (`services/alert_dispatcher.py`)
- **Retrieval:** 6D dense cosine similarity + keyword boost + 1-hop graph traversal over `WHO_DGHS_KNOWLEDGE_GRAPH` (`WHO-DGHS-SOP-01` through `05`).
- **Guardrail Audit:** Verifies larval metamorphosis viability ($\tau_{\text{retention}}$ vs. $\tau_{\text{emergence}}$), Ross-Macdonald $R_0$, and WHO chemical dosing ceilings (`Temephos 50EC` & `Deltamethrin`) before logging SHA-256 cryptographic dispatches.

## 3. Model Context Protocol (`MCP`) Server Interface
- **Discovery Endpoint:** `GET /mcp` and `GET /mcp/tools`
- **Invocation Endpoint:** `POST /mcp/invoke`
- **Live Technical Whitepaper:** `GET /docs`
